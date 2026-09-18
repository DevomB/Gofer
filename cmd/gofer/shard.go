package main

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"time"
)

// Gofer shard v1: a NumPy .npz (deflated ZIP archive of .npy arrays) holding one
// self-play run at a single board size. It replaces JSONL on the hot path: the
// binary feature planes are stored as uint8 and every column is loadable with a
// single np.load, so the trainer never parses JSON floats. Layout is the
// contract in docs/training-data-format.md; keep them in sync.
const (
	ShardFormat  = "gofer-shard"
	ShardVersion = 1
)

// ShardMeta is written into the shard's "meta" array as UTF-8 JSON bytes.
type ShardMeta struct {
	Format    string  `json:"format"`
	Version   int     `json:"version"`
	BoardSize int     `json:"board_size"`
	Rows      int     `json:"rows"`
	Games     int     `json:"games"`
	GitCommit string  `json:"git_commit,omitempty"`
	Model     string  `json:"model,omitempty"`
	Komi      float64 `json:"komi"`
	Seed      int64   `json:"seed"`
	CreatedAt string  `json:"created_at"`
}

func isShardPath(path string) bool {
	return strings.HasSuffix(strings.ToLower(path), ".npz")
}

// npyArray is one named column: dtype descr, shape, and little-endian payload.
type npyArray struct {
	name  string
	descr string
	shape []int
	data  []byte
}

// WriteSampleShard writes samples as a gofer shard. All samples must share one
// board size and carry exported features; ownership and policy_next are
// converted to the side-to-move frame so every column shares one perspective.
func WriteSampleShard(path string, samples []Sample, meta ShardMeta) error {
	arrays, size, games, err := buildShardArrays(samples)
	if err != nil {
		return err
	}
	meta.Format = ShardFormat
	meta.Version = ShardVersion
	meta.BoardSize = size
	meta.Rows = len(samples)
	meta.Games = games
	if meta.GitCommit == "" {
		meta.GitCommit = buildInfoVersion()
	}
	if meta.CreatedAt == "" {
		meta.CreatedAt = time.Now().UTC().Format(time.RFC3339)
	}
	metaJSON, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	arrays = append(arrays, npyArray{name: "meta", descr: "|u1", shape: []int{len(metaJSON)}, data: metaJSON})

	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if err := writeNPZ(f, arrays); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	// Rename last so a crashed run never leaves a truncated shard for the trainer.
	return os.Rename(tmp, path)
}

func buildShardArrays(samples []Sample) ([]npyArray, int, int, error) {
	n := len(samples)
	if n == 0 {
		return nil, 0, 0, fmt.Errorf("shard: no samples")
	}
	planeLen := len(samples[0].FeaturesSpatial)
	pol := len(samples[0].Policy)
	size := int(math.Round(math.Sqrt(float64(pol - 1))))
	points := size * size
	if size < 2 || points+1 != pol || planeLen == 0 || planeLen%points != 0 {
		return nil, 0, 0, fmt.Errorf("shard: bad first sample (policy=%d spatial=%d)", pol, planeLen)
	}
	planes := planeLen / points
	globalsLen := len(samples[0].FeaturesGlobal)

	spatial := make([]byte, 0, n*planeLen)
	globals := make([]byte, 0, n*globalsLen*4)
	policy := make([]byte, 0, n*pol*4)
	policyNext := make([]byte, 0, n*pol*4)
	value := make([]byte, 0, n*4)
	score := make([]byte, 0, n*4)
	ownership := make([]byte, 0, n*points)
	full := make([]byte, 0, n)
	gameID := make([]byte, 0, n*4)
	moveNum := make([]byte, 0, n*2)
	zeroPolicy := make([]float32, pol)
	gameSet := map[int]struct{}{}

	for i, s := range samples {
		if len(s.FeaturesSpatial) != planeLen || len(s.Policy) != pol || len(s.FeaturesGlobal) != globalsLen {
			return nil, 0, 0, fmt.Errorf("shard: sample %d shape differs from sample 0 (mixed board sizes?)", i)
		}
		for _, v := range s.FeaturesSpatial {
			if v != 0 && v != 1 {
				return nil, 0, 0, fmt.Errorf("shard: sample %d has non-binary feature %v; shard v1 stores planes as uint8", i, v)
			}
			spatial = append(spatial, byte(v))
		}
		globals = appendF32(globals, s.FeaturesGlobal)
		policy = appendF32(policy, s.Policy)
		if len(s.PolicyNext) == pol {
			policyNext = appendF32(policyNext, s.PolicyNext)
		} else {
			policyNext = appendF32(policyNext, zeroPolicy)
		}
		value = appendF32(value, []float32{s.Value})
		score = appendF32(score, []float32{s.ScoreMargin})
		// Ownership labels are absolute (Black=+1); flip to the side-to-move frame
		// to match value, score, and the own/opp input planes.
		sign := float32(1)
		if s.ToPlay == White {
			sign = -1
		}
		for p := 0; p < points; p++ {
			var o float32
			if len(s.Ownership) == points {
				o = s.Ownership[p] * sign
			}
			ownership = append(ownership, byte(int8(clampOwnership(o))))
		}
		if s.FullSearch {
			full = append(full, 1)
		} else {
			full = append(full, 0)
		}
		gameID = binary.LittleEndian.AppendUint32(gameID, uint32(int32(s.GameID)))
		moveNum = binary.LittleEndian.AppendUint16(moveNum, uint16(int16(s.MoveNum)))
		gameSet[s.GameID] = struct{}{}
	}

	arrays := []npyArray{
		{"spatial", "|u1", []int{n, planes, size, size}, spatial},
		{"globals", "<f4", []int{n, globalsLen}, globals},
		{"policy", "<f4", []int{n, pol}, policy},
		{"policy_opp", "<f4", []int{n, pol}, policyNext},
		{"value", "<f4", []int{n}, value},
		{"score", "<f4", []int{n}, score},
		{"ownership", "|i1", []int{n, points}, ownership},
		{"full_search", "|u1", []int{n}, full},
		{"game_id", "<i4", []int{n}, gameID},
		{"move_num", "<i2", []int{n}, moveNum},
	}
	return arrays, size, len(gameSet), nil
}

func clampOwnership(o float32) int {
	switch {
	case o > 0.5:
		return 1
	case o < -0.5:
		return -1
	default:
		return 0
	}
}

func appendF32(dst []byte, vals []float32) []byte {
	for _, v := range vals {
		dst = binary.LittleEndian.AppendUint32(dst, math.Float32bits(v))
	}
	return dst
}

func writeNPZ(f *os.File, arrays []npyArray) error {
	zw := zip.NewWriter(f)
	for _, a := range arrays {
		w, err := zw.CreateHeader(&zip.FileHeader{Name: a.name + ".npy", Method: zip.Deflate})
		if err != nil {
			return err
		}
		if _, err := w.Write(npyHeader(a.descr, a.shape)); err != nil {
			return err
		}
		if _, err := w.Write(a.data); err != nil {
			return err
		}
	}
	return zw.Close()
}

// npyHeader returns a NPY v1.0 header; total header length is padded to a
// multiple of 64 bytes as numpy requires for aligned memory mapping.
func npyHeader(descr string, shape []int) []byte {
	dims := make([]string, len(shape))
	for i, d := range shape {
		dims[i] = fmt.Sprint(d)
	}
	shapeStr := strings.Join(dims, ", ")
	if len(shape) == 1 {
		shapeStr += ","
	}
	dict := fmt.Sprintf("{'descr': '%s', 'fortran_order': False, 'shape': (%s), }", descr, shapeStr)
	const prefix = 10 // magic(6) + version(2) + header_len(2)
	pad := 64 - (prefix+len(dict)+1)%64
	if pad == 64 {
		pad = 0
	}
	dict += strings.Repeat(" ", pad) + "\n"
	var buf bytes.Buffer
	buf.WriteString("\x93NUMPY")
	buf.Write([]byte{1, 0})
	binary.Write(&buf, binary.LittleEndian, uint16(len(dict)))
	buf.WriteString(dict)
	return buf.Bytes()
}
