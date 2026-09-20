package main

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"syscall"
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

	// maxShardBytes bounds the in-memory build of one shard. A cycle that needs
	// more than this is a configuration mistake, not a workload: at 9x9 it is
	// millions of rows, and at 19x19 it still allows a quarter of a million.
	maxShardBytes = 2 << 30 // 2 GiB
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

// writeFileAtomic writes data to a sibling temp file and renames it into place.
//
// The Python orchestrator caches hot-path artifacts by existence: run_match in
// training/pipeline/runner.py reuses any arena report it finds instead of
// replaying the match. A truncated file left behind by a kill mid-write is
// therefore permanent — every resume skips the match and then fails to parse
// the report. Renaming last means a reader sees either the previous file or
// the complete new one.
func writeFileAtomic(path string, data []byte, perm os.FileMode) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, perm); err != nil {
		return err
	}
	if err := renameWithRetry(tmp, path); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}

// renameRetries and renameBackoff mirror write_json_atomic in
// training/pipeline/state.py, which already retries for the same reason: on
// Windows an indexer or AV scanner briefly holds the destination open and
// MoveFileEx fails. Retrying beats failing a whole self-play or gating stage.
const (
	renameRetries = 10
	renameBackoff = 50 * time.Millisecond
)

func renameWithRetry(oldpath, newpath string) error {
	return retryRename(os.Rename, isTransientRenameErr, oldpath, newpath, renameRetries, renameBackoff)
}

// retryRename takes the rename and the predicate as arguments so the backoff
// loop is testable without provoking a real sharing violation.
func retryRename(rename func(string, string) error, transient func(error) bool,
	oldpath, newpath string, attempts int, backoff time.Duration) error {
	var err error
	for attempt := 0; attempt < attempts; attempt++ {
		if err = rename(oldpath, newpath); err == nil {
			return nil
		}
		if !transient(err) {
			return err
		}
		if attempt < attempts-1 {
			time.Sleep(backoff * time.Duration(attempt+1))
		}
	}
	return err
}

// isTransientRenameErr reports whether a failed rename is worth retrying. Only
// Windows produces these: another handle on the destination yields a sharing,
// lock or access error that clears on its own within milliseconds. Elsewhere a
// rename failure is real, and retrying would only delay reporting it.
func isTransientRenameErr(err error) bool {
	if err == nil || runtime.GOOS != "windows" {
		return false
	}
	var errno syscall.Errno
	if !errors.As(err, &errno) {
		return false
	}
	switch uintptr(errno) {
	case 5, 32, 33: // ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION
		return true
	}
	return false
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
	return renameWithRetry(tmp, path)
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

	// Every column is materialised in memory before the shard is written, so an
	// oversized cycle would otherwise be killed by the OS with nothing to point
	// at. Fail here instead, naming the numbers that produced it. This also
	// keeps the capacities below from overflowing int on 32-bit builds, where
	// n*planeLen wraps silently.
	perRow := int64(planeLen) + int64(points) + 1 + int64(globalsLen)*4 + int64(pol)*8 + 14
	if total := int64(n) * perRow; total > maxShardBytes {
		return nil, 0, 0, fmt.Errorf(
			"shard: %d rows of %d planes on %dx%d needs %.1f GiB in memory (limit %d GiB); "+
				"lower selfplay.games_per_cycle or split the cycle",
			n, planes, size, size, float64(total)/(1<<30), maxShardBytes>>30)
	}

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
		var err error
		if globals, err = appendF32(globals, s.FeaturesGlobal, i, "features_global"); err != nil {
			return nil, 0, 0, err
		}
		if policy, err = appendF32(policy, s.Policy, i, "policy"); err != nil {
			return nil, 0, 0, err
		}
		next := s.PolicyNext
		if len(next) != pol {
			next = zeroPolicy
		}
		if policyNext, err = appendF32(policyNext, next, i, "policy_opp"); err != nil {
			return nil, 0, 0, err
		}
		if value, err = appendF32(value, []float32{s.Value}, i, "value"); err != nil {
			return nil, 0, 0, err
		}
		if score, err = appendF32(score, []float32{s.ScoreMargin}, i, "score"); err != nil {
			return nil, 0, 0, err
		}
		// Ownership is required on every row. The learner applies its ownership
		// loss to all rows unmasked (training/gofer_train/losses.py), so a
		// zero-filled row is indistinguishable from a genuinely neutral board
		// and would train the head toward "neutral everywhere". Self-play always
		// labels it (labelGameSamples); -convert-sgf writes JSONL, not shards.
		if len(s.Ownership) != points {
			return nil, 0, 0, fmt.Errorf("shard: sample %d has %d ownership labels, want %d; shard v1 requires ownership on every row", i, len(s.Ownership), points)
		}
		// Ownership labels are absolute (Black=+1); flip to the side-to-move frame
		// to match value, score, and the own/opp input planes.
		sign := float32(1)
		if s.ToPlay == White {
			sign = -1
		}
		for p := 0; p < points; p++ {
			ownership = append(ownership, byte(int8(clampOwnership(s.Ownership[p]*sign))))
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

// appendF32 encodes a float32 column, rejecting NaN and Inf. The spatial planes
// are already validated value-by-value above; the float columns deserve the same
// guard, because one non-finite label silently poisons a whole training batch's
// loss with nothing on disk to trace it back to.
func appendF32(dst []byte, vals []float32, sample int, field string) ([]byte, error) {
	for _, v := range vals {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("shard: sample %d has non-finite %s value %v; shard v1 stores finite float32 columns", sample, field, v)
		}
		dst = binary.LittleEndian.AppendUint32(dst, math.Float32bits(v))
	}
	return dst, nil
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
