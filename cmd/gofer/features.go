package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
)

const (
	FeatureSchemaVersion = 2
	featurePlanesV1      = 5
	featurePlanesV2      = 8
	featureGlobalsV2     = 4
)

// FeatureSchemaV1 describes tensor layout for NN input (simplified paper subset).
type FeatureSchemaV1 struct {
	BoardSize int `json:"board_size"`
	Planes    int `json:"planes"`
	Globals   int `json:"globals"`
}

// BuildFeaturesV1 encodes board state as NCHW-flat planes + globals.
// Planes: own stones, opponent stones, empty, ko point, player-to-move marker.
func BuildFeaturesV1(b *Board) []float32 {
	size := b.Size()
	n := size * size
	planes := featurePlanesV1
	out := make([]float32, planes*n+2)
	player := b.Player()
	opp := player.Opposite()
	ko := b.Ko()
	for i := 0; i < n; i++ {
		switch b.AtIndex(i) {
		case player:
			out[i] = 1
		case opp:
			out[n+i] = 1
		case Empty:
			out[2*n+i] = 1
		}
		if i == ko {
			out[3*n+i] = 1
		}
	}
	if player == Black {
		out[planes*n] = 1
	} else {
		out[planes*n+1] = 1
	}
	return out
}

// BuildFeaturesV2 encodes board state for bootstrap ONNX (schema version 2).
// Planes: own, opp, empty, ko, to-move fill; history t-3, t-2, t-1.
// Globals: komi/10, move_num/(size²+1), black-to-move, white-to-move.
func BuildFeaturesV2(b *Board) (spatial []float32, globals []float32) {
	size := b.Size()
	n := size * size
	planes := featurePlanesV2
	spatial = make([]float32, planes*n)
	player := b.Player()
	opp := player.Opposite()
	ko := b.Ko()
	for i := 0; i < n; i++ {
		switch b.AtIndex(i) {
		case player:
			spatial[i] = 1
		case opp:
			spatial[n+i] = 1
		case Empty:
			spatial[2*n+i] = 1
		}
		if i == ko {
			spatial[3*n+i] = 1
		}
	}
	tm := float32(0)
	if player == Black {
		tm = 1
	}
	for i := 0; i < n; i++ {
		spatial[4*n+i] = tm
	}
	hist := b.historyMoves(3)
	for h := 0; h < len(hist); h++ {
		snap := hist[h]
		if snap.move.Pass {
			continue
		}
		idx := snap.move.Point.Idx(size)
		if idx >= 0 {
			spatial[(5+h)*n+idx] = 1
		}
	}
	denom := float32(size*size + 1)
	globals = []float32{
		float32(b.Komi()) / 10,
		float32(b.MoveNum()) / denom,
	}
	if player == Black {
		globals = append(globals, 1, 0)
	} else {
		globals = append(globals, 0, 1)
	}
	return spatial, globals
}

// FeatureHashV2 hashes spatial + globals for golden tests.
func FeatureHashV2(spatial, globals []float32) string {
	h := sha256.New()
	var buf [4]byte
	for _, v := range spatial {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
		h.Write(buf[:])
	}
	for _, v := range globals {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
		h.Write(buf[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

// FeatureHash returns stable hex hash for golden tests.
func FeatureHash(f []float32) string {
	h := sha256.New()
	var buf [4]byte
	for _, v := range f {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
		h.Write(buf[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}
