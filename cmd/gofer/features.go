package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
)

const (
	// FeatureSchemaVersion 3 fixes the history planes, which were left-aligned:
	// with fewer than three moves played the most recent one landed in t-3 or
	// t-2 and the t-1 plane stayed empty, which is how a pass is encoded. The
	// plane count and shapes are unchanged, so a v2 model loads against v3
	// features and silently misreads the opening. Anything trained on v2 data
	// has to be retrained, not reused.
	FeatureSchemaVersion = 3
	featurePlanesV1      = 5
	featurePlanesV2      = 8
	featureGlobalsV2     = 4
	// Planes 5, 6, 7 are t-3, t-2, t-1 (docs/model-input-schema.md).
	firstHistoryPlane = 5
	historyPlanes     = 3
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
	fillStonePlanes(b, spatial, player, n)
	fillToMovePlane(spatial, player, n)
	fillHistoryPlanes(b, spatial, n)
	return spatial, buildGlobalFeatures(b, player, size)
}

func fillStonePlanes(b *Board, spatial []float32, player Color, n int) {
	opp := player.Opposite()
	for i := 0; i < n; i++ {
		switch b.AtIndex(i) {
		case player:
			spatial[i] = 1
		case opp:
			spatial[n+i] = 1
		case Empty:
			spatial[2*n+i] = 1
		}
		if i == b.Ko() {
			spatial[3*n+i] = 1
		}
	}
}

func fillToMovePlane(spatial []float32, player Color, n int) {
	if player != Black {
		return
	}
	for i := 0; i < n; i++ {
		spatial[4*n+i] = 1
	}
}

// fillHistoryPlanes writes the last three moves into planes 5, 6, 7 as t-3,
// t-2, t-1 (docs/model-input-schema.md).
//
// The history is RIGHT-aligned: the most recent move is always t-1, however
// few moves have been played. Left-aligning it -- writing historyMoves(3)[0]
// into plane 5 unconditionally -- put the only move of a one-move game into
// t-3 and left t-1 empty, and two moves in put the latest into t-2. Only from
// move three on did t-1 mean what the schema says.
//
// A pass is encoded by writing nothing, so an empty t-1 plane is how the
// network is told "the opponent just passed". Under the old layout a real
// opening move produced exactly that, which made a played move and a pass
// indistinguishable for the first two plies of every game -- in the opening,
// where a network is learning whether passing is reasonable.
func fillHistoryPlanes(b *Board, spatial []float32, n int) {
	hist := b.historyMoves(historyPlanes)
	offset := historyPlanes - len(hist)
	for h, snap := range hist {
		if snap.move.Pass {
			continue
		}
		idx := snap.move.Point.Idx(b.Size())
		if idx >= 0 {
			spatial[(firstHistoryPlane+offset+h)*n+idx] = 1
		}
	}
}

func buildGlobalFeatures(b *Board, player Color, size int) []float32 {
	denom := float32(size*size + 1)
	globals := []float32{
		float32(b.Komi()) / 10,
		float32(b.MoveNum()) / denom,
	}
	if player == Black {
		globals = append(globals, 1, 0)
	} else {
		globals = append(globals, 0, 1)
	}
	return globals
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
