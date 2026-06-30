package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
)

const featurePlanesV1 = 5

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
