package main

import (
	"math/rand"
)

const maxPoints = 19 * 19

var zobristTable [maxPoints][3]uint64

func init() {
	rng := rand.New(rand.NewSource(0x60eef0bacafe))
	for i := 0; i < maxPoints; i++ {
		for c := Empty; c <= White; c++ {
			zobristTable[i][c] = rng.Uint64()
		}
	}
}
