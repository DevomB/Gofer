//go:build js && wasm

package main

import "math"

func mathFloat32bits(f float32) uint32     { return math.Float32bits(f) }
func mathFloat32frombits(u uint32) float32 { return math.Float32frombits(u) }
