package main

import (
	"math"
	"math/rand"
	"testing"
)

func TestSampleDirichletIsSimplex(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range []int{2, 5, 82} {
		d := sampleDirichlet(n, dirichletAlpha, rng)
		if len(d) != n {
			t.Fatalf("len=%d want %d", len(d), n)
		}
		sum := 0.0
		for _, v := range d {
			if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("bad component %v", v)
			}
			sum += v
		}
		if math.Abs(sum-1) > 1e-9 {
			t.Fatalf("dirichlet sum=%v want 1", sum)
		}
	}
}

func TestBlendDirichletStaysOnSimplex(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	priors := uniformPriors(82)
	out := blendDirichlet(priors, rng)
	sum := 0.0
	for _, v := range out {
		if v < 0 {
			t.Fatalf("negative prior %v", v)
		}
		sum += v
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Fatalf("blended sum=%v want 1", sum)
	}
}

func TestSampleGammaMeanApproxShape(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	const shape = 0.15
	const iters = 200000
	total := 0.0
	for i := 0; i < iters; i++ {
		g := sampleGamma(shape, rng)
		if g < 0 || math.IsNaN(g) {
			t.Fatalf("bad gamma %v", g)
		}
		total += g
	}
	mean := total / iters
	// E[Gamma(shape,1)] = shape; allow generous slack for a low-shape estimator.
	if math.Abs(mean-shape) > 0.03 {
		t.Fatalf("gamma mean=%v want ~%v", mean, shape)
	}
}
