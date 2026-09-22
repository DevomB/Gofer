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

func TestTemperPriorsFlattens(t *testing.T) {
	// Temperature 2 takes square roots: 0.8, 0.4, 0.4, 0.2, renormalised by 1.8.
	got := temperPriors([]float64{0.64, 0.16, 0.16, 0.04}, 2)
	for i, root := range []float64{0.8, 0.4, 0.4, 0.2} {
		if want := root / 1.8; math.Abs(got[i]-want) > 1e-12 {
			t.Fatalf("temperPriors[%d] = %v, want %v", i, got[i], want)
		}
	}
}

// Self-play advances the tree between moves, so every root after the first was
// expanded as an ordinary node. Its priors must still get the root temperature,
// plus noise on a full search and no noise on a fast one.
func TestReusedRootGetsRootPriors(t *testing.T) {
	rs := Chinese()
	b := NewBoard(9, 6.5)
	cfg := DefaultConfig()
	cfg.Workers = 1
	eng := NewEngine(rs, Heuristic{}, cfg)
	rng := rand.New(rand.NewSource(1))
	eng.ConfigureSelfplayMove(200, true)
	m := eng.SelectMove(b, rng, 0)
	rs.Play(b, m)
	eng.AdvanceTree(m)

	root := eng.arena.Get(eng.root)
	if !root.Expanded {
		t.Fatal("the played move was never expanded, so this test reuses nothing")
	}
	moves := make([]Move, len(root.Children))
	for i, cidx := range root.Children {
		moves[i] = eng.arena.Get(cidx).Move
	}
	tempered := temperPriors(policyPriors(b, moves, Heuristic{}.Evaluate(b).Policy), cfg.RootTemperature)
	differs := func() int {
		n := 0
		for i, cidx := range root.Children {
			if math.Abs(eng.arena.Get(cidx).Prior-tempered[i]) > 1e-12 {
				n++
			}
		}
		return n
	}

	eng.ConfigureSelfplayMove(50, false)
	eng.SelectMove(b, rng, 0)
	if n := differs(); n != 0 {
		t.Fatalf("fast search: %d of %d root priors are not the tempered policy (noise leaked in)", n, len(moves))
	}
	eng.ConfigureSelfplayMove(200, true)
	eng.SelectMove(b, rng, 0)
	if n := differs(); n == 0 {
		t.Fatal("full search on a reused root ran without exploration noise")
	}
}
