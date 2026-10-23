package main

import (
	"flag"
	"fmt"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/DevomB/gofer/internal/arena"
	"github.com/DevomB/gofer/internal/mcts"
	"github.com/DevomB/gofer/internal/nn"
)

// netFlags are the shared flags that configure a network evaluator.
type netFlags struct {
	model    string
	device   string
	workers  int
	threads  int
	maxBatch int
	wait     time.Duration
	cache    int
	noSym    bool
}

func (f *netFlags) register(fs *flag.FlagSet) {
	fs.StringVar(&f.model, "model", "", "ONNX model exported by training/ (empty: network-free uniform evaluator)")
	fs.StringVar(&f.device, "device", "cpu", "inference device: cpu, cuda, tensorrt")
	fs.IntVar(&f.workers, "workers", 0, "concurrent inference sessions (0: CPU cores, or 2 on GPU)")
	fs.IntVar(&f.threads, "threads", 1, "intra-op threads per CPU session")
	fs.IntVar(&f.maxBatch, "batch", 0, "max inference batch (0: 32 on CPU, 512 on GPU)")
	fs.DurationVar(&f.wait, "batch-wait", 0, "max time to fill a batch (0: 200µs on CPU, 2ms on GPU)")
	fs.IntVar(&f.cache, "cache", 200_000, "position cache entries")
	fs.BoolVar(&f.noSym, "no-symmetry", false, "disable random board symmetry at inference")
}

// evaluator builds the evaluator described by the flags.
func (f *netFlags) evaluator(size int, seed int64) (nn.Evaluator, func(), error) {
	if f.model == "" {
		return nn.Uniform{}, func() {}, nil
	}
	return openNet(f.model, size, seed, f)
}

func openNet(path string, size int, seed int64, f *netFlags) (*nn.Net, func(), error) {
	gpu := f.device != "cpu"
	cfg := nn.NetConfig{
		Size:           size,
		Workers:        f.workers,
		MaxBatch:       f.maxBatch,
		Wait:           f.wait,
		RandomSymmetry: !f.noSym,
		CacheEntries:   f.cache,
		Seed:           seed,
	}
	if cfg.Workers == 0 {
		cfg.Workers = runtime.NumCPU()
		if gpu {
			cfg.Workers = 2
		}
	}
	if cfg.MaxBatch == 0 {
		cfg.MaxBatch = 32
		if gpu {
			cfg.MaxBatch = 512
		}
	}
	if cfg.Wait == 0 {
		cfg.Wait = 200 * time.Microsecond
		if gpu {
			cfg.Wait = 2 * time.Millisecond
		}
	}
	opts := nn.ORTOptions{Device: f.device, CPUThreads: f.threads}
	net, err := nn.NewNet(cfg, func() (nn.Backend, error) { return nn.NewORTBackend(path, size, opts) })
	if err != nil {
		return nil, nil, err
	}
	return net, func() { net.Close() }, nil
}

// searchFlags configure MCTS.
type searchFlags struct {
	cpuct  float64
	fpu    float64
	batch  int
	visits int
}

func (f *searchFlags) register(fs *flag.FlagSet, visits int) {
	d := mcts.DefaultConfig()
	fs.Float64Var(&f.cpuct, "cpuct", d.CPuct, "PUCT exploration constant")
	fs.Float64Var(&f.fpu, "fpu", d.FPUReduction, "first-play-urgency reduction")
	fs.IntVar(&f.batch, "leaf-batch", d.Batch, "leaves evaluated together per search step")
	fs.IntVar(&f.visits, "visits", visits, "playouts per move")
}

func (f *searchFlags) config(seed int64) mcts.Config {
	c := mcts.DefaultConfig()
	c.CPuct, c.FPUReduction, c.Batch, c.Seed = f.cpuct, f.fpu, f.batch, seed
	return c
}

// playerSpec builds arena players from strings such as
//
//	nn:models/gen-0004.onnx,visits=200
//	rollout,visits=400,playouts=2
//	uniform,visits=50
//	gnugo,level=5
//	gtp:/path/to/engine --mode gtp
type playerSpec struct {
	kind   string
	arg    string
	opts   map[string]string
	visits int
}

func parsePlayerSpec(s string, defVisits int) (playerSpec, error) {
	ps := playerSpec{opts: map[string]string{}, visits: defVisits}
	if rest, ok := strings.CutPrefix(s, "gtp:"); ok {
		ps.kind, ps.arg = "gtp", rest
		return ps, nil
	}
	parts := strings.Split(s, ",")
	head := parts[0]
	ps.kind = head
	if k, a, ok := strings.Cut(head, ":"); ok {
		ps.kind, ps.arg = k, a
	}
	for _, kv := range parts[1:] {
		k, v, ok := strings.Cut(kv, "=")
		if !ok {
			return ps, fmt.Errorf("bad option %q in %q", kv, s)
		}
		ps.opts[k] = v
	}
	if v, ok := ps.opts["visits"]; ok {
		n, err := strconv.Atoi(v)
		if err != nil {
			return ps, err
		}
		ps.visits = n
	}
	switch ps.kind {
	case "nn", "rollout", "uniform", "gnugo", "gtp":
	default:
		return ps, fmt.Errorf("unknown player kind %q", ps.kind)
	}
	return ps, nil
}

func (ps playerSpec) label() string {
	switch ps.kind {
	case "nn":
		name := ps.arg[strings.LastIndex(ps.arg, "/")+1:]
		return fmt.Sprintf("%s@%d", strings.TrimSuffix(name, ".onnx"), ps.visits)
	case "gnugo":
		return "gnugo-l" + ps.optOr("level", "10")
	case "gtp":
		return strings.Fields(ps.arg)[0]
	}
	return fmt.Sprintf("%s@%d", ps.kind, ps.visits)
}

func (ps playerSpec) optOr(k, def string) string {
	if v, ok := ps.opts[k]; ok {
		return v
	}
	return def
}

// nets shares one evaluator per model path across all arena games.
type nets struct {
	mu   sync.Mutex
	m    map[string]*nn.Net
	f    *netFlags
	size int
}

func (n *nets) get(path string) (*nn.Net, error) {
	n.mu.Lock()
	defer n.mu.Unlock()
	if net, ok := n.m[path]; ok {
		return net, nil
	}
	net, _, err := openNet(path, n.size, int64(len(n.m)+1), n.f)
	if err != nil {
		return nil, err
	}
	n.m[path] = net
	return net, nil
}

func (n *nets) close() {
	for _, net := range n.m {
		net.Close()
	}
}

func (ps playerSpec) factory(pool *nets, sf *searchFlags, seed *int64) func() (arena.Player, error) {
	return func() (arena.Player, error) {
		cfg := sf.config(0)
		pool.mu.Lock()
		*seed++
		cfg.Seed = *seed
		pool.mu.Unlock()
		switch ps.kind {
		case "nn":
			net, err := pool.get(ps.arg)
			if err != nil {
				return nil, err
			}
			return &arena.MCTSPlayer{Label: ps.label(), Eval: net, Config: cfg, Visits: ps.visits}, nil
		case "rollout":
			k, _ := strconv.Atoi(ps.optOr("playouts", "1"))
			return &arena.MCTSPlayer{Label: ps.label(), Eval: nn.NewRollout(k, cfg.Seed), Config: cfg, Visits: ps.visits}, nil
		case "uniform":
			return &arena.MCTSPlayer{Label: ps.label(), Eval: nn.Uniform{}, Config: cfg, Visits: ps.visits}, nil
		case "gnugo":
			bin := ps.optOr("bin", "gnugo")
			return arena.NewGTPPlayer(ps.label(), []string{bin, "--mode", "gtp", "--level", ps.optOr("level", "10"), "--chinese-rules"}), nil
		default:
			return arena.NewGTPPlayer(ps.label(), strings.Fields(ps.arg)), nil
		}
	}
}
