package nn

import (
	"errors"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"github.com/DevomB/gofer/internal/game"
)

// NetConfig controls a batched network evaluator.
type NetConfig struct {
	Size           int           // board size the network was trained for
	Workers        int           // backends running batches concurrently (CPU: ~cores)
	MaxBatch       int           // largest batch handed to a backend
	Wait           time.Duration // how long a worker waits to fill a batch
	RandomSymmetry bool          // evaluate each position under a random dihedral symmetry
	CacheEntries   int           // position cache size; 0 disables caching
	Seed           int64
}

// Net batches concurrent Eval calls onto one or more backends.
type Net struct {
	cfg    NetConfig
	points int
	reqs   chan *request
	cache  *cache
	wg     sync.WaitGroup
	closed atomic.Bool
	stop   chan struct{}
	bks    []Backend
	seed   atomic.Int64

	Stats Stats
}

// Stats counts evaluator activity.
type Stats struct {
	Positions atomic.Int64 // positions sent to the network
	Batches   atomic.Int64
	CacheHits atomic.Int64
	NetNanos  atomic.Int64 // time spent inside Backend.Run
}

type request struct {
	key     uint64
	sym     int
	spatial []float32
	global  []float32
	out     *Output
	group   *callGroup
}

type callGroup struct {
	wg  sync.WaitGroup
	mu  sync.Mutex
	err error
}

func (g *callGroup) fail(err error) {
	g.mu.Lock()
	if g.err == nil {
		g.err = err
	}
	g.mu.Unlock()
}

// NewNet starts cfg.Workers workers, one per backend from newBackend.
func NewNet(cfg NetConfig, newBackend func() (Backend, error)) (*Net, error) {
	if cfg.Workers < 1 {
		cfg.Workers = 1
	}
	if cfg.MaxBatch < 1 {
		cfg.MaxBatch = 64
	}
	if cfg.Wait <= 0 {
		cfg.Wait = time.Millisecond
	}
	n := &Net{
		cfg:    cfg,
		points: cfg.Size * cfg.Size,
		reqs:   make(chan *request, cfg.MaxBatch*cfg.Workers*4),
		stop:   make(chan struct{}),
	}
	n.seed.Store(cfg.Seed)
	if cfg.CacheEntries > 0 {
		n.cache = newCache(cfg.CacheEntries)
	}
	for i := 0; i < cfg.Workers; i++ {
		bk, err := newBackend()
		if err != nil {
			n.Close()
			return nil, err
		}
		n.bks = append(n.bks, bk)
		n.wg.Add(1)
		go n.worker(bk)
	}
	return n, nil
}

// Eval implements Evaluator.
func (n *Net) Eval(boards []*game.Board, out []Output) error {
	if n.closed.Load() {
		return errors.New("nn: evaluator closed")
	}
	g := &callGroup{}
	rng := rand.New(rand.NewSource(n.seed.Add(1)))
	for i, b := range boards {
		if b.Size() != n.cfg.Size {
			return errors.New("nn: board size does not match the network")
		}
		key := b.Key()
		if n.cache != nil {
			if o, ok := n.cache.get(key); ok {
				out[i] = o
				n.Stats.CacheHits.Add(1)
				continue
			}
		}
		sym := 0
		if n.cfg.RandomSymmetry {
			sym = rng.Intn(8)
		}
		r := &request{
			key:     key,
			sym:     sym,
			spatial: make([]float32, NumPlanes*n.points),
			global:  make([]float32, NumGlobals),
			out:     &out[i],
			group:   g,
		}
		Encode(b, sym, r.spatial, r.global)
		g.wg.Add(1)
		n.reqs <- r
	}
	g.wg.Wait()
	return g.err
}

func (n *Net) worker(bk Backend) {
	defer n.wg.Done()
	batch := make([]*request, 0, n.cfg.MaxBatch)
	spatial := make([]float32, 0, n.cfg.MaxBatch*NumPlanes*n.points)
	global := make([]float32, 0, n.cfg.MaxBatch*NumGlobals)
	for {
		batch = batch[:0]
		select {
		case <-n.stop:
			return
		case r := <-n.reqs:
			batch = append(batch, r)
		}
		deadline := time.NewTimer(n.cfg.Wait)
	fill:
		for len(batch) < n.cfg.MaxBatch {
			select {
			case r := <-n.reqs:
				batch = append(batch, r)
			case <-deadline.C:
				break fill
			}
		}
		deadline.Stop()

		spatial, global = spatial[:0], global[:0]
		for _, r := range batch {
			spatial = append(spatial, r.spatial...)
			global = append(global, r.global...)
		}
		t0 := time.Now()
		logits, values, err := bk.Run(spatial, global, len(batch))
		n.Stats.NetNanos.Add(int64(time.Since(t0)))
		n.Stats.Batches.Add(1)
		n.Stats.Positions.Add(int64(len(batch)))
		for i, r := range batch {
			if err != nil {
				r.group.fail(err)
			} else {
				*r.out = n.decode(r, logits[i*(n.points+1):(i+1)*(n.points+1)], values[i])
				if n.cache != nil {
					n.cache.put(r.key, *r.out)
				}
			}
			r.group.wg.Done()
		}
	}
}

// decode undoes the input symmetry on the policy.
func (n *Net) decode(r *request, logits []float32, value float32) Output {
	probs := append([]float32(nil), logits...)
	Softmax(probs)
	policy := make([]float32, n.points+1)
	t := symmetry(n.cfg.Size, r.sym)
	for p := 0; p < n.points; p++ {
		policy[p] = probs[t[p]]
	}
	policy[n.points] = probs[n.points]
	return Output{Policy: policy, Value: value}
}

// Close stops the workers and releases backends.
func (n *Net) Close() error {
	if n.closed.Swap(true) {
		return nil
	}
	close(n.stop)
	n.wg.Wait()
	var err error
	for _, bk := range n.bks {
		if e := bk.Close(); e != nil && err == nil {
			err = e
		}
	}
	return err
}

// cache is a sharded position cache that drops a shard when it fills up.
type cache struct {
	shards [64]cacheShard
	per    int
}

type cacheShard struct {
	mu sync.Mutex
	m  map[uint64]Output
}

func newCache(entries int) *cache {
	c := &cache{per: entries/64 + 1}
	for i := range c.shards {
		c.shards[i].m = make(map[uint64]Output)
	}
	return c
}

func (c *cache) get(k uint64) (Output, bool) {
	s := &c.shards[k&63]
	s.mu.Lock()
	o, ok := s.m[k]
	s.mu.Unlock()
	return o, ok
}

func (c *cache) put(k uint64, o Output) {
	s := &c.shards[k&63]
	s.mu.Lock()
	if len(s.m) >= c.per {
		clear(s.m)
	}
	s.m[k] = o
	s.mu.Unlock()
}
