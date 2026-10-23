//go:build js && wasm

// Command gofer-wasm runs the Gofer engine (rules and search) in the browser.
// The network is evaluated by JavaScript (onnxruntime-web) through an
// asynchronous callback, so the same search plays on the web page as in the
// native binary. It installs a global `gofer` object; see web/engine.js.
package main

import (
	"errors"
	"math/rand"
	"syscall/js"

	"github.com/DevomB/gofer/internal/game"
	"github.com/DevomB/gofer/internal/mcts"
	"github.com/DevomB/gofer/internal/nn"
)

const size = 9

type engine struct {
	komi   float64
	board  *game.Board
	moves  []game.Point
	search *mcts.Search
	eval   *jsEval
}

func (e *engine) reset() {
	e.board = game.NewBoard(game.Options{Size: size, Komi: e.komi, Superko: true})
	e.moves = nil
	cfg := mcts.DefaultConfig()
	cfg.Seed = rand.Int63()
	e.search = mcts.New(cfg, e.eval)
	e.search.Reset(e.board)
}

func (e *engine) play(m game.Point) error {
	if err := e.board.Play(m); err != nil {
		return err
	}
	e.moves = append(e.moves, m)
	e.search.Advance(m, e.board)
	return nil
}

// jsEval evaluates positions by calling a JavaScript function
// (spatial Float32Array, global Float32Array, n) -> Promise<{policy, value}>.
type jsEval struct {
	fn    js.Value
	rng   *rand.Rand
	cache map[uint64]nn.Output
}

func (j *jsEval) Eval(boards []*game.Board, out []nn.Output) error {
	if j.fn.Type() != js.TypeFunction {
		return errors.New("gofer: no evaluator set")
	}
	points := size * size
	var todo []int
	for i, b := range boards {
		if o, ok := j.cache[b.Key()]; ok {
			out[i] = o
		} else {
			todo = append(todo, i)
		}
	}
	if len(todo) == 0 {
		return nil
	}
	m := len(todo)
	spatial := make([]float32, m*nn.NumPlanes*points)
	global := make([]float32, m*nn.NumGlobals)
	syms := make([]int, m)
	for k, i := range todo {
		syms[k] = j.rng.Intn(8)
		nn.Encode(boards[i], syms[k], spatial[k*nn.NumPlanes*points:(k+1)*nn.NumPlanes*points], global[k*nn.NumGlobals:(k+1)*nn.NumGlobals])
	}
	res, err := await(j.fn.Invoke(float32Array(spatial), float32Array(global), m))
	if err != nil {
		return err
	}
	logits := goFloats(res.Get("policy"), m*(points+1))
	values := goFloats(res.Get("value"), m)
	for k, i := range todo {
		row := logits[k*(points+1) : (k+1)*(points+1)]
		nn.Softmax(row)
		pol := make([]float32, points+1)
		t := nn.Symmetry(size, syms[k])
		for p := 0; p < points; p++ {
			pol[p] = row[t[p]]
		}
		pol[points] = row[points]
		out[i] = nn.Output{Policy: pol, Value: values[k]}
		if len(j.cache) > 200_000 {
			clear(j.cache)
		}
		j.cache[boards[i].Key()] = out[i]
	}
	return nil
}

// await blocks the calling goroutine until a JS promise settles. It must not
// run on the event-loop goroutine (js.FuncOf callbacks spawn goroutines first).
func await(p js.Value) (js.Value, error) {
	type result struct {
		v   js.Value
		err error
	}
	ch := make(chan result, 1)
	var onOK, onErr js.Func
	onOK = js.FuncOf(func(_ js.Value, args []js.Value) any {
		ch <- result{v: args[0]}
		return nil
	})
	onErr = js.FuncOf(func(_ js.Value, args []js.Value) any {
		ch <- result{err: errors.New(args[0].Call("toString").String())}
		return nil
	})
	p.Call("then", onOK, onErr)
	r := <-ch
	onOK.Release()
	onErr.Release()
	return r.v, r.err
}

func float32Array(xs []float32) js.Value {
	arr := js.Global().Get("Float32Array").New(len(xs))
	buf := js.Global().Get("Uint8Array").New(arr.Get("buffer"))
	b := make([]byte, 4*len(xs))
	for i, x := range xs {
		u := mathFloat32bits(x)
		b[4*i], b[4*i+1], b[4*i+2], b[4*i+3] = byte(u), byte(u>>8), byte(u>>16), byte(u>>24)
	}
	js.CopyBytesToJS(buf, b)
	return arr
}

func goFloats(v js.Value, n int) []float32 {
	b := make([]byte, 4*n)
	js.CopyBytesToGo(b, js.Global().Get("Uint8Array").New(v.Get("buffer"), v.Get("byteOffset"), 4*n))
	out := make([]float32, n)
	for i := range out {
		out[i] = mathFloat32frombits(uint32(b[4*i]) | uint32(b[4*i+1])<<8 | uint32(b[4*i+2])<<16 | uint32(b[4*i+3])<<24)
	}
	return out
}

func (e *engine) state() map[string]any {
	b := e.board
	cells := make([]any, b.NumPoints())
	for p := range cells {
		cells[p] = int(b.At(game.Point(p)))
	}
	var legal []any
	for p := game.Point(0); int(p) < b.NumPoints(); p++ {
		if b.IsLegal(p) {
			legal = append(legal, int(p))
		}
	}
	last := -2
	if m, ok := b.LastMove(0); ok {
		last = int(m)
	}
	return map[string]any{
		"size":      size,
		"cells":     cells,
		"toPlay":    int(b.ToPlay()),
		"ko":        int(b.Ko()),
		"last":      last,
		"moveNum":   b.MoveNum(),
		"gameOver":  b.GameOver(),
		"margin":    b.Margin(),
		"capturesB": b.Captures(game.Black),
		"capturesW": b.Captures(game.White),
		"legal":     legal,
	}
}

// promise runs fn on a new goroutine and returns a JS Promise for its result.
func promise(fn func() (any, error)) js.Value {
	handler := js.FuncOf(func(_ js.Value, args []js.Value) any {
		resolve, reject := args[0], args[1]
		go func() {
			v, err := fn()
			if err != nil {
				reject.Invoke(js.Global().Get("Error").New(err.Error()))
				return
			}
			resolve.Invoke(v)
		}()
		return nil
	})
	return js.Global().Get("Promise").New(handler)
}

func main() {
	e := &engine{komi: 7.5, eval: &jsEval{rng: rand.New(rand.NewSource(rand.Int63())), cache: map[uint64]nn.Output{}}}
	e.reset()
	busy := false
	api := map[string]any{
		"setEvaluator": js.FuncOf(func(_ js.Value, args []js.Value) any {
			e.eval.fn = args[0]
			return nil
		}),
		"newGame": js.FuncOf(func(_ js.Value, args []js.Value) any {
			if len(args) > 0 && args[0].Type() == js.TypeNumber {
				e.komi = args[0].Float()
			}
			e.reset()
			return js.ValueOf(e.state())
		}),
		"state": js.FuncOf(func(js.Value, []js.Value) any { return js.ValueOf(e.state()) }),
		"play": js.FuncOf(func(_ js.Value, args []js.Value) any {
			if busy {
				return js.ValueOf(map[string]any{"ok": false, "error": "engine is thinking"})
			}
			if err := e.play(game.Point(args[0].Int())); err != nil {
				return js.ValueOf(map[string]any{"ok": false, "error": err.Error()})
			}
			return js.ValueOf(map[string]any{"ok": true, "state": e.state()})
		}),
		"undo": js.FuncOf(func(_ js.Value, args []js.Value) any {
			n := 1
			if len(args) > 0 {
				n = args[0].Int()
			}
			moves := e.moves
			if n > len(moves) {
				n = len(moves)
			}
			e.reset()
			for _, m := range moves[:len(moves)-n] {
				e.play(m)
			}
			return js.ValueOf(e.state())
		}),
		// genmove(visits, onProgress?) -> Promise<{move, value, visits, pv, top, state}>
		"genmove": js.FuncOf(func(_ js.Value, args []js.Value) any {
			visits := args[0].Int()
			var progress js.Value
			if len(args) > 1 {
				progress = args[1]
			}
			return promise(func() (any, error) {
				if busy {
					return nil, errors.New("engine is thinking")
				}
				busy = true
				defer func() { busy = false }()
				if e.search.PassWins() {
					res := summary(e.search, 0, 0)
					if err := e.play(game.Pass); err != nil {
						return nil, err
					}
					res["move"] = int(game.Pass)
					res["state"] = e.state()
					return js.ValueOf(res), nil
				}
				for done := 0; done < visits; {
					step := min(48, visits-done)
					if err := e.search.Run(step); err != nil {
						return nil, err
					}
					done += step
					if progress.Type() == js.TypeFunction {
						progress.Invoke(js.ValueOf(summary(e.search, done, visits)))
					}
				}
				res := summary(e.search, visits, visits)
				m := e.search.BestMove()
				if err := e.play(m); err != nil {
					return nil, err
				}
				res["move"] = int(m)
				res["state"] = e.state()
				return js.ValueOf(res), nil
			})
		}),
	}
	js.Global().Set("gofer", js.ValueOf(api))
	if cb := js.Global().Get("goferReady"); cb.Type() == js.TypeFunction {
		cb.Invoke()
	}
	select {}
}

func summary(s *mcts.Search, done, total int) map[string]any {
	var top []any
	for i, c := range s.Children() {
		if i == 5 || c.Visits == 0 {
			break
		}
		top = append(top, map[string]any{"move": int(c.Move), "visits": c.Visits, "q": c.Q, "prior": float64(c.Prior)})
	}
	var pv []any
	for _, m := range s.PV(8) {
		pv = append(pv, int(m))
	}
	return map[string]any{"value": s.RootValue(), "visits": s.Visits(), "done": done, "total": total, "top": top, "pv": pv}
}
