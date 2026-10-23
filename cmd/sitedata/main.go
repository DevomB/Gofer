//go:build onnx

// Command sitedata builds the JSON the web page draws from a training run:
//
//	web/data/hero.json         Gofer's own analysis of every position of a game
//	web/data/generations.json  per generation: Elo, opening policy, one self-play game
//
//	go run -tags onnx ./cmd/sitedata -run runs/main -hero docs/results/games/gnugo-l10-vs-gofer.sgf
//
// Every number on the page traces back to this command and the run directory.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/DevomB/gofer/internal/game"
	"github.com/DevomB/gofer/internal/mcts"
	"github.com/DevomB/gofer/internal/nn"
	"github.com/DevomB/gofer/internal/sgf"
)

const size = 9

func main() {
	run := flag.String("run", "runs/main", "training run directory (models/, sgf/, eval/ratings)")
	ratings := flag.String("ratings", "docs/results/ratings.json", "Bradley-Terry ratings written by training/report.py")
	model := flag.String("model", "models/gofer-9x9.onnx", "network that analyses the hero game")
	hero := flag.String("hero", "docs/results/games/gnugo-l10-vs-gofer.sgf", "game to analyse for the hero")
	extra := flag.String("extra-sgf", "", "self-play SGFs for generations the run has none for (dir with gen-NNNN/ subdirs)")
	visits := flag.Int("visits", 800, "search visits per analysed position")
	out := flag.String("out", "web/data", "output directory")
	flag.Parse()
	log.SetFlags(0)

	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}
	h, err := analyseGame(*hero, *model, *visits)
	if err != nil {
		log.Fatal(err)
	}
	write(filepath.Join(*out, "hero.json"), h)

	g, err := generations(*run, *ratings, *extra)
	if err != nil {
		log.Fatal(err)
	}
	write(filepath.Join(*out, "generations.json"), g)
}

func write(path string, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		log.Fatal(err)
	}
	log.Printf("wrote %s (%d bytes)", path, len(b))
}

// pt is a point index (y*9+x, y=0 the top row), -1 for pass.
func pt(p game.Point) int {
	if p == game.Pass {
		return -1
	}
	return int(p)
}

func round(x float64, digits int) float64 {
	s := math.Pow(10, float64(digits))
	return math.Round(x*s) / s
}

// ---- hero ----

type candidate struct {
	P  int     `json:"p"`
	N  int     `json:"n"`
	W  float64 `json:"w"` // win probability for the side to move
	Pr float64 `json:"pr"`
}

type frame struct {
	Color string      `json:"c"`
	P     int         `json:"p"`
	BW    float64     `json:"bw"` // Black's win probability before the move
	N     int         `json:"n"`
	Top   []candidate `json:"top"`
	PV    []int       `json:"pv"`
}

type heroGame struct {
	Black  string  `json:"black"`
	White  string  `json:"white"`
	Result string  `json:"result"`
	Komi   float64 `json:"komi"`
	Visits int     `json:"visits"`
	Moves  []frame `json:"moves"`
}

func readGame(path string) (*sgf.Game, []sgf.Move, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	g, err := sgf.Parse(string(data))
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", path, err)
	}
	moves, err := g.MainLine()
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", path, err)
	}
	return g, moves, nil
}

func analyseGame(path, model string, visits int) (*heroGame, error) {
	g, moves, err := readGame(path)
	if err != nil {
		return nil, err
	}
	net, err := nn.NewNet(nn.NetConfig{Size: size, Workers: 4, MaxBatch: 32, RandomSymmetry: true, CacheEntries: 200_000, Seed: 1},
		func() (nn.Backend, error) { return nn.NewORTBackend(model, size, nn.ORTOptions{}) })
	if err != nil {
		return nil, err
	}
	defer net.Close()

	b := game.NewBoard(game.Options{Size: size, Komi: g.Komi, Superko: true})
	cfg := mcts.DefaultConfig()
	cfg.Seed = 1
	s := mcts.New(cfg, net)
	s.Reset(b)
	h := &heroGame{Black: g.Root.Get("PB"), White: g.Root.Get("PW"), Result: g.Result, Komi: g.Komi, Visits: visits}
	for i, m := range moves {
		if err := s.Run(visits); err != nil {
			return nil, err
		}
		v := s.RootValue()
		bw := (1 + v) / 2
		if b.ToPlay() == game.White {
			bw = 1 - bw
		}
		f := frame{Color: colorName(m.Color), P: pt(m.Point), BW: round(bw, 3), N: s.Visits()}
		for j, c := range s.Children() {
			if j == 5 || c.Visits == 0 {
				break
			}
			f.Top = append(f.Top, candidate{P: pt(c.Move), N: c.Visits, W: round((1+c.Q)/2, 3), Pr: round(float64(c.Prior), 3)})
		}
		for _, p := range s.PV(6) {
			f.PV = append(f.PV, pt(p))
		}
		h.Moves = append(h.Moves, f)
		if err := b.Play(m.Point); err != nil {
			return nil, fmt.Errorf("move %d: %w", i+1, err)
		}
		s.Advance(m.Point, b)
	}
	return h, nil
}

func colorName(c game.Color) string {
	if c == game.White {
		return "W"
	}
	return "B"
}

// ---- generations ----

type generation struct {
	Gen     int       `json:"gen"`
	Elo     float64   `json:"elo"`
	Stderr  float64   `json:"se"`
	Opening []float64 `json:"opening"` // raw policy on the empty board, 81 points + pass
	Value   float64   `json:"value"`   // Black's win probability on the empty board
	Game    *spGame   `json:"game,omitempty"`
	// From the run's metrics.jsonl: evaluation matches and the policy loss of
	// this network on the self-play data it generated (before training on it).
	Results    []result `json:"results,omitempty"`
	PolicyLoss float64  `json:"policyLoss,omitempty"`
}

type result struct {
	Opp    string `json:"opp"`
	Wins   int    `json:"w"`
	Losses int    `json:"l"`
}

type ladder struct {
	Gens    []generation       `json:"gens"`
	Anchors map[string]float64 `json:"anchors"` // Elo of the fixed opponents on the same scale
}

type spGame struct {
	Result string `json:"result"`
	Moves  []int  `json:"moves"`
}

func generations(run, ratingsPath, extra string) (*ladder, error) {
	data, err := os.ReadFile(ratingsPath)
	if err != nil {
		return nil, err
	}
	var ratings map[string]struct {
		Elo    float64 `json:"elo"`
		Stderr float64 `json:"stderr"`
	}
	if err := json.Unmarshal(data, &ratings); err != nil {
		return nil, err
	}
	metrics, err := readMetrics(filepath.Join(run, "metrics.jsonl"))
	if err != nil {
		return nil, err
	}
	models, err := filepath.Glob(filepath.Join(run, "models", "gen-*.onnx"))
	if err != nil {
		return nil, err
	}
	sort.Strings(models)
	l := &ladder{Anchors: map[string]float64{}}
	for name, r := range ratings {
		if !strings.HasPrefix(name, "gen-") {
			l.Anchors[name] = round(r.Elo, 1)
		}
	}
	for i, path := range models {
		name := strings.TrimSuffix(filepath.Base(path), ".onnx")
		r, ok := ratings[name]
		if !ok {
			return nil, fmt.Errorf("%s: no rating", name)
		}
		pol, val, err := openingPolicy(path)
		if err != nil {
			return nil, err
		}
		gen := generation{Gen: i, Elo: round(r.Elo, 1), Stderr: round(r.Stderr, 1), Opening: pol, Value: round(val, 3)}
		if m, ok := metrics[i]; ok {
			gen.Results, gen.PolicyLoss = m.results, round(m.policyLoss, 3)
		}
		dirs := []string{filepath.Join(run, "sgf", name)}
		if extra != "" {
			dirs = append(dirs, filepath.Join(extra, name))
		}
		for _, dir := range dirs {
			if gen.Game, err = typicalGame(dir); err != nil {
				return nil, err
			}
			if gen.Game != nil {
				break
			}
		}
		log.Printf("%s: elo %.0f, opening max %.3f, game %v", name, r.Elo, maxOf(pol[:size*size]), gen.Game != nil)
		l.Gens = append(l.Gens, gen)
	}
	return l, nil
}

type genMetrics struct {
	results    []result
	policyLoss float64
}

func readMetrics(path string) (map[int]genMetrics, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	out := map[int]genMetrics{}
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var rec struct {
			Gen       int     `json:"gen"`
			Phase     string  `json:"phase"`
			ValPolicy float64 `json:"val_policy"`
			Results   []struct {
				B      string `json:"b"`
				Wins   int    `json:"wins"`
				Losses int    `json:"losses"`
			} `json:"results"`
		}
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		m := out[rec.Gen]
		switch rec.Phase {
		case "eval":
			m.results = nil
			for _, r := range rec.Results {
				m.results = append(m.results, result{Opp: r.B, Wins: r.Wins, Losses: r.Losses})
			}
		case "train":
			m.policyLoss = rec.ValPolicy
		}
		out[rec.Gen] = m
	}
	return out, nil
}

func maxOf(xs []float64) float64 {
	m := 0.0
	for _, x := range xs {
		m = max(m, x)
	}
	return m
}

// openingPolicy evaluates the empty board through all eight symmetries and
// averages: what the network "wants" to play first, before any search.
func openingPolicy(model string) ([]float64, float64, error) {
	bk, err := nn.NewORTBackend(model, size, nn.ORTOptions{})
	if err != nil {
		return nil, 0, err
	}
	defer bk.Close()
	b := game.NewBoard(game.Options{Size: size, Komi: 7.5, Superko: true})
	points, planes := size*size, nn.NumPlanes*size*size
	spatial := make([]float32, 8*planes)
	global := make([]float32, 8*nn.NumGlobals)
	for s := 0; s < 8; s++ {
		nn.Encode(b, s, spatial[s*planes:(s+1)*planes], global[s*nn.NumGlobals:(s+1)*nn.NumGlobals])
	}
	logits, values, err := bk.Run(spatial, global, 8)
	if err != nil {
		return nil, 0, err
	}
	pol := make([]float64, points+1)
	value := 0.0
	for s := 0; s < 8; s++ {
		row := logits[s*(points+1) : (s+1)*(points+1)]
		nn.Softmax(row)
		t := nn.Symmetry(size, s)
		for p := 0; p < points; p++ {
			pol[p] += float64(row[t[p]]) / 8
		}
		pol[points] += float64(row[points]) / 8
		value += float64(values[s]) / 8
	}
	for i := range pol {
		pol[i] = round(pol[i], 4)
	}
	return pol, (1 + value) / 2, nil
}

// typicalGame picks the self-play game that was scored (not resigned) with the
// median length, so the final position shows what that generation's Go looks like.
func typicalGame(dir string) (*spGame, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.sgf"))
	if err != nil || len(files) == 0 {
		return nil, err
	}
	sort.Strings(files)
	type cand struct {
		g     *sgf.Game
		moves []sgf.Move
	}
	var scored []cand
	for _, f := range files {
		g, moves, err := readGame(f)
		if err != nil {
			return nil, err
		}
		if !strings.HasSuffix(g.Result, "+R") {
			scored = append(scored, cand{g, moves})
		}
	}
	if len(scored) == 0 {
		return nil, nil
	}
	sort.SliceStable(scored, func(i, j int) bool { return len(scored[i].moves) < len(scored[j].moves) })
	c := scored[len(scored)/2]
	sp := &spGame{Result: c.g.Result}
	for _, m := range c.moves {
		sp.Moves = append(sp.Moves, pt(m.Point))
	}
	return sp, nil
}
