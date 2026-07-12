package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"
)

type cliFlags struct {
	gtp, play, analyze, selfplay, watch, arena, convertSGF, verifyJSONL bool
	convertMaxRows                                         int
	convertEpsilon                                         float64
	games                                      int
	size                                       int
	komi                                       float64
	playouts, gtpPlayouts                      int
	think                                      time.Duration
	topN                                       int
	eval, humanColor, out, sgfDir              string
	blackEval, whiteEval, arenaJSON            string
	blackPlayouts, whitePlayouts               int
	selfplayFullOnly                           bool
	selfplayEval                               string
	selfplayONNXFraction                       float64
	selfplayParallel                           int
	selfplayTempMoves                          int
	selfplayFastPlayouts                       int
	selfplayFullPlayouts                       int
	selfplayCapRandomizeP                      float64
	selfplayFixedPlayouts                      bool
	seed                                       int64
	sgfPath, setup                             string
	modelPath, onnxURL, onnxURL2               string
	evalBackend, modelPath2                    string
	batchSize                                  int
	evalTimeout                                time.Duration
	arenaEnhanced                              string
	arenaParallel                              int
	arenaOpeningMoves                          int
	arenaTemp                                  float64
}

func parseCLIFlags() cliFlags {
	var f cliFlags
	flag.BoolVar(&f.gtp, "gtp", false, "run GTP protocol on stdin/stdout")
	flag.BoolVar(&f.play, "play", false, "interactive terminal game")
	flag.BoolVar(&f.analyze, "analyze", false, "analyze current position")
	flag.BoolVar(&f.selfplay, "selfplay", false, "run self-play and emit training samples")
	flag.BoolVar(&f.watch, "watch", false, "engine vs engine demo (mutually exclusive with other modes)")
	flag.BoolVar(&f.arena, "arena", false, "run arena match (baseline vs challenger)")
	flag.BoolVar(&f.convertSGF, "convert-sgf", false, "convert SGF archives to supervised JSONL (pass input dirs as args)")
	flag.BoolVar(&f.verifyJSONL, "verify-jsonl", false, "verify converted JSONL sample schema (-o path)")
	flag.IntVar(&f.convertMaxRows, "convert-max-rows", 0, "max training rows for -convert-sgf (0 = unlimited)")
	flag.Float64Var(&f.convertEpsilon, "convert-epsilon", defaultPolicyEpsilon, "label-smoothing epsilon for -convert-sgf policy targets")
	flag.IntVar(&f.games, "games", 1, "number of games (with -selfplay or -arena)")
	flag.IntVar(&f.size, "size", 9, "board size")
	flag.Float64Var(&f.komi, "komi", 6.5, "komi")
	flag.IntVar(&f.playouts, "playouts", 0, "MCTS playouts per move (0 = size default)")
	flag.IntVar(&f.gtpPlayouts, "gtp-playouts", 0, "MCTS playouts per GTP move (0 = size default)")
	flag.DurationVar(&f.think, "think-time", 0, "search time budget per move (overrides -playouts when set)")
	flag.IntVar(&f.topN, "top", 5, "top moves to show (with -analyze)")
	flag.StringVar(&f.eval, "eval", "heuristic", "evaluator: uniform, heuristic, batched, onnx")
	flag.StringVar(&f.humanColor, "color", "b", "your color in -play: b or w")
	flag.StringVar(&f.out, "o", "", "output path (self-play JSON, game SGF, or GTP SGF on quit)")
	flag.StringVar(&f.sgfDir, "sgf-dir", "", "write self-play games as SGF files to directory")
	flag.StringVar(&f.sgfPath, "sgf", "", "replay SGF file and print score")
	flag.StringVar(&f.setup, "moves", "", "comma-separated setup moves for -analyze (e.g. D4,Q16)")
	flag.StringVar(&f.blackEval, "black-eval", "heuristic", "baseline evaluator for -arena")
	flag.StringVar(&f.whiteEval, "white-eval", "uniform", "challenger evaluator for -arena")
	flag.StringVar(&f.arenaJSON, "json", "", "write arena match JSON report to path")
	flag.IntVar(&f.blackPlayouts, "black-playouts", 0, "baseline playouts for -arena (0 = -playouts)")
	flag.IntVar(&f.whitePlayouts, "white-playouts", 0, "challenger playouts for -arena (0 = -playouts)")
	flag.BoolVar(&f.selfplayFullOnly, "full-only", true, "export only full-search self-play positions")
	flag.StringVar(&f.selfplayEval, "selfplay-eval", "mix", "self-play evaluator: heuristic, onnx, mix")
	flag.Float64Var(&f.selfplayONNXFraction, "selfplay-onnx-fraction", 0.7, "fraction of mix-mode games using ONNX (odd games use ONNX)")
	flag.IntVar(&f.selfplayParallel, "selfplay-parallel", 8, "concurrent self-play games (feeds real batches to the ONNX sidecar)")
	flag.IntVar(&f.selfplayTempMoves, "selfplay-temp-moves", 16, "opening plies sampled from the visit distribution for game diversity (0 = always argmax)")
	flag.IntVar(&f.selfplayFastPlayouts, "selfplay-fast-playouts", 0, "self-play fast search cap per move (0 = full/4)")
	flag.IntVar(&f.selfplayFullPlayouts, "selfplay-full-playouts", 0, "self-play full search cap per move (0 = -playouts)")
	flag.Float64Var(&f.selfplayCapRandomizeP, "selfplay-cap-randomize-p", 0.20, "fraction of self-play moves that use the full cap (0 = fixed full cap every move)")
	flag.BoolVar(&f.selfplayFixedPlayouts, "selfplay-fixed-playouts", false, "disable cap randomization: every move uses -playouts (sets fast=full and cap-randomize-p=0)")
	flag.Int64Var(&f.seed, "seed", 1, "RNG seed for -arena and -selfplay")
	flag.StringVar(&f.modelPath, "model", "models/gofer-9x9-bootstrap.onnx", "ONNX model path (sidecar or in-process primary)")
	flag.StringVar(&f.modelPath2, "model-2", "models/gofer-9x9-candidate.onnx", "second ONNX model for -white-eval onnx2 / champion-vs-challenger")
	flag.StringVar(&f.evalBackend, "eval-backend", "inprocess", "ONNX inference transport: inprocess (ORT in Go binary, requires -tags=onnx build) or sidecar (HTTP)")
	flag.StringVar(&f.onnxURL, "onnx-url", "http://127.0.0.1:8080", "ONNX inference sidecar base URL")
	flag.StringVar(&f.onnxURL2, "onnx-url-2", "http://127.0.0.1:8081", "second ONNX sidecar base URL (eval name onnx2, for champion-vs-challenger)")
	flag.IntVar(&f.batchSize, "batch-size", 8, "batched evaluator minimum batch size")
	flag.DurationVar(&f.evalTimeout, "eval-timeout", 500*time.Millisecond, "batched/onnx eval timeout before heuristic fallback")
	flag.StringVar(&f.arenaEnhanced, "arena-enhanced", "none", "arena forced root playouts: none, baseline, both")
	flag.IntVar(&f.arenaParallel, "arena-parallel", 8, "concurrent arena games (shared evaluators feed real batches to the sidecars)")
	flag.IntVar(&f.arenaOpeningMoves, "arena-opening-moves", 8, "opening plies sampled from the visit distribution so match games differ (0 = deterministic, identical games)")
	flag.Float64Var(&f.arenaTemp, "arena-temp", 1.0, "sampling temperature for arena opening plies")
	flag.Parse()
	SetEvalConfig(EvalConfig{
		ModelPath:   f.modelPath,
		ModelPath2:  f.modelPath2,
		ONNXURL:     f.onnxURL,
		ONNXURL2:    f.onnxURL2,
		Backend:     f.evalBackend,
		BatchSize:   f.batchSize,
		EvalTimeout: f.evalTimeout,
		MaxWait:     2 * time.Millisecond,
	})
	return f
}

func runMode(f cliFlags) bool {
	n := 0
	for _, on := range []bool{f.gtp, f.play, f.analyze, f.selfplay, f.watch, f.arena, f.convertSGF, f.verifyJSONL} {
		if on {
			n++
		}
	}
	if n > 1 {
		fmt.Fprintln(os.Stderr, "only one mode flag allowed (-gtp, -play, -analyze, -selfplay, -watch, -arena, -convert-sgf, -verify-jsonl)")
		os.Exit(1)
	}
	switch {
	case f.gtp:
		runGTP(f.gtpPlayouts, f.size, f.think, f.eval, f.out)
	case f.play:
		p := f.playouts
		if p <= 0 {
			p = defaultPlayoutsForSize(f.size)
		}
		runPlay(f.size, f.komi, p, f.think, f.eval, f.humanColor, f.out)
	case f.analyze:
		p := f.playouts
		if p <= 0 && f.think <= 0 {
			p = defaultPlayoutsForSize(f.size)
		}
		runAnalyze(f.size, f.komi, p, f.think, f.topN, f.eval, parseSetupMoves(f.setup))
	case f.selfplay:
		runSelfplayCLI(f)
	case f.watch:
		p := f.playouts
		if p <= 0 && f.think <= 0 {
			p = defaultPlayoutsForSize(f.size)
		}
		runWatch(f.size, f.komi, p, f.think, f.eval)
	case f.arena:
		runArenaCLI(f)
	case f.convertSGF:
		if f.out == "" {
			fmt.Fprintln(os.Stderr, "-convert-sgf requires -o output.jsonl")
			os.Exit(1)
		}
		runSGFConvertCLI(f, flag.Args())
	case f.verifyJSONL:
		runVerifyJSONLCLI(f.out, 5)
	default:
		return false
	}
	return true
}

func parseSetupMoves(setup string) []string {
	if setup == "" {
		return nil
	}
	var moves []string
	for _, m := range strings.Split(setup, ",") {
		if s := strings.TrimSpace(m); s != "" {
			moves = append(moves, s)
		}
	}
	return moves
}

func printUsage(size int, komi float64) {
	b := NewBoard(size, komi)
	r := Chinese()
	fmt.Printf("gofer %dx%d komi=%.1f, %d legal moves\n", size, size, komi, len(r.LegalMoves(b)))
	fmt.Println("\nUsage:")
	fmt.Println("  gofer -play              interactive game")
	fmt.Println("  gofer -analyze           MCTS analysis")
	fmt.Println("  gofer -gtp                GTP engine (Sabaki/Lizzie)")
	fmt.Println("  gofer -watch              engine vs engine demo")
	fmt.Println("  gofer -selfplay           generate training games")
	fmt.Println("  gofer -arena              baseline vs challenger match")
	fmt.Println("  gofer -convert-sgf        SGF dirs → supervised JSONL (-o required)")
	fmt.Println("  gofer -sgf game.sgf       replay SGF")
}

func runArenaCLI(f cliFlags) {
	cfg := MatchConfig{
		Games:         f.games,
		Size:          f.size,
		Komi:          f.komi,
		Playouts:      f.playouts,
		BlackPlayouts: f.blackPlayouts,
		WhitePlayouts: f.whitePlayouts,
		ThinkTime:     f.think,
		BlackEval:     f.blackEval,
		WhiteEval:     f.whiteEval,
		Seed:          f.seed,
		ArenaEnhanced: f.arenaEnhanced,
		Parallel:      f.arenaParallel,
		OpeningMoves:  f.arenaOpeningMoves,
		OpeningTemp:   f.arenaTemp,
	}
	result := RunMatch(cfg)
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if f.arenaJSON != "" {
		if err := os.WriteFile(f.arenaJSON, data, 0644); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	fmt.Printf("arena %d games: baseline(%s)=%d challenger(%s)=%d draws=%d black=%d white=%d win_rate_challenger=%.3f CI=[%.3f,%.3f] promoted=%v hash=%s\n",
		result.Games, result.BaselineEval, result.WinsBaseline, result.ChallengerEval, result.WinsChallenger,
		result.Draws, result.WinsBlack, result.WinsWhite,
		result.WinRateChallenger, result.WilsonCILow, result.WilsonCIHigh, result.Promoted, result.ConfigHash)
	if f.arenaJSON == "" {
		fmt.Println(string(data))
	}
}

func runSGFReplay(path string) {
	n, bl, wl, err := ReplaySGFFile(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("replayed %d moves: black=%.1f white=%.1f\n", n, bl, wl)
}

func runGTP(playouts, defaultSize int, think time.Duration, evalMode, sgfOut string) {
	if playouts <= 0 {
		playouts = defaultPlayoutsForSize(defaultSize)
	}
	s := NewSession(SessionConfig{
		Playouts:  playouts,
		ThinkTime: think,
		Eval:      evalMode,
		BoardSize: defaultSize,
	})
	in := bufio.NewScanner(os.Stdin)
	for in.Scan() {
		line := in.Text()
		if strings.TrimSpace(line) == "quit" {
			if sgfOut != "" {
				if err := s.log.WriteSGF(sgfOut); err != nil {
					fmt.Fprintln(os.Stderr, err)
				} else {
					fmt.Fprintf(os.Stderr, "saved %s\n", sgfOut)
				}
			}
			fmt.Print("= \n\n")
			break
		}
		out := s.Handle(line)
		if strings.HasPrefix(out, "?") {
			fmt.Printf("? %s\n\n", out)
		} else {
			fmt.Printf("= %s\n\n", out)
		}
	}
}

func runSelfplayCLI(f cliFlags) {
	playouts := f.playouts
	if playouts <= 0 {
		playouts = defaultPlayoutsForSize(f.size)
	}
	cfg := DefaultSelfplayConfig()
	cfg.Games = f.games
	cfg.BoardSize = f.size
	cfg.Komi = f.komi
	cfg.Playouts = playouts
	cfg.FullOnlyExport = f.selfplayFullOnly
	cfg.EvalMode = f.selfplayEval
	cfg.ONNXFraction = f.selfplayONNXFraction
	cfg.Seed = f.seed
	cfg.Parallel = f.selfplayParallel
	cfg.TemperatureMoves = f.selfplayTempMoves
	if f.selfplayFixedPlayouts {
		cfg.FastPlayouts = playouts
		cfg.FullPlayouts = playouts
		cfg.CapRandomizeP = 0
	} else {
		cfg.FastPlayouts = f.selfplayFastPlayouts
		cfg.FullPlayouts = f.selfplayFullPlayouts
		cfg.CapRandomizeP = f.selfplayCapRandomizeP
	}
	samples, logs := RunSelfplayWithLogs(cfg)
	if f.out != "" {
		writeSelfplayJSON(f.out, samples)
	} else if f.sgfDir == "" {
		data, _ := MarshalSampleExport(samples)
		fmt.Println(string(data))
	}
	if f.sgfDir != "" {
		writeSelfplaySGFs(f.sgfDir, logs)
	}
}

func writeSelfplayJSON(path string, samples []Sample) {
	if isJSONLPath(path) {
		if err := WriteSampleJSONL(path, samples); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	data, err := MarshalSampleExport(samples)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func writeSelfplaySGFs(dir string, logs []*GameLog) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	for i, log := range logs {
		path := fmt.Sprintf("%s/game-%03d.sgf", dir, i+1)
		if err := log.WriteSGF(path); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	fmt.Printf("wrote %d SGF files to %s\n", len(logs), dir)
}
