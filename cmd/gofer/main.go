// Command gofer is a Go (the board game) engine: GTP server, terminal play,
// self-play data generation, arena matches and benchmarks.
package main

import (
	"fmt"
	"os"
)

var version = "3.0.0-dev"

const usage = `gofer — AlphaZero-style Go engine

Usage:
  gofer gtp      [flags]   serve GTP on stdin/stdout (Sabaki, Lizzie, GoGui)
  gofer play     [flags]   play against the engine in the terminal
  gofer selfplay [flags]   generate training shards (.npz)
  gofer arena    [flags]   play a match between two engines
  gofer bench    [flags]   measure search and network throughput
  gofer version

Run "gofer <command> -h" for the flags of a command.
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	cmds := map[string]func([]string) error{
		"gtp":      cmdGTP,
		"play":     cmdPlay,
		"selfplay": cmdSelfplay,
		"arena":    cmdArena,
		"bench":    cmdBench,
	}
	switch name := os.Args[1]; name {
	case "version", "-version", "--version":
		fmt.Println("gofer", version)
		return
	case "help", "-h", "--help":
		fmt.Print(usage)
		return
	default:
		run, ok := cmds[name]
		if !ok {
			fmt.Fprintf(os.Stderr, "gofer: unknown command %q\n\n%s", name, usage)
			os.Exit(2)
		}
		if err := run(os.Args[2:]); err != nil {
			fmt.Fprintln(os.Stderr, "gofer:", err)
			os.Exit(1)
		}
	}
}
