package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/DevomB/gofer/internal/gofer"
)

func main() {
	gtpMode := flag.Bool("gtp", false, "run GTP protocol on stdin/stdout")
	size := flag.Int("size", 9, "board size")
	komi := flag.Float64("komi", 6.5, "komi")
	flag.Parse()

	if *gtpMode {
		runGTP()
		return
	}

	b := gofer.NewBoard(*size, *komi)
	r := gofer.Chinese()
	moves := r.LegalMoves(b)
	fmt.Printf("gofer chinese: %dx%d komi=%.1f, %d legal moves\n",
		*size, *size, *komi, len(moves))
}

func runGTP() {
	s := gofer.NewSession()
	in := bufio.NewScanner(os.Stdin)
	for in.Scan() {
		line := in.Text()
		if strings.TrimSpace(line) == "quit" {
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
