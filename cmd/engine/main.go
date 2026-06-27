package main

import (
	"flag"
	"fmt"

	"github.com/DevomB/gofer/internal/rules"
)

func main() {
	size := flag.Int("size", 9, "board size")
	komi := flag.Float64("komi", 6.5, "komi")
	flag.Parse()
	b := rules.NewBoard(*size, *komi)
	r := rules.Chinese()
	moves := r.LegalMoves(b)
	fmt.Printf("gofer chinese: %dx%d komi=%.1f, %d legal moves\n",
		*size, *size, *komi, len(moves))
}
