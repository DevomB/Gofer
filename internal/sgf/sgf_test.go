package sgf_test

import (
	"testing"

	"github.com/DevomB/gofer/internal/sgf"
)

func TestParseCoord(t *testing.T) {
	p, err := sgf.ParseCoord(9, "bc")
	if err != nil {
		t.Fatal(err)
	}
	if p.X != 1 || p.Y != 2 {
		t.Fatalf("bc on 9x9 want (1,2) got (%d,%d)", p.X, p.Y)
	}
}
