package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// SGFMove is a color and optional board point from SGF (pass when Point is nil).
type SGFMove struct {
	Color Color
	Point *Point
}

// SGFNode is one SGF node with properties and child variations.
type SGFNode struct {
	Props    map[string][]string
	Children []*SGFNode
}

// SGFGame is a parsed SGF game tree.
type SGFGame struct {
	Size int
	Komi float64
	Root *SGFNode
}

// ParseSGFCoord decodes SGF coords (aa = upper-left).
func ParseSGFCoord(size int, s string) (Point, error) {
	if len(s) != 2 {
		return Point{}, fmt.Errorf("sgf coord %q: want 2 letters", s)
	}
	x := int(s[0] - 'a')
	y := int(s[1] - 'a')
	if x < 0 || y < 0 || x >= size || y >= size {
		return Point{}, fmt.Errorf("sgf coord %q off %dx%d board", s, size, size)
	}
	return At(x, y), nil
}

// ParseSGF parses an SGF file (FF[4] subset: SZ, KM, AB, AW, PL, B, W).
func ParseSGF(data string) (*SGFGame, error) {
	p := newSGFParser(data)
	if err := p.skipWS(); err != nil {
		return nil, err
	}
	if p.peek() != '(' {
		return nil, fmt.Errorf("sgf: expected '(' at start")
	}
	root, err := p.parseSequence()
	if err != nil {
		return nil, err
	}
	g := &SGFGame{Size: 19, Komi: 7.5, Root: root}
	if root != nil {
		g.applyMeta(root)
	}
	return g, nil
}

func (g *SGFGame) applyMeta(n *SGFNode) {
	if sz, ok := n.Props["SZ"]; ok && len(sz) > 0 {
		if v, err := strconv.Atoi(sz[0]); err == nil && v >= 2 {
			g.Size = v
		}
	}
	if km, ok := n.Props["KM"]; ok && len(km) > 0 {
		if v, err := strconv.ParseFloat(km[0], 64); err == nil {
			g.Komi = v
		}
	}
}

// MainLine returns moves along the first child at each variation point.
func (g *SGFGame) MainLine() ([]SGFMove, error) {
	var moves []SGFMove
	if err := g.collectMainLine(g.Root, &moves); err != nil {
		return nil, err
	}
	return moves, nil
}

func (g *SGFGame) collectMainLine(n *SGFNode, moves *[]SGFMove) error {
	if n == nil {
		return nil
	}
	if err := appendSGFMoveProps(g.Size, n.Props, moves); err != nil {
		return err
	}
	if len(n.Children) > 0 {
		return g.collectMainLine(n.Children[0], moves)
	}
	return nil
}

// sgfMoveToPlay maps an SGF move to an engine Move (pass when Point is nil).
func sgfMoveToPlay(m SGFMove) Move {
	if m.Point == nil {
		return PassMove
	}
	return StoneMove(*m.Point)
}

func appendSGFMoveProps(size int, props map[string][]string, moves *[]SGFMove) error {
	for tag, vals := range props {
		if tag != "B" && tag != "W" {
			continue
		}
		c := Black
		if tag == "W" {
			c = White
		}
		for _, v := range vals {
			if v == "" {
				*moves = append(*moves, SGFMove{Color: c})
				continue
			}
			pt, err := ParseSGFCoord(size, v)
			if err != nil {
				return err
			}
			*moves = append(*moves, SGFMove{Color: c, Point: &pt})
		}
	}
	return nil
}

// SetupSGF applies AB/AW/PL from the root node onto b.
func (g *SGFGame) Setup(b *Board) error {
	if g.Root == nil {
		return nil
	}
	n := g.Root
	for _, coord := range n.Props["AB"] {
		pt, err := ParseSGFCoord(g.Size, coord)
		if err != nil {
			return err
		}
		b.SetStoneIndex(pt.Idx(g.Size), Black)
	}
	for _, coord := range n.Props["AW"] {
		pt, err := ParseSGFCoord(g.Size, coord)
		if err != nil {
			return err
		}
		b.SetStoneIndex(pt.Idx(g.Size), White)
	}
	if pl, ok := n.Props["PL"]; ok && len(pl) > 0 && pl[0] == "2" {
		b.SetPlayer(White)
	}
	b.Rehash()
	return nil
}

// ExportSGF writes a minimal SGF for the main line.
func ExportSGF(g *SGFGame, moves []SGFMove) string {
	var b strings.Builder
	b.WriteString("(;FF[4]")
	fmt.Fprintf(&b, "SZ[%d]", g.Size)
	fmt.Fprintf(&b, "KM[%.1f]", g.Komi)
	for _, m := range moves {
		tag := 'B'
		if m.Color == White {
			tag = 'W'
		}
		b.WriteByte(';')
		b.WriteByte(byte(tag))
		b.WriteByte('[')
		if m.Point != nil {
			p := *m.Point
			b.WriteByte('a' + byte(p.X))
			b.WriteByte('a' + byte(p.Y))
		}
		b.WriteByte(']')
	}
	b.WriteString(")")
	return b.String()
}

// ReplaySGFFile replays path under Chinese rules and returns move count and score.
func ReplaySGFFile(path string) (moveCount int, black, white float64, err error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, 0, 0, err
	}
	g, err := ParseSGF(string(data))
	if err != nil {
		return 0, 0, 0, err
	}
	r := Chinese()
	b := NewBoard(g.Size, g.Komi)
	if err := g.Setup(b); err != nil {
		return 0, 0, 0, err
	}
	moves, err := g.MainLine()
	if err != nil {
		return 0, 0, 0, err
	}
	for i, m := range moves {
		if b.Player() != m.Color {
			return i, 0, 0, fmt.Errorf("move %d wrong side", i)
		}
		var play Move
		if m.Point == nil {
			play = PassMove
		} else {
			play = StoneMove(*m.Point)
		}
		if !r.Play(b, play) {
			return i, 0, 0, fmt.Errorf("illegal move %d", i)
		}
	}
	bl, wl := r.Score(b)
	_ = ExportSGF(g, moves)
	return len(moves), bl, wl, nil
}
