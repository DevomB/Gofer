// Package sgf reads and writes the subset of SGF (FF[4]) that Gofer needs:
// board size, komi, setup stones, side to move, result and the main line.
package sgf

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/DevomB/gofer/internal/game"
)

// Node is one SGF node with its properties and child variations.
type Node struct {
	Props    map[string][]string
	Children []*Node
}

// Move is one main-line move.
type Move struct {
	Color game.Color
	Point game.Point // game.Pass for passes
}

// Game is a parsed SGF game tree.
type Game struct {
	Size   int
	Komi   float64
	Result string
	Root   *Node
}

// Parse parses an SGF document and returns its first game tree.
func Parse(data string) (*Game, error) {
	p := &parser{s: data}
	p.skipWS()
	if p.peek() != '(' {
		return nil, fmt.Errorf("sgf: expected '(' at start")
	}
	root, err := p.sequence()
	if err != nil {
		return nil, err
	}
	g := &Game{Size: 19, Komi: 7.5, Root: root}
	if root != nil {
		if v := root.Get("SZ"); v != "" {
			n, err := strconv.Atoi(strings.SplitN(v, ":", 2)[0])
			if err != nil || n < 2 || n > game.MaxSize {
				return nil, fmt.Errorf("sgf: bad SZ[%s]", v)
			}
			g.Size = n
		}
		if v := root.Get("KM"); v != "" {
			k, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return nil, fmt.Errorf("sgf: bad KM[%s]", v)
			}
			g.Komi = k
		}
		g.Result = root.Get("RE")
	}
	return g, nil
}

// Get returns the first value of a property, or "".
func (n *Node) Get(key string) string {
	if v := n.Props[key]; len(v) > 0 {
		return v[0]
	}
	return ""
}

// Coord decodes an SGF coordinate ("aa" = top-left). An empty value or "tt"
// on boards up to 19x19 is a pass.
func Coord(s string, size int) (game.Point, error) {
	if s == "" || (s == "tt" && size <= 19) {
		return game.Pass, nil
	}
	if len(s) != 2 {
		return game.Pass, fmt.Errorf("sgf: coord %q", s)
	}
	x, y := int(s[0]-'a'), int(s[1]-'a')
	if x < 0 || y < 0 || x >= size || y >= size {
		return game.Pass, fmt.Errorf("sgf: coord %q off %dx%d board", s, size, size)
	}
	return game.Point(y*size + x), nil
}

func coordString(p game.Point, size int) string {
	if p == game.Pass {
		return ""
	}
	return string([]byte{'a' + byte(int(p)%size), 'a' + byte(int(p)/size)})
}

// MainLine returns moves along the first variation.
func (g *Game) MainLine() ([]Move, error) {
	var out []Move
	for n := g.Root; n != nil; {
		for _, tag := range []string{"B", "W"} {
			c := game.Black
			if tag == "W" {
				c = game.White
			}
			for _, v := range n.Props[tag] {
				p, err := Coord(v, g.Size)
				if err != nil {
					return nil, err
				}
				out = append(out, Move{Color: c, Point: p})
			}
		}
		if len(n.Children) == 0 {
			break
		}
		n = n.Children[0]
	}
	return out, nil
}

// Board builds the initial position (setup stones and side to move).
func (g *Game) Board(superko bool) (*game.Board, error) {
	b := game.NewBoard(game.Options{Size: g.Size, Komi: g.Komi, Superko: superko})
	if g.Root == nil {
		return b, nil
	}
	var black, white []game.Point
	for tag, dst := range map[string]*[]game.Point{"AB": &black, "AW": &white} {
		for _, v := range g.Root.Props[tag] {
			p, err := Coord(v, g.Size)
			if err != nil || p == game.Pass {
				return nil, fmt.Errorf("sgf: bad %s[%s]", tag, v)
			}
			*dst = append(*dst, p)
		}
	}
	if len(black)+len(white) > 0 {
		if err := b.Setup(black, white); err != nil {
			return nil, err
		}
	}
	if g.Root.Get("PL") == "W" || g.Root.Get("PL") == "2" {
		b.SetToPlay(game.White)
	}
	return b, nil
}

// Replay plays the main line and returns the final board.
func (g *Game) Replay() (*game.Board, []Move, error) {
	b, err := g.Board(true)
	if err != nil {
		return nil, nil, err
	}
	moves, err := g.MainLine()
	if err != nil {
		return nil, nil, err
	}
	for i, m := range moves {
		if b.ToPlay() != m.Color {
			b.SetToPlay(m.Color) // SGF allows consecutive moves by one color
		}
		if err := b.Play(m.Point); err != nil {
			return nil, nil, fmt.Errorf("sgf: move %d: %w", i+1, err)
		}
	}
	return b, moves, nil
}

// Record describes a finished game for export.
type Record struct {
	Size        int
	Komi        float64
	Black       string
	White       string
	Result      string
	Moves       []Move
	MoveComment map[int]string // optional per-move comments, keyed by move index
}

// Write renders a record as SGF.
func Write(r Record) string {
	var b strings.Builder
	fmt.Fprintf(&b, "(;FF[4]GM[1]SZ[%d]KM[%s]RU[Chinese]", r.Size, strconv.FormatFloat(r.Komi, 'f', -1, 64))
	if r.Black != "" {
		fmt.Fprintf(&b, "PB[%s]", escape(r.Black))
	}
	if r.White != "" {
		fmt.Fprintf(&b, "PW[%s]", escape(r.White))
	}
	if r.Result != "" {
		fmt.Fprintf(&b, "RE[%s]", escape(r.Result))
	}
	for i, m := range r.Moves {
		fmt.Fprintf(&b, ";%s[%s]", m.Color, coordString(m.Point, r.Size))
		if c, ok := r.MoveComment[i]; ok {
			fmt.Fprintf(&b, "C[%s]", escape(c))
		}
	}
	b.WriteString(")\n")
	return b.String()
}

func escape(s string) string {
	return strings.NewReplacer(`\`, `\\`, `]`, `\]`).Replace(s)
}

type parser struct {
	s string
	i int
}

func (p *parser) eof() bool { return p.i >= len(p.s) }

func (p *parser) peek() byte {
	if p.eof() {
		return 0
	}
	return p.s[p.i]
}

func (p *parser) skipWS() {
	for !p.eof() && strings.IndexByte(" \t\r\n", p.s[p.i]) >= 0 {
		p.i++
	}
}

// sequence parses "(" node* subtree* ")" and returns its first node.
func (p *parser) sequence() (*Node, error) {
	p.i++ // '('
	var first, cur *Node
	for {
		p.skipWS()
		if p.eof() {
			return nil, fmt.Errorf("sgf: unclosed '('")
		}
		switch p.peek() {
		case ';':
			n, err := p.node()
			if err != nil {
				return nil, err
			}
			if first == nil {
				first = n
			} else {
				cur.Children = append(cur.Children, n)
			}
			cur = n
		case '(':
			child, err := p.sequence()
			if err != nil {
				return nil, err
			}
			switch {
			case cur != nil:
				cur.Children = append(cur.Children, child)
			case first == nil:
				first = child
			}
		case ')':
			p.i++
			return first, nil
		default:
			return nil, fmt.Errorf("sgf: unexpected %q at offset %d", p.peek(), p.i)
		}
	}
}

func (p *parser) node() (*Node, error) {
	p.i++ // ';'
	n := &Node{Props: map[string][]string{}}
	for {
		p.skipWS()
		c := p.peek()
		if c < 'A' || c > 'Z' {
			return n, nil
		}
		start := p.i
		for !p.eof() && p.s[p.i] >= 'A' && p.s[p.i] <= 'Z' {
			p.i++
		}
		key := p.s[start:p.i]
		p.skipWS()
		if p.peek() != '[' {
			return nil, fmt.Errorf("sgf: property %s without value", key)
		}
		for p.skipWS(); p.peek() == '['; p.skipWS() {
			v, err := p.value()
			if err != nil {
				return nil, err
			}
			n.Props[key] = append(n.Props[key], v)
		}
	}
}

func (p *parser) value() (string, error) {
	p.i++ // '['
	var b strings.Builder
	for !p.eof() {
		c := p.s[p.i]
		switch {
		case c == ']':
			p.i++
			return b.String(), nil
		case c == '\\' && p.i+1 < len(p.s):
			b.WriteByte(p.s[p.i+1])
			p.i += 2
		default:
			b.WriteByte(c)
			p.i++
		}
	}
	return "", fmt.Errorf("sgf: unclosed '['")
}
