package main

import (
	"fmt"
	"strings"
)

type sgfParser struct {
	s string
	i int
}

func newSGFParser(data string) *sgfParser {
	return &sgfParser{s: data}
}

func (p *sgfParser) eof() bool { return p.i >= len(p.s) }

func (p *sgfParser) peek() byte {
	if p.eof() {
		return 0
	}
	return p.s[p.i]
}

func (p *sgfParser) skipWS() error {
	for !p.eof() {
		c := p.s[p.i]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			p.i++
			continue
		}
		break
	}
	return nil
}

func (p *sgfParser) parseSequence() (*SGFNode, error) {
	if err := p.skipWS(); err != nil {
		return nil, err
	}
	if p.peek() != '(' {
		return nil, fmt.Errorf("sgf: expected '('")
	}
	p.i++
	return p.parseSequenceBody()
}

func (p *sgfParser) parseSequenceBody() (*SGFNode, error) {
	var first, cur *SGFNode
	for {
		if err := p.skipWS(); err != nil {
			return nil, err
		}
		if p.eof() {
			return nil, fmt.Errorf("sgf: unclosed '('")
		}
		switch p.peek() {
		case ';':
			node, err := p.parseNode()
			if err != nil {
				return nil, err
			}
			first, cur = sgfLinkNode(first, cur, node)
		case '(':
			child, err := p.parseSequence()
			if err != nil {
				return nil, err
			}
			first, cur = sgfLinkChild(first, cur, child)
		case ')':
			p.i++
			return first, nil
		default:
			return nil, fmt.Errorf("sgf: unexpected %q at %d", p.peek(), p.i)
		}
	}
}

func sgfLinkNode(first, cur, node *SGFNode) (*SGFNode, *SGFNode) {
	if first == nil {
		return node, node
	}
	if cur != nil {
		cur.Children = append(cur.Children, node)
	}
	return first, node
}

func sgfLinkChild(first, cur, child *SGFNode) (*SGFNode, *SGFNode) {
	if cur != nil {
		cur.Children = append(cur.Children, child)
		return first, cur
	}
	if first == nil {
		return child, nil
	}
	return first, cur
}

func (p *sgfParser) parseNode() (*SGFNode, error) {
	if p.peek() != ';' {
		return nil, fmt.Errorf("sgf: expected ';'")
	}
	p.i++
	n := &SGFNode{Props: map[string][]string{}}
	for p.hasMoreProps() {
		key, err := p.readPropKey()
		if err != nil {
			return nil, err
		}
		vals, err := p.readPropValues()
		if err != nil {
			return nil, err
		}
		if len(vals) > 0 {
			n.Props[key] = append(n.Props[key], vals...)
		}
	}
	return n, nil
}

func (p *sgfParser) hasMoreProps() bool {
	if err := p.skipWS(); err != nil {
		return false
	}
	if p.eof() {
		return false
	}
	c := p.peek()
	return c != '(' && c != ')' && c != ';' && c >= 'A' && c <= 'Z'
}

func (p *sgfParser) readPropKey() (string, error) {
	if err := p.skipWS(); err != nil {
		return "", err
	}
	start := p.i
	for !p.eof() && p.s[p.i] >= 'A' && p.s[p.i] <= 'Z' {
		p.i++
	}
	if start == p.i {
		return "", fmt.Errorf("sgf: expected property key")
	}
	return p.s[start:p.i], nil
}

func (p *sgfParser) readPropValues() ([]string, error) {
	var vals []string
	for {
		if err := p.skipWS(); err != nil {
			return vals, err
		}
		if p.peek() != '[' {
			break
		}
		val, err := p.parseValue()
		if err != nil {
			return nil, err
		}
		vals = append(vals, val)
	}
	return vals, nil
}

func (p *sgfParser) parseValue() (string, error) {
	if p.peek() != '[' {
		return "", fmt.Errorf("sgf: expected '['")
	}
	p.i++
	var b strings.Builder
	for !p.eof() {
		c := p.s[p.i]
		if c == ']' {
			p.i++
			return b.String(), nil
		}
		if c == '\\' && p.i+1 < len(p.s) {
			p.i++
			b.WriteByte(p.s[p.i])
			p.i++
			continue
		}
		b.WriteByte(c)
		p.i++
	}
	return "", fmt.Errorf("sgf: unclosed '['")
}
