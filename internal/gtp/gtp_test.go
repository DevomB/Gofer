package gtp

import (
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/DevomB/gofer/internal/mcts"
	"github.com/DevomB/gofer/internal/nn"
)

func run(t *testing.T, e *Engine, script string) []string {
	t.Helper()
	var out strings.Builder
	if err := e.Serve(strings.NewReader(script), &out); err != nil {
		t.Fatal(err)
	}
	var resp []string
	for _, block := range strings.Split(strings.TrimSpace(out.String()), "\n\n") {
		resp = append(resp, block)
	}
	return resp
}

func TestGTPSession(t *testing.T) {
	e := NewEngine(Engine{Eval: nn.Uniform{}, Search: mcts.DefaultConfig(), Visits: 50, Superko: true})
	resp := run(t, e, strings.Join([]string{
		"1 protocol_version",
		"2 boardsize 9",
		"3 clear_board",
		"4 komi 7.5",
		"5 play B E5",
		"6 play W E5",
		"7 genmove w",
		"8 undo",
		"9 undo",
		"10 play b D4",
		"11 final_score",
		"12 bogus",
		"13 quit",
	}, "\n"))
	want := []string{"=1 2", "=2", "=3", "=4", "=5", "?6 illegal move", "", "=8", "=9", "=10", "=11 B+73.5", "?12 unknown command", "=13"}
	if len(resp) != len(want) {
		t.Fatalf("got %d responses: %q", len(resp), resp)
	}
	for i, w := range want {
		if w == "" {
			if !strings.HasPrefix(resp[i], "=7 ") {
				t.Fatalf("genmove response %q", resp[i])
			}
			continue
		}
		if strings.TrimSpace(resp[i]) != w {
			t.Fatalf("response %d = %q want %q", i, resp[i], w)
		}
	}
}

func TestLzAnalyzeStreamsUntilNextCommand(t *testing.T) {
	e := NewEngine(Engine{Eval: nn.Uniform{}, Search: mcts.DefaultConfig(), Visits: 50})
	pr, pw := io.Pipe()
	var out safeBuffer
	done := make(chan error, 1)
	go func() { done <- e.Serve(pr, &out) }()
	io.WriteString(pw, "1 lz-analyze b 5\n")
	time.Sleep(300 * time.Millisecond)
	io.WriteString(pw, "2 name\n3 quit\n")
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	s := out.String()
	if !strings.HasPrefix(s, "=1\ninfo move ") || !strings.Contains(s, " pv ") || !strings.Contains(s, "\n\n=2 Gofer\n\n=3") {
		t.Fatalf("unexpected lz-analyze transcript:\n%s", s)
	}
}

type safeBuffer struct {
	mu sync.Mutex
	b  strings.Builder
}

func (s *safeBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *safeBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}
