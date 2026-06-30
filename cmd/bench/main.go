// Command bench runs package benchmarks and optional JSON regression output.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
)

type benchResult struct {
	Name    string  `json:"name"`
	NsPerOp float64 `json:"ns_per_op"`
	Bytes   int64   `json:"bytes_per_op"`
	Allocs  int64   `json:"allocs_per_op"`
}

type report struct {
	Timestamp string        `json:"timestamp"`
	Results   []benchResult `json:"results"`
}

const (
	defaultRegressionThreshold = 1.10 // 10% slower allowed vs baseline
	microBenchThreshold        = 1.25 // looser for sub-microsecond benches (Windows noise)
	microBenchNs               = 1000
	benchSampleRuns            = 3 // max-of-N dampens noise; pinned CI runner would be tighter
)

func regressionLimit(baselineNs float64) float64 {
	if baselineNs < microBenchNs {
		return microBenchThreshold
	}
	return defaultRegressionThreshold
}

func main() {
	jsonOut := flag.String("json", "", "write regression JSON to path")
	baselinePath := flag.String("baseline", "", "baseline JSON for -check")
	check := flag.Bool("check", false, "fail if >10% regression vs -baseline")
	flag.Parse()

	results, err := runBenchesMax(benchSampleRuns)
	if err != nil {
		os.Exit(1)
	}

	if *jsonOut != "" {
		rep := report{
			Timestamp: time.Now().UTC().Format(time.RFC3339),
			Results:   results,
		}
		data, err := json.MarshalIndent(rep, "", "  ")
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if err := os.WriteFile(*jsonOut, data, 0644); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}

	if *check {
		if *baselinePath == "" {
			fmt.Fprintln(os.Stderr, "-check requires -baseline")
			os.Exit(1)
		}
		if err := checkRegression(*baselinePath, results); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
}

func runBenchesMax(runs int) ([]benchResult, error) {
	if runs < 1 {
		runs = 1
	}
	var merged []benchResult
	for i := 0; i < runs; i++ {
		out, err := exec.Command("go", "test", "-bench=.", "-benchmem", "./...").CombinedOutput()
		fmt.Print(string(out))
		if err != nil {
			if exit, ok := err.(*exec.ExitError); ok {
				return nil, fmt.Errorf("bench run %d failed: exit %d", i+1, exit.ExitCode())
			}
			return nil, err
		}
		cur := parseBenchOutput(string(out))
		if i == 0 {
			merged = cur
			continue
		}
		merged = mergeMaxResults(merged, cur)
	}
	return merged, nil
}

func mergeMaxResults(a, b []benchResult) []benchResult {
	bm := make(map[string]benchResult, len(b))
	for _, r := range b {
		bm[r.Name] = r
	}
	out := make([]benchResult, len(a))
	for i, r := range a {
		out[i] = r
		if o, ok := bm[r.Name]; ok && o.NsPerOp > r.NsPerOp {
			out[i] = o
		}
	}
	for _, o := range b {
		found := false
		for _, r := range a {
			if r.Name == o.Name {
				found = true
				break
			}
		}
		if !found {
			out = append(out, o)
		}
	}
	return out
}

func checkRegression(baselinePath string, current []benchResult) error {
	data, err := os.ReadFile(baselinePath)
	if err != nil {
		return fmt.Errorf("read baseline: %w", err)
	}
	var base report
	if err := json.Unmarshal(data, &base); err != nil {
		return fmt.Errorf("parse baseline: %w", err)
	}
	baseMap := make(map[string]benchResult, len(base.Results))
	for _, r := range base.Results {
		baseMap[r.Name] = r
	}
	var regressions []string
	for _, cur := range current {
		b, ok := baseMap[cur.Name]
		if !ok {
			continue
		}
		if b.NsPerOp > 0 && cur.NsPerOp > b.NsPerOp*regressionLimit(b.NsPerOp) {
			regressions = append(regressions,
				fmt.Sprintf("%s: %.0f ns/op > %.0f baseline (%.1f%%)",
					cur.Name, cur.NsPerOp, b.NsPerOp, (cur.NsPerOp/b.NsPerOp-1)*100))
		}
	}
	if len(regressions) > 0 {
		return fmt.Errorf("bench regression:\n  %s", strings.Join(regressions, "\n  "))
	}
	return nil
}

var benchLine = regexp.MustCompile(`^(Benchmark\S+)-\d+\s+\d+\s+(\d+(?:\.\d+)?)\s+ns/op\s+(\d+)\s+B/op\s+(\d+)\s+allocs/op`)

func parseBenchOutput(s string) []benchResult {
	var results []benchResult
	for _, line := range strings.Split(s, "\n") {
		m := benchLine.FindStringSubmatch(strings.TrimSpace(line))
		if m == nil {
			continue
		}
		ns, _ := strconv.ParseFloat(m[2], 64)
		b, _ := strconv.ParseInt(m[3], 10, 64)
		a, _ := strconv.ParseInt(m[4], 10, 64)
		results = append(results, benchResult{
			Name:    m[1],
			NsPerOp: ns,
			Bytes:   b,
			Allocs:  a,
		})
	}
	return results
}
