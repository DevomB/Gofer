// Command bench runs package benchmarks and optional JSON regression output (M9).
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

func main() {
	jsonOut := flag.String("json", "", "write regression JSON to path")
	flag.Parse()

	cmd := exec.Command("go", "test", "-bench=.", "-benchmem", "./...")
	out, err := cmd.CombinedOutput()
	fmt.Print(string(out))
	if err != nil {
		if exit, ok := err.(*exec.ExitError); ok {
			os.Exit(exit.ExitCode())
		}
		os.Exit(1)
	}

	if *jsonOut != "" {
		results := parseBenchOutput(string(out))
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
