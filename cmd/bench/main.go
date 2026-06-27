// Command bench runs all package benchmarks (M1 harness; full regression in M9).
package main

import (
	"fmt"
	"os"
	"os/exec"
)

func main() {
	cmd := exec.Command("go", "test", "-bench=.", "-benchmem", "./...")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		if exit, ok := err.(*exec.ExitError); ok {
			os.Exit(exit.ExitCode())
		}
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
