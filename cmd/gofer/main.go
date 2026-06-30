package main

func main() {
	f := parseCLIFlags()
	if f.sgfPath != "" {
		runSGFReplay(f.sgfPath)
		return
	}
	if runMode(f) {
		return
	}
	printUsage(f.size, f.komi)
}
