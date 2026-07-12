package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	defaultPolicyEpsilon = 0.10
	komiMin              = 6.5
	komiMax              = 7.5
	badOpeningPlies      = 16
)

// SGFConvertConfig controls supervised SGF → JSONL conversion.
type SGFConvertConfig struct {
	OutPath      string
	MaxRows      int
	PerSourceCap int // 0 = auto equal split when MaxRows > 0
	Epsilon      float64
	BoardSize    int
}

// SGFConvertStats aggregates conversion outcomes.
type SGFConvertStats struct {
	FilesSeen      int
	GamesAccepted  int
	GamesRejected  int
	RowsWritten    int
	RowsSkippedDup int
	PerSourceCap   int
	RejectReason   map[string]int
	SourceGames    map[string]int
	SourceRows     map[string]int
}

func newSGFConvertStats() *SGFConvertStats {
	return &SGFConvertStats{
		RejectReason: make(map[string]int),
		SourceGames:  make(map[string]int),
		SourceRows:   make(map[string]int),
	}
}

func (s *SGFConvertStats) reject(reason string) {
	s.GamesRejected++
	s.RejectReason[reason]++
}

// SampleJSONLWriter streams schema-v1 training rows to JSONL.
type SampleJSONLWriter struct {
	f   *os.File
	w   *bufio.Writer
	n   int
	err error
}

// NewSampleJSONLWriter creates path and writes the header line.
func NewSampleJSONLWriter(path string) (*SampleJSONLWriter, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return nil, err
	}
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	sw := &SampleJSONLWriter{f: f, w: bufio.NewWriter(f)}
	header, err := json.Marshal(map[string]any{
		"schema_version": SampleSchemaVersion,
		"git_commit":     buildInfoVersion(),
		"type":           "header",
	})
	if err != nil {
		f.Close()
		return nil, err
	}
	if _, err := sw.w.Write(header); err != nil {
		f.Close()
		return nil, err
	}
	if _, err := sw.w.WriteString("\n"); err != nil {
		f.Close()
		return nil, err
	}
	return sw, nil
}

// WriteSample appends one training row.
func (sw *SampleJSONLWriter) WriteSample(s Sample) error {
	if sw.err != nil {
		return sw.err
	}
	line, err := json.Marshal(s)
	if err != nil {
		sw.err = err
		return err
	}
	if _, err := sw.w.Write(line); err != nil {
		sw.err = err
		return err
	}
	if _, err := sw.w.WriteString("\n"); err != nil {
		sw.err = err
		return err
	}
	sw.n++
	return nil
}

// Rows returns rows written (excluding header).
func (sw *SampleJSONLWriter) Rows() int { return sw.n }

// Close flushes and closes the file.
func (sw *SampleJSONLWriter) Close() error {
	if sw.err != nil {
		_ = sw.w.Flush()
		_ = sw.f.Close()
		return sw.err
	}
	if err := sw.w.Flush(); err != nil {
		_ = sw.f.Close()
		return err
	}
	return sw.f.Close()
}

type dedupKey struct {
	hash   uint64
	toPlay Color
}

func smoothedPolicyTarget(legal []Move, played Move, size int, eps float64) []float32 {
	n := size*size + 1
	out := make([]float32, n)
	if len(legal) == 0 {
		return out
	}
	playedIdx := -1
	for i, m := range legal {
		if movesEqual(m, played) {
			playedIdx = i
			break
		}
	}
	if playedIdx < 0 {
		return nil
	}
	if len(legal) == 1 {
		out[policyIndex(legal[0], size)] = 1
		return out
	}
	otherShare := float32(eps) / float32(len(legal)-1)
	peak := float32(1.0 - eps)
	for i, m := range legal {
		idx := policyIndex(m, size)
		if i == playedIdx {
			out[idx] = peak
		} else {
			out[idx] = otherShare
		}
	}
	return out
}

func komiInRange(k float64) bool {
	return k >= komiMin-1e-9 && k <= komiMax+1e-9
}

func isNineCorner(p Point, size int) bool {
	if size != 9 {
		return false
	}
	last := size - 1
	return (p.X == 0 || p.X == last) && (p.Y == 0 || p.Y == last)
}

func hasBadOpening(moves []SGFMove, size int) bool {
	limit := badOpeningPlies
	if len(moves) < limit {
		limit = len(moves)
	}
	for i := 0; i < limit; i++ {
		m := moves[i]
		if m.Point != nil && isNineCorner(*m.Point, size) {
			return true
		}
	}
	return false
}

func sgfHasHandicap(root *SGFNode) bool {
	if root == nil {
		return false
	}
	if ha, ok := root.Props["HA"]; ok && len(ha) > 0 {
		if n, err := strconv.Atoi(ha[0]); err == nil && n >= 1 {
			return true
		}
		if ha[0] != "" && ha[0] != "0" {
			return true
		}
	}
	if ab, ok := root.Props["AB"]; ok && len(ab) > 0 {
		return true
	}
	if aw, ok := root.Props["AW"]; ok && len(aw) > 0 {
		return true
	}
	return false
}

func findSGFProp(n *SGFNode, key string) string {
	if n == nil {
		return ""
	}
	if vals, ok := n.Props[key]; ok && len(vals) > 0 && vals[0] != "" {
		return vals[0]
	}
	for _, c := range n.Children {
		if v := findSGFProp(c, key); v != "" {
			return v
		}
	}
	return ""
}

func parseSGFResult(re string) (winner Color, ok bool) {
	re = strings.TrimSpace(re)
	if re == "" {
		return Empty, false
	}
	upper := strings.ToUpper(re)
	if upper == "0" || upper == "DRAW" || strings.HasPrefix(upper, "DRAW") {
		return Empty, true
	}
	if strings.HasPrefix(upper, "B") {
		return Black, true
	}
	if strings.HasPrefix(upper, "W") {
		return White, true
	}
	return Empty, false
}

// ConvertSGFBytes turns one SGF record into training samples (no dedup).
func ConvertSGFBytes(data []byte, cfg SGFConvertConfig) ([]Sample, string, error) {
	g, err := ParseSGF(string(data))
	if err != nil {
		return nil, "parse", err
	}
	if g.Size != cfg.BoardSize {
		return nil, "size", fmt.Errorf("size %d", g.Size)
	}
	if !komiInRange(g.Komi) {
		return nil, "komi", fmt.Errorf("komi %.2f", g.Komi)
	}
	if sgfHasHandicap(g.Root) {
		return nil, "handicap", fmt.Errorf("handicap setup")
	}
	re := findSGFProp(g.Root, "RE")
	if _, ok := parseSGFResult(re); !ok {
		return nil, "result", fmt.Errorf("re %q", re)
	}
	moves, err := g.MainLine()
	if err != nil {
		return nil, "mainline", err
	}
	if hasBadOpening(moves, g.Size) {
		return nil, "bad_opening", fmt.Errorf("corner in opening")
	}

	rs := Chinese()
	b := NewBoard(g.Size, g.Komi)
	if err := g.Setup(b); err != nil {
		return nil, "setup", err
	}

	eps := cfg.Epsilon
	if eps <= 0 {
		eps = defaultPolicyEpsilon
	}

	var samples []Sample
	for moveNum, sm := range moves {
		if b.Player() != sm.Color {
			return nil, "side", fmt.Errorf("move %d wrong side", moveNum)
		}
		legal := rs.LegalMoves(b)
		if onlyPass(legal) {
			break
		}
		play := sgfMoveToPlay(sm)
		policy := smoothedPolicyTarget(legal, play, g.Size, eps)
		if policy == nil {
			return nil, "illegal", fmt.Errorf("illegal move %d", moveNum)
		}
		spatial, globals := BuildFeaturesV2(b)
		samples = append(samples, Sample{
			BoardHash:       b.Hash(),
			MoveNum:         moveNum,
			Policy:          policy,
			FeaturesSpatial: spatial,
			FeaturesGlobal:  globals,
			ToPlay:          b.Player(),
			Komi:            g.Komi,
		})
		if !rs.Play(b, play) {
			return nil, "illegal", fmt.Errorf("replay illegal %d", moveNum)
		}
	}

	bl, wl := rs.Score(b)
	ownership := OwnershipLabel(b)
	labelGameSamples(samples, bl, wl, ownership)
	return samples, "", nil
}

func inferSourceLabel(path string) string {
	p := strings.ToLower(filepath.ToSlash(path))
	switch {
	case strings.Contains(p, "/cgos/"):
		return "cgos"
	case strings.Contains(p, "/minigo/"):
		return "minigo"
	case strings.Contains(p, "/aeb/"):
		return "aeb"
	default:
		return filepath.Base(filepath.Dir(path))
	}
}

type convertState struct {
	cfg          SGFConvertConfig
	dedup        map[dedupKey]struct{}
	stats        *SGFConvertStats
	w            *SampleJSONLWriter
	sourceRows   map[string]int
	perSourceCap int
}

func (st *convertState) sourceAtCap(source string) bool {
	if st.perSourceCap <= 0 {
		return false
	}
	return st.sourceRows[source] >= st.perSourceCap
}

func (st *convertState) globalAtCap() bool {
	return st.cfg.MaxRows > 0 && st.stats.RowsWritten >= st.cfg.MaxRows
}

func equalPerSourceCap(maxRows, numSources int) int {
	if maxRows <= 0 || numSources <= 0 {
		return 0
	}
	return (maxRows + numSources - 1) / numSources
}

type sourceQueue struct {
	label string
	paths []string
	idx   int
}

func collectSGFSourceQueues(inputs []string) ([]sourceQueue, error) {
	var queues []sourceQueue
	for _, root := range inputs {
		label := inferSourceLabel(filepath.ToSlash(root) + "/stub.sgf")
		paths, err := listSGFFiles(root)
		if err != nil {
			return nil, err
		}
		if len(paths) == 0 {
			continue
		}
		queues = append(queues, sourceQueue{label: label, paths: paths})
	}
	return queues, nil
}

func listSGFFiles(root string) ([]string, error) {
	info, err := os.Stat(root)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		if strings.HasSuffix(strings.ToLower(root), ".sgf") {
			return []string{root}, nil
		}
		return nil, nil
	}
	var paths []string
	err = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.HasSuffix(strings.ToLower(path), ".sgf") {
			return nil
		}
		paths = append(paths, path)
		return nil
	})
	return paths, err
}

func (st *convertState) convertSGFFile(path, source string) error {
	if st.globalAtCap() || st.sourceAtCap(source) {
		return errStopConvert
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	st.stats.FilesSeen++
	samples, reason, err := ConvertSGFBytes(data, st.cfg)
	if err != nil {
		if reason == "" {
			reason = "error"
		}
		st.stats.reject(reason)
		return nil
	}
	st.stats.GamesAccepted++
	st.stats.SourceGames[source]++
	for _, s := range samples {
		if st.globalAtCap() || st.sourceAtCap(source) {
			return errStopConvert
		}
		key := dedupKey{hash: s.BoardHash, toPlay: s.ToPlay}
		if _, ok := st.dedup[key]; ok {
			st.stats.RowsSkippedDup++
			continue
		}
		st.dedup[key] = struct{}{}
		if err := st.w.WriteSample(s); err != nil {
			return err
		}
		st.stats.RowsWritten++
		st.sourceRows[source]++
		st.stats.SourceRows[source]++
	}
	return nil
}

func runBalancedSGFConvert(cfg SGFConvertConfig, inputs []string) (*SGFConvertStats, error) {
	queues, err := collectSGFSourceQueues(inputs)
	if err != nil {
		return nil, err
	}
	if len(queues) == 0 {
		return nil, fmt.Errorf("convert: no SGF files under inputs")
	}
	perSourceCap := cfg.PerSourceCap
	if perSourceCap <= 0 && cfg.MaxRows > 0 {
		perSourceCap = equalPerSourceCap(cfg.MaxRows, len(queues))
	}
	stats := newSGFConvertStats()
	stats.PerSourceCap = perSourceCap
	st := &convertState{
		cfg:          cfg,
		dedup:        make(map[dedupKey]struct{}),
		stats:        stats,
		sourceRows:   make(map[string]int),
		perSourceCap: perSourceCap,
	}
	w, err := NewSampleJSONLWriter(cfg.OutPath)
	if err != nil {
		return nil, err
	}
	st.w = w
	defer w.Close()

	for {
		progress := false
		for i := range queues {
			q := &queues[i]
			if st.sourceAtCap(q.label) || q.idx >= len(q.paths) {
				continue
			}
			progress = true
			path := q.paths[q.idx]
			q.idx++
			if err := st.convertSGFFile(path, q.label); err != nil {
				if err == errStopConvert {
					if st.globalAtCap() {
						_ = w.Close()
						return stats, nil
					}
					continue
				}
				return stats, err
			}
			if st.globalAtCap() {
				_ = w.Close()
				return stats, nil
			}
		}
		if !progress {
			break
		}
	}
	if err := w.Close(); err != nil {
		return stats, err
	}
	return stats, nil
}

// RunSGFConvert walks inputs round-robin by source with equal per-source caps.
func RunSGFConvert(cfg SGFConvertConfig, inputs []string) (*SGFConvertStats, error) {
	if cfg.OutPath == "" {
		return nil, fmt.Errorf("convert: missing output path")
	}
	if cfg.BoardSize <= 0 {
		cfg.BoardSize = 9
	}
	if cfg.Epsilon <= 0 {
		cfg.Epsilon = defaultPolicyEpsilon
	}
	return runBalancedSGFConvert(cfg, inputs)
}

var errStopConvert = fmt.Errorf("max rows reached")

// VerifyJSONLSamples checks structural invariants on converted JSONL rows.
func VerifyJSONLSamples(path string, spot int) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	const maxLine = 16 << 20
	buf := make([]byte, 0, 1024*1024)
	sc.Buffer(buf, maxLine)
	rows := 0
	var examples []Sample
	for sc.Scan() {
		var row map[string]json.RawMessage
		if err := json.Unmarshal(sc.Bytes(), &row); err != nil {
			return err
		}
		if typ, _ := parseJSONString(row["type"]); typ == "header" {
			continue
		}
		var s Sample
		if err := json.Unmarshal(sc.Bytes(), &s); err != nil {
			return err
		}
		if err := verifySample(s); err != nil {
			return fmt.Errorf("row %d: %w", rows+1, err)
		}
		if len(examples) < spot {
			examples = append(examples, s)
		}
		rows++
	}
	if err := sc.Err(); err != nil {
		return err
	}
	if rows == 0 {
		return fmt.Errorf("no sample rows in %s", path)
	}
	for i, s := range examples {
		peak, share := policyPeak(s.Policy)
		fmt.Printf("spot[%d] move=%d to_play=%d value=%.0f komi=%.1f peak_idx=%d peak=%.3f\n",
			i, s.MoveNum, s.ToPlay, s.Value, s.Komi, peak, share)
	}
	fmt.Printf("verify-jsonl: %d rows OK\n", rows)
	return nil
}

func parseJSONString(raw json.RawMessage) (string, bool) {
	if len(raw) == 0 {
		return "", false
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return "", false
	}
	return s, true
}

func verifySample(s Sample) error {
	size := 9
	n := size*size + 1
	if len(s.Policy) != n {
		return fmt.Errorf("policy len %d", len(s.Policy))
	}
	if len(s.FeaturesSpatial) != featurePlanesV2*size*size {
		return fmt.Errorf("spatial len %d", len(s.FeaturesSpatial))
	}
	if len(s.FeaturesGlobal) != featureGlobalsV2 {
		return fmt.Errorf("globals len %d", len(s.FeaturesGlobal))
	}
	if len(s.Ownership) != 0 && len(s.Ownership) != size*size {
		return fmt.Errorf("ownership len %d", len(s.Ownership))
	}
	var sum float32
	for _, v := range s.Policy {
		if v < -1e-6 {
			return fmt.Errorf("negative policy %v", v)
		}
		sum += v
	}
	if sum < 0.999 || sum > 1.001 {
		return fmt.Errorf("policy sum %.4f", sum)
	}
	if s.Value != -1 && s.Value != 0 && s.Value != 1 {
		return fmt.Errorf("value %.2f", s.Value)
	}
	if err := verifySpatialPlanes(s.FeaturesSpatial, s.FeaturesGlobal, size); err != nil {
		return err
	}
	return nil
}

func verifySpatialPlanes(spatial []float32, globals []float32, size int) error {
	n := size * size
	for i := 0; i < n; i++ {
		own := spatial[i]
		opp := spatial[n+i]
		empty := spatial[2*n+i]
		if own+opp+empty < 0.99 || own+opp+empty > 1.01 {
			return fmt.Errorf("cell %d stone planes sum %.2f", i, own+opp+empty)
		}
	}
	blackMove := globals[2] > 0.5
	for i := 0; i < n; i++ {
		if spatial[4*n+i] > 0.5 && !blackMove {
			return fmt.Errorf("to-move plane black but globals white")
		}
	}
	return nil
}

func policyPeak(policy []float32) (idx int, share float32) {
	idx = -1
	for i, v := range policy {
		if v > share {
			share = v
			idx = i
		}
	}
	return idx, share
}

func runVerifyJSONLCLI(path string, spot int) {
	if path == "" {
		fmt.Fprintln(os.Stderr, "-verify-jsonl requires -o path")
		os.Exit(1)
	}
	if spot <= 0 {
		spot = 5
	}
	if err := VerifyJSONLSamples(path, spot); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func runSGFConvertCLI(f cliFlags, inputs []string) {
	cfg := SGFConvertConfig{
		OutPath:   f.out,
		MaxRows:   f.convertMaxRows,
		Epsilon:   f.convertEpsilon,
		BoardSize: f.size,
	}
	stats, err := RunSGFConvert(cfg, inputs)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("convert-sgf: files=%d games_ok=%d games_reject=%d rows=%d dup_skip=%d per_source_cap=%d out=%s\n",
		stats.FilesSeen, stats.GamesAccepted, stats.GamesRejected, stats.RowsWritten, stats.RowsSkippedDup,
		stats.PerSourceCap, cfg.OutPath)
	if len(stats.SourceGames) > 0 {
		fmt.Println("source games:", stats.SourceGames)
		fmt.Println("source rows:", stats.SourceRows)
	}
	if len(stats.RejectReason) > 0 {
		fmt.Println("reject reasons:", stats.RejectReason)
	}
}
