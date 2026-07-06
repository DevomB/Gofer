//go:build onnx

package main

// ORTBackend evaluates positions in-process via ONNX Runtime (see ort_infer.go).
type ORTBackend struct {
	session  *ortSession
	Fallback Evaluator
}

func newORTBackend(modelPath string, fallback Evaluator) (*ORTBackend, error) {
	sess, err := newORTSession(modelPath)
	if err != nil {
		return nil, err
	}
	if fallback == nil {
		fallback = Heuristic{}
	}
	return &ORTBackend{session: sess, Fallback: fallback}, nil
}

// Close releases the ORT session (process-wide ORT env stays up).
func (o *ORTBackend) Close() {
	if o.session != nil {
		o.session.Close()
		o.session = nil
	}
}

// EvalBatch implements EvalBackend.
func (o ORTBackend) EvalBatch(boards []*Board) []Result {
	out := make([]Result, len(boards))
	if len(boards) == 0 {
		return out
	}
	results, err := o.session.evalBatch(boards)
	if err != nil {
		for i, b := range boards {
			out[i] = o.Fallback.Evaluate(b)
		}
		return out
	}
	for i := range boards {
		if i < len(results) && len(results[i].Policy) == boards[i].Size()*boards[i].Size()+1 {
			out[i] = results[i]
		} else {
			out[i] = o.Fallback.Evaluate(boards[i])
		}
	}
	return out
}
