//go:build onnx

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	ortBoardSize  = 9
	ortPlanes     = 8
	ortGlobals    = 4
	ortPolicySize = ortBoardSize*ortBoardSize + 1
)

var (
	ortEnvOnce    sync.Once
	ortEnvInitErr error
)

func resolveORTSharedLibrary() (string, error) {
	if p := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"); p != "" {
		return p, nil
	}
	if runtime.GOOS == "windows" && runtime.GOARCH == "amd64" {
		modRoot := os.Getenv("ONNXRUNTIME_GO_MODROOT")
		if modRoot == "" {
			return "", fmt.Errorf("set ONNXRUNTIME_SHARED_LIBRARY_PATH or ONNXRUNTIME_GO_MODROOT")
		}
		return filepath.Join(modRoot, "test_data", "onnxruntime.dll"), nil
	}
	return "", fmt.Errorf("set ONNXRUNTIME_SHARED_LIBRARY_PATH to libonnxruntime.so.1.26.0")
}

func ensureORTEnv() error {
	ortEnvOnce.Do(func() {
		lib, err := resolveORTSharedLibrary()
		if err != nil {
			ortEnvInitErr = err
			return
		}
		ort.SetSharedLibraryPath(lib)
		ortEnvInitErr = ort.InitializeEnvironment()
	})
	return ortEnvInitErr
}

// ortSession holds one loaded ONNX model session.
type ortSession struct {
	session *ort.DynamicAdvancedSession
}

func newORTSession(modelPath string) (*ortSession, error) {
	if err := ensureORTEnv(); err != nil {
		return nil, fmt.Errorf("ort env: %w", err)
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("session options: %w", err)
	}
	defer opts.Destroy()
	if err := opts.SetIntraOpNumThreads(1); err != nil {
		return nil, err
	}
	if err := opts.SetInterOpNumThreads(1); err != nil {
		return nil, err
	}
	sess, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"spatial_input", "global_input"},
		[]string{"policy_logits", "value"},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("new session: %w", err)
	}
	return &ortSession{session: sess}, nil
}

func (s *ortSession) Close() {
	if s.session != nil {
		s.session.Destroy()
		s.session = nil
	}
}

func (s *ortSession) evalBatch(boards []*Board) ([]Result, error) {
	batch := len(boards)
	if batch == 0 {
		return nil, nil
	}
	spatialFlat := make([]float32, batch*ortPlanes*ortBoardSize*ortBoardSize)
	globalFlat := make([]float32, batch*ortGlobals)
	planeN := ortPlanes * ortBoardSize * ortBoardSize
	for i, b := range boards {
		if b.Size() != ortBoardSize {
			return nil, fmt.Errorf("board size %d, model expects %d", b.Size(), ortBoardSize)
		}
		sp, gl := BuildFeaturesV2(b)
		copy(spatialFlat[i*planeN:(i+1)*planeN], sp)
		copy(globalFlat[i*ortGlobals:(i+1)*ortGlobals], gl)
	}
	spIn, err := ort.NewTensor(ort.NewShape(int64(batch), ortPlanes, ortBoardSize, ortBoardSize), spatialFlat)
	if err != nil {
		return nil, err
	}
	defer spIn.Destroy()
	glIn, err := ort.NewTensor(ort.NewShape(int64(batch), ortGlobals), globalFlat)
	if err != nil {
		return nil, err
	}
	defer glIn.Destroy()
	logitsOut, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(batch), ortPolicySize))
	if err != nil {
		return nil, err
	}
	defer logitsOut.Destroy()
	valueOut, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(batch)))
	if err != nil {
		return nil, err
	}
	defer valueOut.Destroy()
	if err := s.session.Run([]ort.Value{spIn, glIn}, []ort.Value{logitsOut, valueOut}); err != nil {
		return nil, err
	}
	logits := logitsOut.GetData()
	values := valueOut.GetData()
	out := make([]Result, batch)
	for i := 0; i < batch; i++ {
		row := logits[i*ortPolicySize : (i+1)*ortPolicySize]
		out[i] = Result{
			Value:    float64(values[i]),
			Policy:   softmaxPolicy(row),
			HasValue: true,
		}
	}
	return out, nil
}

// evalOne runs a single position (parity harness convenience).
func (s *ortSession) evalOne(spatial, globals []float32) (policy []float32, value float32, err error) {
	if len(spatial) != ortPlanes*ortBoardSize*ortBoardSize {
		return nil, 0, fmt.Errorf("spatial len %d want %d", len(spatial), ortPlanes*ortBoardSize*ortBoardSize)
	}
	if len(globals) != ortGlobals {
		return nil, 0, fmt.Errorf("globals len %d want %d", len(globals), ortGlobals)
	}
	spIn, err := ort.NewTensor(ort.NewShape(1, ortPlanes, ortBoardSize, ortBoardSize), spatial)
	if err != nil {
		return nil, 0, err
	}
	defer spIn.Destroy()
	glIn, err := ort.NewTensor(ort.NewShape(1, ortGlobals), globals)
	if err != nil {
		return nil, 0, err
	}
	defer glIn.Destroy()
	logitsOut, err := ort.NewEmptyTensor[float32](ort.NewShape(1, ortPolicySize))
	if err != nil {
		return nil, 0, err
	}
	defer logitsOut.Destroy()
	valueOut, err := ort.NewEmptyTensor[float32](ort.NewShape(1))
	if err != nil {
		return nil, 0, err
	}
	defer valueOut.Destroy()
	if err := s.session.Run([]ort.Value{spIn, glIn}, []ort.Value{logitsOut, valueOut}); err != nil {
		return nil, 0, err
	}
	logits := logitsOut.GetData()
	value = valueOut.GetData()[0]
	return softmaxPolicy(logits), value, nil
}

// newORTEval opens a session for parity tests (alias).
func newORTEval(modelPath string) (*ortSession, error) {
	return newORTSession(modelPath)
}
