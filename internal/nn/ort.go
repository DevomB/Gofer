//go:build onnx

package nn

import (
	"fmt"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// ORTAvailable reports whether this binary was built with ONNX Runtime.
const ORTAvailable = true

var (
	ortOnce sync.Once
	ortErr  error
)

func initORT() error {
	ortOnce.Do(func() {
		lib := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH")
		if lib == "" {
			ortErr = fmt.Errorf("nn: set ONNXRUNTIME_SHARED_LIBRARY_PATH to libonnxruntime.so")
			return
		}
		ort.SetSharedLibraryPath(lib)
		ortErr = ort.InitializeEnvironment()
	})
	return ortErr
}

// ORTOptions selects the execution provider for an ONNX model.
type ORTOptions struct {
	Device     string // "cpu", "cuda" or "tensorrt"
	DeviceID   int
	CPUThreads int // intra-op threads per session (CPU)
}

type ortBackend struct {
	sess   *ort.DynamicAdvancedSession
	size   int
	points int
}

// NewORTBackend loads an exported Gofer model (inputs spatial/global,
// outputs policy/value) for a board of the given size.
func NewORTBackend(path string, size int, o ORTOptions) (Backend, error) {
	if err := initORT(); err != nil {
		return nil, err
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()
	threads := o.CPUThreads
	if threads < 1 {
		threads = 1
	}
	if err := opts.SetIntraOpNumThreads(threads); err != nil {
		return nil, err
	}
	if err := opts.SetInterOpNumThreads(1); err != nil {
		return nil, err
	}
	switch o.Device {
	case "", "cpu":
	case "cuda":
		cuda, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return nil, fmt.Errorf("nn: CUDA provider unavailable (GPU build of ONNX Runtime required): %w", err)
		}
		defer cuda.Destroy()
		if err := cuda.Update(map[string]string{"device_id": fmt.Sprint(o.DeviceID)}); err != nil {
			return nil, err
		}
		if err := opts.AppendExecutionProviderCUDA(cuda); err != nil {
			return nil, err
		}
	case "tensorrt":
		trt, err := ort.NewTensorRTProviderOptions()
		if err != nil {
			return nil, fmt.Errorf("nn: TensorRT provider unavailable: %w", err)
		}
		defer trt.Destroy()
		if err := trt.Update(map[string]string{"device_id": fmt.Sprint(o.DeviceID), "trt_fp16_enable": "1"}); err != nil {
			return nil, err
		}
		if err := opts.AppendExecutionProviderTensorRT(trt); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("nn: unknown device %q", o.Device)
	}
	sess, err := ort.NewDynamicAdvancedSession(path, []string{"spatial", "global"}, []string{"policy", "value"}, opts)
	if err != nil {
		return nil, fmt.Errorf("nn: load %s: %w", path, err)
	}
	return &ortBackend{sess: sess, size: size, points: size * size}, nil
}

func (b *ortBackend) Run(spatial, global []float32, n int) ([]float32, []float32, error) {
	side := int64(b.size)
	sp, err := ort.NewTensor(ort.NewShape(int64(n), NumPlanes, side, side), spatial)
	if err != nil {
		return nil, nil, err
	}
	defer sp.Destroy()
	gl, err := ort.NewTensor(ort.NewShape(int64(n), NumGlobals), global)
	if err != nil {
		return nil, nil, err
	}
	defer gl.Destroy()
	pol, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(n), int64(b.points+1)))
	if err != nil {
		return nil, nil, err
	}
	defer pol.Destroy()
	val, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(n)))
	if err != nil {
		return nil, nil, err
	}
	defer val.Destroy()
	if err := b.sess.Run([]ort.Value{sp, gl}, []ort.Value{pol, val}); err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), pol.GetData()...), append([]float32(nil), val.GetData()...), nil
}

func (b *ortBackend) Close() error { return b.sess.Destroy() }
