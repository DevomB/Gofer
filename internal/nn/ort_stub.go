//go:build !onnx

package nn

import "errors"

// ORTAvailable reports whether this binary was built with ONNX Runtime.
const ORTAvailable = false

// ORTOptions selects the execution provider for an ONNX model.
type ORTOptions struct {
	Device     string
	DeviceID   int
	CPUThreads int
}

// NewORTBackend is unavailable without the onnx build tag.
func NewORTBackend(path string, size int, o ORTOptions) (Backend, error) {
	return nil, errors.New("nn: built without ONNX Runtime; rebuild with `make build` (CGO_ENABLED=1, -tags onnx)")
}
