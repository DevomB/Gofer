package selfplay

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
)

// npyArray is one array destined for an .npz file.
type npyArray struct {
	name  string
	descr string // numpy dtype string, e.g. "|u1", "<f4", "|i1"
	shape []int
	data  any // []uint8, []int8 or []float32
}

// writeNPZ writes arrays as a compressed NumPy .npz archive (readable with np.load).
func writeNPZ(path string, arrays []npyArray) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	zw := zip.NewWriter(f)
	for _, a := range arrays {
		w, err := zw.CreateHeader(&zip.FileHeader{Name: a.name + ".npy", Method: zip.Deflate})
		if err != nil {
			f.Close()
			return err
		}
		if err := writeNPY(w, a); err != nil {
			f.Close()
			return err
		}
	}
	if err := zw.Close(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path) // readers never see a partial shard
}

func writeNPY(w io.Writer, a npyArray) error {
	dims := make([]string, len(a.shape))
	for i, d := range a.shape {
		dims[i] = fmt.Sprint(d)
	}
	shape := strings.Join(dims, ", ")
	if len(a.shape) == 1 {
		shape += ","
	}
	header := fmt.Sprintf("{'descr': '%s', 'fortran_order': False, 'shape': (%s), }", a.descr, shape)
	// magic(6) + version(2) + len(2) + header + '\n' must be a multiple of 64.
	pad := 64 - (10+len(header)+1)%64
	if pad == 64 {
		pad = 0
	}
	header += strings.Repeat(" ", pad) + "\n"
	var buf bytes.Buffer
	buf.WriteString("\x93NUMPY\x01\x00")
	binary.Write(&buf, binary.LittleEndian, uint16(len(header)))
	buf.WriteString(header)
	if _, err := w.Write(buf.Bytes()); err != nil {
		return err
	}
	switch d := a.data.(type) {
	case []uint8:
		_, err := w.Write(d)
		return err
	case []int8:
		b := make([]byte, len(d))
		for i, v := range d {
			b[i] = byte(v)
		}
		_, err := w.Write(b)
		return err
	case []float32:
		return binary.Write(w, binary.LittleEndian, d)
	}
	return fmt.Errorf("npz: unsupported type %T", a.data)
}
