//go:build aria_ffi

package aria

/*
#cgo CFLAGS: -I${SRCDIR}/../../ffi/include
#cgo LDFLAGS: -L${SRCDIR}/../../target/debug -L${SRCDIR}/../../target/release -laria_ffi
#include "aria.h"
#include <stdlib.h>
*/
import "C"
import (
	"encoding/json"
	"errors"
	"os"
	"unsafe"
)

type Engine struct {
	h *C.AriaModel
}

func Open(bundle string) (*Engine, error) {
	cs := C.CString(bundle)
	defer C.free(unsafe.Pointer(cs))
	h := C.aria_model_init(cs)
	if h == nil {
		return nil, errors.New(C.GoString(C.aria_last_error()))
	}
	return &Engine{h: h}, nil
}

func (e *Engine) Close() {
	if e.h != nil {
		C.aria_model_destroy(e.h)
		e.h = nil
	}
}

func (e *Engine) Complete(messages any, options any, tools any) (map[string]any, error) {
	mb, _ := json.Marshal(messages)
	ob, _ := json.Marshal(options)
	tb, _ := json.Marshal(tools)
	ms, os_, ts := C.CString(string(mb)), C.CString(string(ob)), C.CString(string(tb))
	defer C.free(unsafe.Pointer(ms))
	defer C.free(unsafe.Pointer(os_))
	defer C.free(unsafe.Pointer(ts))
	buf := make([]byte, 256*1024)
	rc := C.aria_complete(e.h, ms, os_, ts, (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(len(buf)))
	if rc != 0 {
		return nil, errors.New(C.GoString(C.aria_last_error()))
	}
	var out map[string]any
	if err := json.Unmarshal(buf[:clen(buf)], &out); err != nil {
		return nil, err
	}
	return out, nil
}

func clen(b []byte) int {
	for i, c := range b {
		if c == 0 {
			return i
		}
	}
	return len(b)
}

func BundleEnv() string { return os.Getenv("ARIA_BUNDLE") }
