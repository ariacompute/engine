package aria_engine

/*
#cgo LDFLAGS: -L${SRCDIR}/../../target/release -L${SRCDIR}/../../target/debug -laria_ffi
#include <stdlib.h>
#include "../../ffi/include/aria.h"
*/
import "C"

import (
	"encoding/json"
	"errors"
	"fmt"
	"unsafe"
)

type Engine struct {
	handle *C.AriaModel
}

func Open(checkpoint, track string) (*Engine, error) {
	cPath := C.CString(checkpoint)
	cTrack := C.CString(track)
	defer C.free(unsafe.Pointer(cPath))
	defer C.free(unsafe.Pointer(cTrack))
	h := C.aria_model_init(cPath, cTrack)
	if h == nil {
		return nil, errors.New(C.GoString(C.aria_last_error()))
	}
	return &Engine{handle: h}, nil
}

func (e *Engine) SystemOne(request map[string]any) (map[string]any, error) {
	raw, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}
	cReq := C.CString(string(raw))
	defer C.free(unsafe.Pointer(cReq))
	buf := (*C.char)(C.malloc(1 << 20))
	defer C.free(unsafe.Pointer(buf))
	rc := C.aria_systemone(e.handle, cReq, buf, 1<<20)
	if rc != 0 {
		return nil, fmt.Errorf("aria_systemone: %s", C.GoString(C.aria_last_error()))
	}
	var out map[string]any
	if err := json.Unmarshal([]byte(C.GoString(buf)), &out); err != nil {
		return nil, err
	}
	return out, nil
}

func (e *Engine) Destroy() {
	if e.handle != nil {
		C.aria_model_destroy(e.handle)
		e.handle = nil
	}
}
