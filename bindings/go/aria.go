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
	"strings"
	"unsafe"
)

func Open(bundle string) (*Engine, error) {
	cs := C.CString(bundle)
	defer C.free(unsafe.Pointer(cs))
	h := C.aria_model_init(cs)
	if h == nil {
		return nil, errors.New(C.GoString(C.aria_last_error()))
	}
	e := NewEngine()
	e.h = unsafe.Pointer(h)
	return e, nil
}

func (e *Engine) native() *C.AriaModel {
	return (*C.AriaModel)(e.h)
}

func (e *Engine) Close() {
	if e.h != nil {
		C.aria_model_destroy(e.native())
		e.h = nil
	}
}

// Open downloads (if needed) and loads a model using instance setup.
func (e *Engine) Open(ref string) error {
	if _, err := EnsureFfiLib(e.cfg.SiteURL); err != nil {
		return err
	}
	bundle := ref
	if !strings.ContainsAny(ref, "/\\") && !fileExists(ref) {
		var err error
		bundle, err = DownloadModelOpts(ref, DownloadOptions{
			Token:              e.genericToken,
			HFToken:            e.cfg.HFToken,
			ModelScopeAPIToken: e.cfg.ModelScopeAPIToken,
			Site:               e.cfg.SiteURL,
		})
		if err != nil {
			return err
		}
	}
	cs := C.CString(bundle)
	defer C.free(unsafe.Pointer(cs))
	h := C.aria_model_init(cs)
	if h == nil {
		return errors.New(C.GoString(C.aria_last_error()))
	}
	if e.h != nil {
		C.aria_model_destroy(e.native())
	}
	e.h = unsafe.Pointer(h)
	return nil
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
	rc := C.aria_complete(e.native(), ms, os_, ts, (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(len(buf)))
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
