//go:build aria_ffi

package aria

import (
	"os"
	"testing"
)

func TestCompleteOk(t *testing.T) {
	b := BundleEnv()
	if b == "" {
		t.Skip("ARIA_BUNDLE")
	}
	e, err := Open(b)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	out, err := e.Complete([]map[string]string{{"role": "user", "content": "hi"}}, map[string]any{"max_tokens": 2}, []any{})
	if err != nil {
		t.Fatal(err)
	}
	if out["success"] != true {
		t.Fatalf("%v", out)
	}
}

func TestInitMissing(t *testing.T) {
	if os.Getenv("ARIA_FFI_LIB") == "" && os.Getenv("ARIA_LIBDIR") == "" {
		t.Skip("no lib")
	}
	_, err := Open("/no/such")
	if err == nil {
		t.Fatal("expected error")
	}
}
