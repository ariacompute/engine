//go:build aria_ffi

package aria

import (
	"testing"
)

func TestOpenModelLocalNoToken(t *testing.T) {
	_, err := OpenModel("/no/such/path", "", "")
	if err == nil {
		t.Fatal("expected error for missing local bundle")
	}
}
