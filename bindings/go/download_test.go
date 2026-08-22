//go:build aria_ffi

package aria

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestParseBundleName(t *testing.T) {
	cases := []struct {
		in    string
		slug  string
		quant string
		fail  bool
	}{
		{"gemma-4-e2b-it_q4", "gemma-4-e2b-it", "int4", false},
		{"foo_q8", "foo", "int8", false},
		{"foo_q326", "foo", "int326", false},
		{"foo_q3.26", "foo", "int326", false},
		{"foo", "foo", "int4", false},
		{"foo/bar", "", "", true},
		{"foo_q9", "", "", true},
		{"", "", "", true},
	}
	for _, c := range cases {
		slug, quant, err := parseBundleName(c.in)
		if c.fail {
			if err == nil {
				t.Errorf("expected error for %q", c.in)
			}
			continue
		}
		if err != nil {
			t.Errorf("unexpected error for %q: %v", c.in, err)
			continue
		}
		if slug != c.slug || quant != c.quant {
			t.Errorf("parseBundleName(%q) = (%q,%q), want (%q,%q)", c.in, slug, quant, c.slug, c.quant)
		}
	}
}

func TestIsValidBundle(t *testing.T) {
	dir := t.TempDir()
	if isValidBundle(dir) {
		t.Fatal("empty dir should be invalid")
	}
	os.WriteFile(filepath.Join(dir, "weight.bin"), []byte("x"), 0o644)
	if isValidBundle(dir) {
		t.Fatal("missing config should be invalid")
	}
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"format":"other"}`), 0o644)
	if isValidBundle(dir) {
		t.Fatal("wrong format should be invalid")
	}
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"format":"aria-quant-bundle"}`), 0o644)
	if !isValidBundle(dir) {
		t.Fatal("valid bundle reported invalid")
	}
}

func TestDownloadModelMissingToken(t *testing.T) {
	if _, err := DownloadModel("foo_q4", "", ""); err == nil {
		t.Fatal("expected token error")
	}
}

func TestDownloadModelCachedSkip(t *testing.T) {
	home := t.TempDir()
	t.Setenv("ARIA_COMPUTE_HOME", home)
	cache := filepath.Join(home, "models", "foo_q4")
	os.MkdirAll(cache, 0o755)
	os.WriteFile(filepath.Join(cache, "weight.bin"), []byte("x"), 0o644)
	os.WriteFile(filepath.Join(cache, "config.json"), []byte(`{"format":"aria-quant-bundle"}`), 0o644)
	got, err := DownloadModel("foo_q4", "tok", "")
	if err != nil {
		t.Fatal(err)
	}
	if got != cache {
		t.Fatalf("want %q got %q", cache, got)
	}
}

func TestDownloadModelNoNetwork(t *testing.T) {
	home := t.TempDir()
	t.Setenv("ARIA_COMPUTE_HOME", home)
	_, err := DownloadModel("foo_q4", "tok", "http://127.0.0.1:9")
	if err == nil {
		t.Fatal("expected network error")
	}
}

func TestOpenModelLocalNoToken(t *testing.T) {
	_, err := OpenModel("/no/such/path", "", "")
	if err == nil {
		t.Fatal("expected error for missing local bundle")
	}
}

func TestOpenModelNameRequiresToken(t *testing.T) {
	_, err := OpenModel("gemma-4-e2b-it_q4", "", "")
	if err == nil {
		t.Fatal("expected token-required error")
	}
}

var _ = json.Marshal
