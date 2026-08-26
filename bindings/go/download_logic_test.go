package aria

import (
	"archive/tar"
	"compress/gzip"
	"os"
	"path/filepath"
	"runtime"
	"strings"
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
		{"foo_q326_channel", "foo", "int326", false},
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

func TestPreferredPublicHub(t *testing.T) {
	if got := preferredPublicHub("https://ariacompute.com"); got != "huggingface" {
		t.Fatalf("got %s", got)
	}
	if got := preferredPublicHub("https://ariacompute.cn"); got != "modelscope" {
		t.Fatalf("got %s", got)
	}
	if got := preferredPublicHub(""); got != "huggingface" {
		t.Fatalf("got %s", got)
	}
}

func TestHubBearerIgnoresDashboardToken(t *testing.T) {
	if hubBearer("sk-bf-95076ed1-8c1a-4efa-b33c-f52c1d7f9f24") != "" {
		t.Fatal("dashboard sk- token must not be sent to hub")
	}
	if hubBearer("bfvk-test") != "" {
		t.Fatal("dashboard bfvk- token must not be sent to hub")
	}
	if hubBearer("hf_abc") != "hf_abc" {
		t.Fatal("hub token should pass through")
	}
}

func TestHubFileURLsFollowUploadLayout(t *testing.T) {
	hf := hubFileURLs("huggingface", "gemma-4-e2b-it_q4", "config.json")
	found := false
	for _, u := range hf {
		if strings.Contains(u, "/ariacompute/gemma-4-e2b-it_q4/resolve/main/v1.0/gemma-4-e2b-it_q4/config.json") {
			found = true
		}
		if strings.Contains(u, "/api/dashboard/") {
			t.Fatalf("dashboard URL leaked: %s", u)
		}
	}
	if !found {
		t.Fatalf("missing HF layout in %v", hf)
	}
	ms := hubFileURLs("modelscope", "gemma-4-e2b-it_q4", "weight.bin")
	found = false
	for _, u := range ms {
		if strings.Contains(u, "/v1.0/gemma-4-e2b-it_q4/weight.bin") {
			found = true
		}
		if strings.Contains(u, "/api/dashboard/") {
			t.Fatalf("dashboard URL leaked: %s", u)
		}
	}
	if !found {
		t.Fatalf("missing ModelScope layout in %v", ms)
	}
}

func writeValidBundle(t *testing.T, dir string) {
	t.Helper()
	os.MkdirAll(dir, 0o755)
	os.WriteFile(filepath.Join(dir, "weight.bin"), []byte("x"), 0o644)
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"format":"aria-quant-bundle"}`), 0o644)
}

func TestDownloadModelCachedSkip(t *testing.T) {
	home := t.TempDir()
	t.Setenv("ARIA_COMPUTE_HOME", home)
	cache := filepath.Join(home, "models", "foo_q4")
	writeValidBundle(t, cache)
	got, err := DownloadModel("foo_q4", "tok", "")
	if err != nil {
		t.Fatal(err)
	}
	if got != cache {
		t.Fatalf("want %q got %q", cache, got)
	}
}

func TestDownloadModelTokenOptionalWhenCached(t *testing.T) {
	home := t.TempDir()
	t.Setenv("ARIA_COMPUTE_HOME", home)
	cache := filepath.Join(home, "models", "foo_q4")
	writeValidBundle(t, cache)
	got, err := DownloadModel("foo_q4", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if got != cache {
		t.Fatalf("want %q got %q", cache, got)
	}
}

func TestResolveHubTokenNamedAndConfig(t *testing.T) {
	home := t.TempDir()
	t.Setenv("ARIA_COMPUTE_HOME", home)
	os.WriteFile(filepath.Join(home, "config.yml"), []byte("hf_token: hf_from_yml\nmodelscope_api_token: \"ms_from_yml\"\n"), 0o644)

	if got := resolveHubToken("huggingface", DownloadOptions{HFToken: "hf_named", Token: "hf_generic"}); got != "hf_named" {
		t.Fatalf("named hf_token: got %q", got)
	}
	if got := resolveHubToken("modelscope", DownloadOptions{ModelScopeAPIToken: "ms_named"}); got != "ms_named" {
		t.Fatalf("named modelscope_api_token: got %q", got)
	}
	if got := resolveHubToken("huggingface", DownloadOptions{}); got != "hf_from_yml" {
		t.Fatalf("config hf_token: got %q", got)
	}
	if got := resolveHubToken("modelscope", DownloadOptions{}); got != "ms_from_yml" {
		t.Fatalf("config modelscope_api_token: got %q", got)
	}
	if got := resolveHubToken("huggingface", DownloadOptions{Token: "sk-bf-not-hub"}); got != "hf_from_yml" {
		t.Fatalf("dashboard token should fall back to config, got %q", got)
	}
	if got := resolveHubToken("modelscope", DownloadOptions{HFToken: "hf_only"}); got != "ms_from_yml" {
		t.Fatalf("wrong-region named token should be ignored, got %q", got)
	}
}

func TestFfiAssetOS(t *testing.T) {
	got, err := ffiAssetOSFor("linux", "amd64")
	if err != nil || got != "linux_x86_64" {
		t.Fatalf("got %q %v", got, err)
	}
	got, err = ffiAssetOSFor("linux", "arm64")
	if err != nil || got != "linux_arm64" {
		t.Fatalf("got %q %v", got, err)
	}
	got, err = ffiAssetOSFor("darwin", "arm64")
	if err != nil || got != "macos" {
		t.Fatalf("got %q %v", got, err)
	}
	got, err = ffiAssetOSFor("windows", "amd64")
	if err != nil || got != "windows_x86_64" {
		t.Fatalf("got %q %v", got, err)
	}
	if _, err := ffiAssetOSFor("linux", "ppc64le"); err == nil {
		t.Fatal("expected error for ppc64le")
	}
}

func TestSelectLatestStable(t *testing.T) {
	releases := []ghRelease{
		{TagName: "v0.7.1"},
		{TagName: "v0.8.0-rc1", Prerelease: true},
		{TagName: "v0.7.2"},
		{TagName: "v0.9.0", Draft: true},
	}
	got, err := selectLatestStable(releases)
	if err != nil {
		t.Fatal(err)
	}
	if got != "0.7.2" {
		t.Fatalf("got %q", got)
	}
}

func TestUpgradeOrgFromSite(t *testing.T) {
	home := t.TempDir()
	t.Setenv("ARIA_COMPUTE_HOME", home)
	if got := upgradeOrg("https://ariacompute.com"); got != "https://github.com/ariacompute" {
		t.Fatalf("got %q", got)
	}
	if got := upgradeOrg("https://ariacompute.cn"); got != "https://gitee.com/ariacompute" {
		t.Fatalf("got %q", got)
	}
}

func TestExtractFfiAndCachedSkip(t *testing.T) {
	home := t.TempDir()
	t.Setenv("ARIA_COMPUTE_HOME", home)
	if runtime.GOOS != "linux" {
		t.Skip("linux .so fixture")
	}
	srcDir := t.TempDir()
	lib := filepath.Join(srcDir, "libaria_ffi.so")
	if err := os.WriteFile(lib, []byte("dummy-ffi"), 0o644); err != nil {
		t.Fatal(err)
	}
	archive := filepath.Join(home, "libaria_ffi_0.1.0_linux_x86_64.tar.gz")
	if err := packTarGz(archive, srcDir, "libaria_ffi.so"); err != nil {
		t.Fatal(err)
	}
	destDir := filepath.Join(home, "lib")
	got, err := extractFfiArchive(archive, destDir, "libaria_ffi.so")
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(got) != "libaria_ffi.so" {
		t.Fatalf("got %q", got)
	}
	raw, err := os.ReadFile(got)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != "dummy-ffi" {
		t.Fatalf("content %q", raw)
	}
	cached, err := EnsureFfiLib("")
	if err != nil {
		t.Fatal(err)
	}
	if cached != got {
		t.Fatalf("want cached %q got %q", got, cached)
	}
}

func packTarGz(archive, srcDir, name string) error {
	f, err := os.Create(archive)
	if err != nil {
		return err
	}
	defer f.Close()
	gw := gzip.NewWriter(f)
	defer gw.Close()
	tw := tar.NewWriter(gw)
	defer tw.Close()
	body, err := os.ReadFile(filepath.Join(srcDir, name))
	if err != nil {
		return err
	}
	hdr := &tar.Header{Name: name, Mode: 0755, Size: int64(len(body))}
	if err := tw.WriteHeader(hdr); err != nil {
		return err
	}
	_, err = tw.Write(body)
	return err
}
