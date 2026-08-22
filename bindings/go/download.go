package aria

import (
	"archive/zip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
)

const defaultSite = "https://ariacompute.com"

func ariaHome() (string, error) {
	if v := os.Getenv("ARIA_COMPUTE_HOME"); v != "" {
		return v, nil
	}
	h, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(h, ".ariacompute"), nil
}

func cacheDir(model string) (string, error) {
	home, err := ariaHome()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, "models", model), nil
}

// parseBundleName returns (slug, quant) from a model name like "gemma-4-e2b-it_q4".
func parseBundleName(model string) (slug, quant string, err error) {
	if model == "" || strings.ContainsAny(model, "/\\") {
		return "", "", fmt.Errorf("invalid model name: %s", model)
	}
	idx := strings.LastIndex(model, "_q")
	if idx != -1 {
		slug = model[:idx]
		suffix := model[idx+2:]
		switch suffix {
		case "4":
			quant = "int4"
		case "8":
			quant = "int8"
		case "326", "3.26":
			quant = "int326"
		default:
			return "", "", fmt.Errorf("unknown quant suffix _q%s", suffix)
		}
		if slug == "" {
			return "", "", fmt.Errorf("invalid model name: %s", model)
		}
		return slug, quant, nil
	}
	return model, "int4", nil
}

func isValidBundle(dir string) bool {
	if fi, err := os.Stat(filepath.Join(dir, "weight.bin")); err != nil || fi.IsDir() {
		return false
	}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return false
	}
	var meta struct {
		Format string `json:"format"`
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		return false
	}
	return meta.Format == "aria-quant-bundle"
}

// DownloadModel downloads `model` from the Dashboard private source into
// ~/.ariacompute/models/{model} and returns that path. If a valid bundle is
// already cached, the download is skipped.
func DownloadModel(model, token, site string) (string, error) {
	if token == "" {
		return "", errors.New("dashboard token is required to download a model")
	}
	if site == "" {
		site = defaultSite
	}
	slug, quant, err := parseBundleName(model)
	if err != nil {
		return "", err
	}
	cache, err := cacheDir(model)
	if err != nil {
		return "", err
	}
	if fi, _ := os.Stat(cache); fi != nil && isValidBundle(cache) {
		return cache, nil
	}

	metaURL := fmt.Sprintf("%s/api/dashboard/models/%s/download?quant=%s&sdk=v1.0&format=json",
		strings.TrimRight(site, "/"), url.PathEscape(slug), url.QueryEscape(quant))
	req, _ := http.NewRequest(http.MethodGet, metaURL, nil)
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("dashboard request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("dashboard request failed: %s", resp.Status)
	}
	var meta struct {
		URL string `json:"url"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&meta); err != nil {
		return "", fmt.Errorf("dashboard meta decode failed: %w", err)
	}
	if meta.URL == "" {
		return "", errors.New("dashboard meta returned empty url")
	}

	req, _ = http.NewRequest(http.MethodGet, meta.URL, nil)
	req.Header.Set("Authorization", "Bearer "+token)
	zipResp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("download stream failed: %w", err)
	}
	defer zipResp.Body.Close()
	if zipResp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download stream failed: %s", zipResp.Status)
	}
	data, err := io.ReadAll(zipResp.Body)
	if err != nil {
		return "", fmt.Errorf("download stream failed: %w", err)
	}

	home, _ := ariaHome()
	staging := filepath.Join(home, "models", "."+model+".partial")
	os.RemoveAll(staging)
	if err := extractZip(data, staging); err != nil {
		os.RemoveAll(staging)
		return "", err
	}
	if !isValidBundle(staging) {
		os.RemoveAll(staging)
		return "", errors.New("downloaded archive did not contain a valid aria-quant-bundle")
	}
	os.RemoveAll(cache)
	if err := os.Rename(staging, cache); err != nil {
		return "", err
	}
	return cache, nil
}

func extractZip(data []byte, dest string) error {
	zr, err := zip.NewReader(strings.NewReader(string(data)), int64(len(data)))
	if err != nil {
		return fmt.Errorf("invalid zip archive: %w", err)
	}
	if err := os.MkdirAll(dest, 0o755); err != nil {
		return err
	}
	for _, f := range zr.File {
		out := filepath.Join(dest, f.Name)
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(out, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
			return err
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		outF, err := os.Create(out)
		if err != nil {
			rc.Close()
			return err
		}
		if _, err := io.Copy(outF, rc); err != nil {
			rc.Close()
			outF.Close()
			return err
		}
		rc.Close()
		outF.Close()
	}
	// flatten a single top-level subdir
	entries, _ := os.ReadDir(dest)
	real := make([]os.DirEntry, 0, len(entries))
	for _, e := range entries {
		if !strings.HasPrefix(e.Name(), ".") {
			real = append(real, e)
		}
	}
	if len(real) == 1 && real[0].IsDir() {
		inner := filepath.Join(dest, real[0].Name())
		if fi, _ := os.Stat(filepath.Join(inner, "config.json")); fi != nil && !fi.IsDir() {
			names, _ := os.ReadDir(inner)
			for _, n := range names {
				os.Rename(filepath.Join(inner, n.Name()), filepath.Join(dest, n.Name()))
			}
			os.Remove(inner)
		}
	}
	return nil
}

// OpenModel opens a model by reference. If ref contains a separator or already
// exists on disk it is loaded directly; otherwise it is treated as a model name
// and auto-downloaded (requires a dashboard token) before loading.
func OpenModel(ref, token, site string) (*Engine, error) {
	if strings.ContainsAny(ref, "/\\") || fileExists(ref) {
		return Open(ref)
	}
	if token == "" {
		return nil, fmt.Errorf("model name %q requires a dashboard token to download", ref)
	}
	bundle, err := DownloadModel(ref, token, site)
	if err != nil {
		return nil, err
	}
	return Open(bundle)
}

func fileExists(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.IsDir()
}
