package aria

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const defaultSite = "https://ariacompute.com"
const defaultSDK = "v1.0"

var hubRequired = []string{"config.json", "weight.bin"}
var hubOptional = []string{
	"tokenizer.json",
	"tokenizer.model",
	"tokenizer_config.json",
	"special_tokens_map.json",
	"vocab.json",
	"merges.txt",
}

var hubClient = &http.Client{Timeout: 10 * time.Minute}

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
		if strings.HasSuffix(suffix, "_channel") || strings.HasSuffix(suffix, "_group") {
			if i := strings.LastIndex(suffix, "_"); i >= 0 {
				suffix = suffix[:i]
			}
		}
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

func preferredPublicHub(site string) string {
	if strings.Contains(strings.ToLower(site), "ariacompute.cn") {
		return "modelscope"
	}
	return "huggingface"
}

func isDashboardToken(token string) bool {
	t := strings.ToLower(strings.TrimSpace(token))
	return strings.HasPrefix(t, "sk-") || strings.HasPrefix(t, "bfvk-")
}

func hubBearer(token string) string {
	t := strings.TrimSpace(token)
	if t == "" || isDashboardToken(t) {
		return ""
	}
	return t
}

func hubPathNames(model string) []string {
	names := []string{model}
	lower := strings.ToLower(model)
	core := model
	for _, suf := range []string{"_channel", "_group"} {
		if strings.HasSuffix(lower, suf) {
			core = model[:len(model)-len(suf)]
			lower = strings.ToLower(core)
			break
		}
	}
	stems := []string{core}
	if strings.HasSuffix(lower, "_q326") {
		stems = append(stems, core[:len(core)-5]+"q3.26")
	} else if strings.HasSuffix(lower, "_q3.26") {
		stems = append(stems, core[:len(core)-6]+"q326")
	}
	for _, stem := range stems {
		for _, share := range []string{"", "_channel", "_group"} {
			cand := stem + share
			found := false
			for _, n := range names {
				if n == cand {
					found = true
					break
				}
			}
			if !found {
				names = append(names, cand)
			}
		}
	}
	return names
}

func hubFileURLs(source, model, file string) []string {
	var urls []string
	for _, name := range hubPathNames(model) {
		if source == "modelscope" {
			for _, repo := range []string{"AriaCompute/" + name, "AriaCompute/model"} {
				urls = append(urls,
					fmt.Sprintf("https://www.modelscope.cn/models/%s/resolve/master/%s/%s/%s", repo, defaultSDK, name, file),
					fmt.Sprintf("https://modelscope.cn/models/%s/resolve/master/%s/%s/%s", repo, defaultSDK, name, file),
				)
			}
		} else {
			for _, repo := range []string{"ariacompute/" + name, "ariacompute/model"} {
				urls = append(urls,
					fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s/%s/%s", repo, defaultSDK, name, file),
				)
			}
		}
	}
	return urls
}

type hubAuthError struct {
	code   int
	source string
}

func (e *hubAuthError) Error() string {
	field := "hf_token"
	if e.source == "modelscope" {
		field = "modelscope_api_token"
	}
	return fmt.Sprintf("auth failed HTTP %d; set %s via aria-engine auth (do not pass a Dashboard sk-/bfvk- key as the hub token)", e.code, field)
}

func fetchURLToFile(url, dest, token string) error {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := hubClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return &hubAuthError{code: resp.StatusCode}
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %s", resp.Status)
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, resp.Body)
	return err
}

func fetchHubFile(source, model, file, dest, token string, required bool) error {
	var last error
	for _, u := range hubFileURLs(source, model, file) {
		err := fetchURLToFile(u, dest, token)
		if err == nil {
			return nil
		}
		if ae, ok := err.(*hubAuthError); ok {
			ae.source = source
			return ae
		}
		last = err
	}
	if required {
		return fmt.Errorf("%s: missing %s: %v", source, file, last)
	}
	return nil
}

// DownloadModel downloads `model` from the regional public hub into
// ~/.ariacompute/models/{model} and returns that path.
//
// Matches aria-engine download: .com → Hugging Face, .cn → ModelScope.
// Dashboard is not used. A Dashboard API key (sk- / bfvk-) is ignored for
// hub auth. If a valid bundle already exists at the cache path, the download is skipped.
func DownloadModel(model, token, site string) (string, error) {
	if _, _, err := parseBundleName(model); err != nil {
		return "", err
	}
	if site == "" {
		site = defaultSite
	}
	source := preferredPublicHub(site)
	hubToken := hubBearer(token)
	cache, err := cacheDir(model)
	if err != nil {
		return "", err
	}
	if fi, _ := os.Stat(cache); fi != nil && isValidBundle(cache) {
		return cache, nil
	}

	home, err := ariaHome()
	if err != nil {
		return "", err
	}
	staging := filepath.Join(home, "models", "."+model+".partial")
	os.RemoveAll(staging)
	if err := os.MkdirAll(staging, 0o755); err != nil {
		return "", err
	}
	cleanup := true
	defer func() {
		if cleanup {
			os.RemoveAll(staging)
		}
	}()

	for _, file := range hubRequired {
		if err := fetchHubFile(source, model, file, filepath.Join(staging, file), hubToken, true); err != nil {
			return "", err
		}
	}
	for _, extra := range hubOptional {
		_ = fetchHubFile(source, model, extra, filepath.Join(staging, extra), hubToken, false)
	}
	if !isValidBundle(staging) {
		return "", fmt.Errorf("%s fetch completed but bundle invalid (need weight.bin + aria-quant-bundle config.json)", source)
	}
	os.RemoveAll(cache)
	if err := os.Rename(staging, cache); err != nil {
		return "", err
	}
	cleanup = false
	return cache, nil
}
