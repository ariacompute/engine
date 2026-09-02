package ariaengine

import (
	"archive/tar"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
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

func unquoteYAML(v string) string {
	v = strings.TrimSpace(v)
	if n := len(v); n >= 2 {
		if (v[0] == '"' && v[n-1] == '"') || (v[0] == '\'' && v[n-1] == '\'') {
			return v[1 : n-1]
		}
	}
	return v
}

func configYMLScalar(key string) string {
	home, err := ariaHome()
	if err != nil {
		return ""
	}
	for _, name := range []string{"engine.yml", "config.yml"} {
		raw, err := os.ReadFile(filepath.Join(home, name))
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(raw), "\n") {
			if line != "" && (line[0] == ' ' || line[0] == '\t') {
				continue
			}
			s := strings.TrimSpace(line)
			if s == "" || strings.HasPrefix(s, "#") {
				continue
			}
			k, v, ok := strings.Cut(s, ":")
			if !ok || strings.TrimSpace(k) != key {
				continue
			}
			return unquoteYAML(v)
		}
	}
	return ""
}

func hubTokenField(source string) string {
	if source == "modelscope" {
		return "modelscope_api_token"
	}
	return "hf_token"
}

// DownloadOptions controls hub download auth. Field names match ariaengine setup
// (hf_token / modelscope_api_token) plus a legacy Token and Site.
type DownloadOptions struct {
	Token              string
	HFToken            string
	ModelScopeAPIToken string
	Site               string
}

func resolveHubToken(source string, opts DownloadOptions) string {
	named := opts.HFToken
	if source == "modelscope" {
		named = opts.ModelScopeAPIToken
	}
	for _, cand := range []string{named, opts.Token, configYMLScalar(hubTokenField(source))} {
		if b := hubBearer(cand); b != "" {
			return b
		}
	}
	return ""
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

type hubSetupError struct {
	code   int
	source string
}

func (e *hubSetupError) Error() string {
	field := "hf_token"
	if e.source == "modelscope" {
		field = "modelscope_api_token"
	}
	return fmt.Sprintf("auth failed HTTP %d; set %s via ariaengine setup (do not pass a Dashboard sk-/bfvk- key as the hub token)", e.code, field)
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
		return &hubSetupError{code: resp.StatusCode}
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
		if ae, ok := err.(*hubSetupError); ok {
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
// Matches ariaengine download: .com → Hugging Face, .cn → ModelScope.
// Hub auth uses hf_token / modelscope_api_token (DownloadModelOpts, else
// ~/.ariacompute/engine.yml from ariaengine setup). Dashboard is not used.
func DownloadModel(model, token, site string) (string, error) {
	return DownloadModelOpts(model, DownloadOptions{Token: token, Site: site})
}

// DownloadModelOpts is DownloadModel with explicit hub tokens (same keys as
// ariaengine setup).
func DownloadModelOpts(model string, opts DownloadOptions) (string, error) {
	if _, _, err := parseBundleName(model); err != nil {
		return "", err
	}
	site := opts.Site
	if site == "" {
		site = defaultSite
	}
	source := preferredPublicHub(site)
	hubToken := resolveHubToken(source, opts)
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

const sdkUserAgent = "ariaengine-sdk/0.1.0"

func ffiLibNameFor(goos string) string {
	switch goos {
	case "windows":
		return "ariaengine_ffi.dll"
	case "darwin":
		return "libariaengine_ffi.dylib"
	default:
		return "libariaengine_ffi.so"
	}
}

func ffiLibName() string { return ffiLibNameFor(runtime.GOOS) }

func libDir() (string, error) {
	home, err := ariaHome()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, "lib"), nil
}

func cachedFfiPath() (string, error) {
	dir, err := libDir()
	if err != nil {
		return "", err
	}
	p := filepath.Join(dir, ffiLibName())
	if fi, err := os.Stat(p); err == nil && !fi.IsDir() {
		return p, nil
	}
	return "", nil
}

func ffiAssetOSFor(goos, goarch string) (string, error) {
	switch goos {
	case "linux":
		switch goarch {
		case "amd64":
			return "linux_x86_64", nil
		case "arm64":
			return "linux_arm64", nil
		}
	case "darwin":
		return "macos", nil
	case "windows":
		if goarch == "amd64" {
			return "windows_x86_64", nil
		}
	}
	return "", fmt.Errorf("unsupported platform %s/%s for libariaengine_ffi", goos, goarch)
}

func stripV(tag string) string {
	t := strings.TrimSpace(tag)
	if strings.HasPrefix(t, "v") || strings.HasPrefix(t, "V") {
		return t[1:]
	}
	return t
}

func parseSemver(tag string) (major, minor, patch int, ok bool) {
	core := stripV(tag)
	if i := strings.IndexAny(core, "-+"); i >= 0 {
		core = core[:i]
	}
	parts := strings.Split(core, ".")
	if len(parts) == 0 || parts[0] == "" {
		return 0, 0, 0, false
	}
	var err error
	major, err = strconv.Atoi(parts[0])
	if err != nil {
		return 0, 0, 0, false
	}
	if len(parts) > 1 {
		minor, _ = strconv.Atoi(parts[1])
	}
	if len(parts) > 2 {
		patch, _ = strconv.Atoi(parts[2])
	}
	return major, minor, patch, true
}

type ghRelease struct {
	TagName    string `json:"tag_name"`
	Tag        string `json:"tag"`
	Draft      bool   `json:"draft"`
	Prerelease bool   `json:"prerelease"`
	Assets     []struct {
		Name               string `json:"name"`
		BrowserDownloadURL string `json:"browser_download_url"`
		DirectAssetURL     string `json:"direct_asset_url"`
	} `json:"assets"`
}

func (r ghRelease) tag() string {
	if r.TagName != "" {
		return r.TagName
	}
	return r.Tag
}

func selectLatestStable(releases []ghRelease) (string, error) {
	bestTag := ""
	bestM, bestN, bestP := -1, -1, -1
	for _, rel := range releases {
		if rel.Draft || rel.Prerelease {
			continue
		}
		tag := rel.tag()
		m, n, p, ok := parseSemver(tag)
		if !ok {
			continue
		}
		if m > bestM || (m == bestM && n > bestN) || (m == bestM && n == bestN && p > bestP) {
			bestM, bestN, bestP = m, n, p
			bestTag = tag
		}
	}
	if bestTag == "" {
		return "", fmt.Errorf("no stable release found for libariaengine_ffi")
	}
	return stripV(bestTag), nil
}

func upgradeOrg(site string) string {
	if cfg := configYMLScalar("upgrade_url"); cfg != "" {
		return strings.TrimRight(cfg, "/")
	}
	hint := strings.ToLower(site)
	if hint == "" {
		hint = strings.ToLower(configYMLScalar("site_url"))
	}
	if hint == "" {
		hint = defaultSite
	}
	if strings.Contains(hint, "ariacompute.cn") || strings.Contains(hint, "gitee.com") {
		return "https://gitee.com/ariacompute"
	}
	return "https://github.com/ariacompute"
}

func releasesAPIURL(org string) string {
	org = strings.TrimRight(org, "/")
	owner := org[strings.LastIndex(org, "/")+1:]
	if owner == "" {
		owner = "ariacompute"
	}
	if strings.Contains(strings.ToLower(org), "gitee.com") {
		return fmt.Sprintf("https://gitee.com/api/v5/repos/%s/engine/releases?per_page=30", owner)
	}
	return fmt.Sprintf("https://api.github.com/repos/%s/engine/releases?per_page=30", owner)
}

func httpGetBytes(url, dest string) ([]byte, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", sdkUserAgent)
	resp, err := hubClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %s", resp.Status)
	}
	if dest != "" {
		if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
			return nil, err
		}
		out, err := os.Create(dest)
		if err != nil {
			return nil, err
		}
		defer out.Close()
		_, err = io.Copy(out, resp.Body)
		return nil, err
	}
	return io.ReadAll(resp.Body)
}

func extractFfiArchive(archive, destDir, want string) (string, error) {
	f, err := os.Open(archive)
	if err != nil {
		return "", err
	}
	defer f.Close()
	gr, err := gzip.NewReader(f)
	if err != nil {
		return "", err
	}
	defer gr.Close()
	tr := tar.NewReader(gr)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		if hdr.Typeflag != tar.TypeReg {
			continue
		}
		if filepath.Base(hdr.Name) != want {
			continue
		}
		if err := os.MkdirAll(destDir, 0o755); err != nil {
			return "", err
		}
		dest := filepath.Join(destDir, want)
		out, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
		if err != nil {
			return "", err
		}
		if _, err := io.Copy(out, tr); err != nil {
			out.Close()
			return "", err
		}
		out.Close()
		return dest, nil
	}
	return "", fmt.Errorf("%s not found in %s", want, archive)
}

// EnsureFfiLib returns a path to libariaengine_ffi, downloading the latest stable
// Release into ~/.ariacompute/lib/ if it is not already present.
func EnsureFfiLib(site string) (string, error) {
	if env := os.Getenv("ARIAENGINE_FFI_LIB"); env != "" {
		if fi, err := os.Stat(env); err == nil && !fi.IsDir() {
			return env, nil
		}
	}
	if cached, err := cachedFfiPath(); err != nil {
		return "", err
	} else if cached != "" {
		return cached, nil
	}

	org := upgradeOrg(site)
	raw, err := httpGetBytes(releasesAPIURL(org), "")
	if err != nil {
		return "", fmt.Errorf("releases API: %w", err)
	}
	var releases []ghRelease
	if err := json.Unmarshal(raw, &releases); err != nil {
		return "", fmt.Errorf("invalid releases JSON from %s: %w", org, err)
	}
	ver, err := selectLatestStable(releases)
	if err != nil {
		return "", err
	}
	assetOS, err := ffiAssetOSFor(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		return "", err
	}
	assetName := fmt.Sprintf("libariaengine_ffi_%s_%s.tar.gz", ver, assetOS)
	var url string
	for _, rel := range releases {
		if stripV(rel.tag()) != ver {
			continue
		}
		for _, a := range rel.Assets {
			if a.Name == assetName {
				url = a.BrowserDownloadURL
				if url == "" {
					url = a.DirectAssetURL
				}
				break
			}
		}
		if url != "" {
			break
		}
	}
	if url == "" {
		return "", fmt.Errorf("release asset not found: %s", assetName)
	}

	home, err := ariaHome()
	if err != nil {
		return "", err
	}
	staging := filepath.Join(home, "tmp", "ffi-"+ver)
	os.RemoveAll(staging)
	if err := os.MkdirAll(staging, 0o755); err != nil {
		return "", err
	}
	defer os.RemoveAll(staging)
	archive := filepath.Join(staging, assetName)
	if _, err := httpGetBytes(url, archive); err != nil {
		return "", err
	}
	dir, err := libDir()
	if err != nil {
		return "", err
	}
	return extractFfiArchive(archive, dir, ffiLibName())
}
