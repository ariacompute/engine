//go:build aria_ffi

package ariaengine

import (
	"os"
	"strings"
)

func fileExists(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.IsDir()
}

// OpenModel opens a model by reference. If ref contains a separator or already
// exists on disk it is loaded directly; otherwise it is treated as a model name
// and auto-downloaded from the regional public hub before loading.
func OpenModel(ref, token, site string) (*Engine, error) {
	return OpenModelOpts(ref, DownloadOptions{Token: token, Site: site})
}

// OpenModelOpts is OpenModel with explicit hf_token / modelscope_api_token.
func OpenModelOpts(ref string, opts DownloadOptions) (*Engine, error) {
	if _, err := EnsureFfiLib(opts.Site); err != nil {
		return nil, err
	}
	if strings.ContainsAny(ref, "/\\") || fileExists(ref) {
		return Open(ref)
	}
	bundle, err := DownloadModelOpts(ref, opts)
	if err != nil {
		return nil, err
	}
	return Open(bundle)
}
