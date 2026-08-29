package aria

import (
	"fmt"
	"strings"
	"unsafe"
)

const (
	IntlSite    = "https://ariacompute.com"
	IntlUpgrade = "https://github.com/ariacompute"
	CNSite      = "https://ariacompute.cn"
	CNUpgrade   = "https://gitee.com/ariacompute"
)

// AuthConfig holds Config / Run fields on an Engine instance (memory only).
type AuthConfig struct {
	Router             string
	SiteURL            string
	UpgradeURL         string
	Compute            string
	HFToken            string
	ModelScopeAPIToken string
}

// AuthUpdates is a partial merge. Nil fields are omitted.
type AuthUpdates struct {
	Router             *string
	SiteURL            *string
	UpgradeURL         *string
	Compute            *string
	HFToken            *string
	ModelScopeAPIToken *string
}

func DefaultAuthConfig() AuthConfig {
	return AuthConfig{Compute: "auto"}
}

func gatewayRegion(url string) string {
	lower := strings.ToLower(url)
	if strings.Contains(lower, "ariacompute.cn") || strings.Contains(lower, "gitee.com/ariacompute") {
		return "cn"
	}
	if strings.Contains(lower, "ariacompute.com") || strings.Contains(lower, "github.com/ariacompute") {
		return "intl"
	}
	return ""
}

func pairURLs(region string) (site, upgrade string) {
	if region == "cn" {
		return CNSite, CNUpgrade
	}
	return IntlSite, IntlUpgrade
}

// FillAuthUrls fills missing site/upgrade URLs from a provided TLD.
func FillAuthUrls(cfg AuthConfig) AuthConfig {
	region := gatewayRegion(cfg.SiteURL)
	if region == "" {
		region = gatewayRegion(cfg.UpgradeURL)
	}
	if region == "" {
		return cfg
	}
	site, upgrade := pairURLs(region)
	if cfg.SiteURL == "" {
		cfg.SiteURL = site
	}
	if cfg.UpgradeURL == "" {
		cfg.UpgradeURL = upgrade
	}
	return cfg
}

// ApplyAuth merges updates into existing. Validates; does not mutate existing.
func ApplyAuth(existing AuthConfig, updates AuthUpdates) (AuthConfig, error) {
	out := existing
	if updates.Router != nil {
		out.Router = *updates.Router
	}
	if updates.SiteURL != nil {
		out.SiteURL = *updates.SiteURL
	}
	if updates.UpgradeURL != nil {
		out.UpgradeURL = *updates.UpgradeURL
	}
	if updates.Compute != nil {
		out.Compute = *updates.Compute
	}
	if updates.HFToken != nil {
		out.HFToken = *updates.HFToken
	}
	if updates.ModelScopeAPIToken != nil {
		out.ModelScopeAPIToken = *updates.ModelScopeAPIToken
	}
	switch out.Compute {
	case "auto", "cpu", "cuda":
	default:
		return AuthConfig{}, fmt.Errorf("invalid compute: %s", out.Compute)
	}
	return FillAuthUrls(out), nil
}

// Engine holds instance auth in memory. The native handle is filled when built with aria_ffi.
type Engine struct {
	cfg          AuthConfig
	genericToken string
	h            unsafe.Pointer
}

// NewEngine constructs an empty Engine. Call Auth then Open to download/load.
func NewEngine() *Engine {
	return &Engine{cfg: DefaultAuthConfig()}
}

// Auth sets Config / Run fields on this instance only. Does not write config.yml.
func (e *Engine) Auth(u AuthUpdates) error {
	next, err := ApplyAuth(e.cfg, u)
	if err != nil {
		return err
	}
	e.cfg = next
	return nil
}

func (e *Engine) AuthStatus() AuthConfig {
	return e.cfg
}

// AuthClear resets instance defaults. Does not delete ~/.ariacompute/config.yml.
func (e *Engine) AuthClear() {
	e.cfg = DefaultAuthConfig()
}
