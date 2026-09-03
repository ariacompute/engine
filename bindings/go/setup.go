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

// SetupConfig holds Config / Run fields on an Engine instance (memory only).
type SetupConfig struct {
	// Local (router registration)
	Router       string
	RouterAPIKey string
	// OAuth (Aria Compute)
	ServeSite   string
	ServeAPIKey string
	// Hub / compute
	SiteURL            string
	UpgradeURL         string
	Compute            string
	HFToken            string
	ModelScopeAPIToken string
}

// SetupUpdates is a partial merge. Nil fields are omitted.
type SetupUpdates struct {
	// Local
	Router       *string
	RouterAPIKey *string
	// OAuth
	ServeSite   *string
	ServeAPIKey *string
	// Hub / compute
	SiteURL            *string
	UpgradeURL         *string
	Compute            *string
	HFToken            *string
	ModelScopeAPIToken *string
}

func DefaultSetupConfig() SetupConfig {
	return SetupConfig{Compute: "auto"}
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

// FillSetupUrls fills missing site/upgrade URLs from a provided TLD.
func FillSetupUrls(cfg SetupConfig) SetupConfig {
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

func validateRouterAPIKey(key string) error {
	t := strings.TrimSpace(key)
	if t == "" {
		return nil
	}
	if strings.HasPrefix(t, "bfvk-") {
		return fmt.Errorf("OAuth key detected (bfvk-); use ServeAPIKey / [2/2] OAuth (Aria Compute)")
	}
	return nil
}

func validateServeAPIKey(key string) error {
	t := strings.TrimSpace(key)
	if t == "" {
		return nil
	}
	if strings.HasPrefix(t, "sk-aria_") {
		return fmt.Errorf("Local router key detected (sk-aria_); use RouterAPIKey / [1/2] Local")
	}
	if !strings.HasPrefix(t, "bfvk-") {
		return fmt.Errorf("serve_api_key must start with bfvk-")
	}
	return nil
}

// ApplySetup merges updates into existing. Validates; does not mutate existing.
func ApplySetup(existing SetupConfig, updates SetupUpdates) (SetupConfig, error) {
	out := existing
	if updates.Router != nil {
		out.Router = *updates.Router
	}
	if updates.RouterAPIKey != nil {
		if err := validateRouterAPIKey(*updates.RouterAPIKey); err != nil {
			return SetupConfig{}, err
		}
		out.RouterAPIKey = *updates.RouterAPIKey
	}
	if updates.ServeSite != nil {
		out.ServeSite = *updates.ServeSite
	}
	if updates.ServeAPIKey != nil {
		if err := validateServeAPIKey(*updates.ServeAPIKey); err != nil {
			return SetupConfig{}, err
		}
		out.ServeAPIKey = *updates.ServeAPIKey
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
		return SetupConfig{}, fmt.Errorf("invalid compute: %s", out.Compute)
	}
	if err := validateRouterAPIKey(out.RouterAPIKey); err != nil {
		return SetupConfig{}, err
	}
	if err := validateServeAPIKey(out.ServeAPIKey); err != nil {
		return SetupConfig{}, err
	}
	return FillSetupUrls(out), nil
}

// Engine holds instance setup in memory. The native handle is filled when built with aria_ffi.
type Engine struct {
	cfg          SetupConfig
	genericToken string
	h            unsafe.Pointer
}

// NewEngine constructs an empty Engine. Call Setup then Open to download/load.
func NewEngine() *Engine {
	return &Engine{cfg: DefaultSetupConfig()}
}

// Setup sets Config / Run fields on this instance only. Does not write engine.yml.
func (e *Engine) Setup(u SetupUpdates) error {
	next, err := ApplySetup(e.cfg, u)
	if err != nil {
		return err
	}
	e.cfg = next
	return nil
}

func (e *Engine) SetupStatus() SetupConfig {
	return e.cfg
}

// SetupClear resets instance defaults. Does not delete ~/.ariacompute/engine.yml.
func (e *Engine) SetupClear() {
	e.cfg = DefaultSetupConfig()
}
