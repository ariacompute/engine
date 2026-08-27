package aria

import (
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
	"unsafe"
)

const (
	IntlCloud   = "https://gateway.ariacompute.com"
	IntlSite    = "https://ariacompute.com"
	IntlUpgrade = "https://github.com/ariacompute"
	CNCloud     = "https://gateway.ariacompute.cn"
	CNSite      = "https://ariacompute.cn"
	CNUpgrade   = "https://gitee.com/ariacompute"
)

// AuthConfig holds the 12 Config / Run fields on an Engine instance (memory only).
type AuthConfig struct {
	CloudAPIKey             string
	CloudURL                string
	SiteURL                 string
	UpgradeURL              string
	HybridMode              string
	HybridExecution         string
	HybridSemantic          bool
	HybridSemanticTimeoutMS int
	HybridSemanticCacheSize int
	Compute                 string
	HFToken                 string
	ModelScopeAPIToken      string
}

// AuthUpdates is a partial merge. Nil fields are omitted.
type AuthUpdates struct {
	CloudAPIKey             *string
	CloudURL                *string
	SiteURL                 *string
	UpgradeURL              *string
	HybridMode              *string
	HybridExecution         *string
	HybridSemantic          *bool
	HybridSemanticTimeoutMS *int
	HybridSemanticCacheSize *int
	Compute                 *string
	HFToken                 *string
	ModelScopeAPIToken      *string
}

func DefaultAuthConfig() AuthConfig {
	return AuthConfig{
		HybridMode:              "balance",
		HybridExecution:         "hybrid",
		HybridSemantic:          true,
		HybridSemanticTimeoutMS: 800,
		HybridSemanticCacheSize: 512,
		Compute:                 "auto",
	}
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

func pairURLs(region string) (cloud, site, upgrade string) {
	if region == "cn" {
		return CNCloud, CNSite, CNUpgrade
	}
	return IntlCloud, IntlSite, IntlUpgrade
}

// FillAuthUrls fills missing cloud/site/upgrade URLs from a provided TLD.
func FillAuthUrls(cfg AuthConfig) AuthConfig {
	region := gatewayRegion(cfg.SiteURL)
	if region == "" {
		region = gatewayRegion(cfg.CloudURL)
	}
	if region == "" {
		region = gatewayRegion(cfg.UpgradeURL)
	}
	if region == "" {
		return cfg
	}
	cloud, site, upgrade := pairURLs(region)
	if cfg.CloudURL == "" {
		cfg.CloudURL = cloud
	}
	if cfg.SiteURL == "" {
		cfg.SiteURL = site
	}
	if cfg.UpgradeURL == "" {
		cfg.UpgradeURL = upgrade
	}
	return cfg
}

func localePrefersCN() bool {
	lang := strings.ToLower(os.Getenv("LANG") + os.Getenv("LC_ALL"))
	return strings.Contains(lang, "zh") || strings.Contains(lang, ".cn") || strings.HasPrefix(lang, "cn")
}

var probeDashboard = defaultProbeDashboard

func defaultProbeDashboard(siteURL, apiKey string) bool {
	url := strings.TrimRight(siteURL, "/") + "/api/dashboard/models"
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return false
	}
	req.Header.Set("User-Agent", "aria-engine-sdk/0.1.0")
	req.Header.Set("Authorization", "Bearer "+apiKey)
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode >= 200 && resp.StatusCode < 300
}

// DetectGatewayPair matches CLI detect: probe Dashboard with the key, else locale fallback.
func DetectGatewayPair(apiKey string) (cloud, site, upgrade string) {
	key := strings.TrimSpace(apiKey)
	first, second := "intl", "cn"
	if localePrefersCN() {
		first, second = "cn", "intl"
	}
	for _, region := range []string{first, second} {
		c, s, u := pairURLs(region)
		if key != "" && probeDashboard(s, key) {
			return c, s, u
		}
	}
	return pairURLs(first)
}

// ApplyAuth merges updates into existing. Validates; does not mutate existing.
func ApplyAuth(existing AuthConfig, updates AuthUpdates) (AuthConfig, error) {
	out := existing
	if updates.CloudAPIKey != nil {
		out.CloudAPIKey = *updates.CloudAPIKey
	}
	if updates.CloudURL != nil {
		out.CloudURL = *updates.CloudURL
	}
	if updates.SiteURL != nil {
		out.SiteURL = *updates.SiteURL
	}
	if updates.UpgradeURL != nil {
		out.UpgradeURL = *updates.UpgradeURL
	}
	if updates.HybridMode != nil {
		out.HybridMode = *updates.HybridMode
	}
	if updates.HybridExecution != nil {
		out.HybridExecution = *updates.HybridExecution
	}
	if updates.HybridSemantic != nil {
		out.HybridSemantic = *updates.HybridSemantic
	}
	if updates.HybridSemanticTimeoutMS != nil {
		out.HybridSemanticTimeoutMS = *updates.HybridSemanticTimeoutMS
	}
	if updates.HybridSemanticCacheSize != nil {
		out.HybridSemanticCacheSize = *updates.HybridSemanticCacheSize
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
	switch out.HybridMode {
	case "cost", "balance", "intelligence":
	default:
		return AuthConfig{}, fmt.Errorf("invalid hybrid_mode: %s", out.HybridMode)
	}
	switch out.HybridExecution {
	case "hybrid", "device", "cloud":
	default:
		return AuthConfig{}, fmt.Errorf("invalid hybrid_execution: %s", out.HybridExecution)
	}
	switch out.Compute {
	case "auto", "cpu", "cuda":
	default:
		return AuthConfig{}, fmt.Errorf("invalid compute: %s", out.Compute)
	}
	if out.HybridSemanticTimeoutMS <= 0 || out.HybridSemanticCacheSize <= 0 {
		return AuthConfig{}, fmt.Errorf("hybrid_semantic_timeout_ms / cache_size must be positive integers")
	}
	out = FillAuthUrls(out)
	if out.CloudAPIKey != "" && (out.CloudURL == "" || out.SiteURL == "" || out.UpgradeURL == "") {
		cloud, site, upgrade := DetectGatewayPair(out.CloudAPIKey)
		if out.CloudURL == "" {
			out.CloudURL = cloud
		}
		if out.SiteURL == "" {
			out.SiteURL = site
		}
		if out.UpgradeURL == "" {
			out.UpgradeURL = upgrade
		}
	}
	return out, nil
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
