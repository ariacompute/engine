package aria

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDefaultAuthConfig(t *testing.T) {
	cfg := DefaultAuthConfig()
	if cfg.HybridMode != "balance" || cfg.HybridExecution != "hybrid" || !cfg.HybridSemantic {
		t.Fatalf("defaults: %+v", cfg)
	}
	if cfg.HybridSemanticTimeoutMS != 800 || cfg.HybridSemanticCacheSize != 512 || cfg.Compute != "auto" {
		t.Fatalf("defaults: %+v", cfg)
	}
}

func TestApplyAuthInvalidEnum(t *testing.T) {
	mode := "fast"
	if _, err := ApplyAuth(DefaultAuthConfig(), AuthUpdates{HybridMode: &mode}); err == nil {
		t.Fatal("expected invalid hybrid_mode")
	}
	ex := "local"
	if _, err := ApplyAuth(DefaultAuthConfig(), AuthUpdates{HybridExecution: &ex}); err == nil {
		t.Fatal("expected invalid hybrid_execution")
	}
	comp := "gpu"
	if _, err := ApplyAuth(DefaultAuthConfig(), AuthUpdates{Compute: &comp}); err == nil {
		t.Fatal("expected invalid compute")
	}
}

func TestFillAuthUrlsFromCNSite(t *testing.T) {
	got := FillAuthUrls(AuthConfig{SiteURL: CNSite})
	if got.CloudURL != CNCloud || got.UpgradeURL != CNUpgrade || got.SiteURL != CNSite {
		t.Fatalf("got %+v", got)
	}
}

func TestAuthInstanceAllFields(t *testing.T) {
	eng := NewEngine()
	key, cloud, site, upgrade := "sk-test", CNCloud, CNSite, CNUpgrade
	mode, exec, compute := "cost", "device", "cpu"
	semantic := false
	timeout, cache := 250, 16
	hf, ms := "hf_abc", "ms_xyz"
	if err := eng.Auth(AuthUpdates{
		CloudAPIKey:             &key,
		CloudURL:                &cloud,
		SiteURL:                 &site,
		UpgradeURL:              &upgrade,
		HybridMode:              &mode,
		HybridExecution:         &exec,
		HybridSemantic:          &semantic,
		HybridSemanticTimeoutMS: &timeout,
		HybridSemanticCacheSize: &cache,
		Compute:                 &compute,
		HFToken:                 &hf,
		ModelScopeAPIToken:      &ms,
	}); err != nil {
		t.Fatal(err)
	}
	st := eng.AuthStatus()
	if st.CloudAPIKey != "sk-test" || st.HybridMode != "cost" || st.HybridExecution != "device" {
		t.Fatalf("%+v", st)
	}
	if st.HybridSemantic || st.HybridSemanticTimeoutMS != 250 || st.Compute != "cpu" {
		t.Fatalf("%+v", st)
	}
	if st.HFToken != "hf_abc" || st.ModelScopeAPIToken != "ms_xyz" || st.SiteURL != CNSite {
		t.Fatalf("%+v", st)
	}
}

func TestAuthPartialMerge(t *testing.T) {
	eng := NewEngine()
	hf, mode := "hf_one", "intelligence"
	if err := eng.Auth(AuthUpdates{HFToken: &hf, HybridMode: &mode}); err != nil {
		t.Fatal(err)
	}
	comp := "cuda"
	if err := eng.Auth(AuthUpdates{Compute: &comp}); err != nil {
		t.Fatal(err)
	}
	st := eng.AuthStatus()
	if st.HFToken != "hf_one" || st.HybridMode != "intelligence" || st.Compute != "cuda" {
		t.Fatalf("%+v", st)
	}
}

func TestAuthInvalidEnumLeavesState(t *testing.T) {
	eng := NewEngine()
	mode := "cost"
	if err := eng.Auth(AuthUpdates{HybridMode: &mode}); err != nil {
		t.Fatal(err)
	}
	bad := "nope"
	if err := eng.Auth(AuthUpdates{HybridMode: &bad}); err == nil {
		t.Fatal("expected error")
	}
	if eng.AuthStatus().HybridMode != "cost" {
		t.Fatalf("state changed: %+v", eng.AuthStatus())
	}
}

func TestAuthClearResetsInstance(t *testing.T) {
	eng := NewEngine()
	hf, mode := "hf_x", "cost"
	if err := eng.Auth(AuthUpdates{HFToken: &hf, HybridMode: &mode}); err != nil {
		t.Fatal(err)
	}
	eng.AuthClear()
	st := eng.AuthStatus()
	if st.HFToken != "" || st.HybridMode != "balance" {
		t.Fatalf("%+v", st)
	}
}

func TestAuthFillsUrlsFromSiteTLD(t *testing.T) {
	eng := NewEngine()
	site := "https://ariacompute.cn"
	if err := eng.Auth(AuthUpdates{SiteURL: &site}); err != nil {
		t.Fatal(err)
	}
	st := eng.AuthStatus()
	if st.CloudURL != CNCloud || st.UpgradeURL != CNUpgrade {
		t.Fatalf("%+v", st)
	}
}

func TestAuthDoesNotWriteConfigYml(t *testing.T) {
	home := t.TempDir()
	t.Setenv("ARIA_COMPUTE_HOME", home)
	eng := NewEngine()
	key, site, hf := "sk-test", "https://ariacompute.com", "hf_x"
	if err := eng.Auth(AuthUpdates{CloudAPIKey: &key, SiteURL: &site, HFToken: &hf}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(home, "config.yml")); err == nil {
		t.Fatal("config.yml was written")
	}
}

func TestAuthDetectUrlsFromKeyMocked(t *testing.T) {
	old := probeDashboard
	probeDashboard = func(site, key string) bool {
		return strings.Contains(site, "ariacompute.cn")
	}
	defer func() { probeDashboard = old }()
	eng := NewEngine()
	key := "sk-region"
	if err := eng.Auth(AuthUpdates{CloudAPIKey: &key}); err != nil {
		t.Fatal(err)
	}
	st := eng.AuthStatus()
	if st.SiteURL != CNSite || st.CloudURL != CNCloud || st.UpgradeURL != CNUpgrade {
		t.Fatalf("%+v", st)
	}
}
