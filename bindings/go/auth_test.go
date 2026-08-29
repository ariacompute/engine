package aria

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultAuthConfig(t *testing.T) {
	cfg := DefaultAuthConfig()
	if cfg.Compute != "auto" || cfg.Router != "" {
		t.Fatalf("defaults: %+v", cfg)
	}
}

func TestApplyAuthInvalidEnum(t *testing.T) {
	comp := "gpu"
	if _, err := ApplyAuth(DefaultAuthConfig(), AuthUpdates{Compute: &comp}); err == nil {
		t.Fatal("expected invalid compute")
	}
}

func TestFillAuthUrlsFromCNSite(t *testing.T) {
	got := FillAuthUrls(AuthConfig{SiteURL: CNSite})
	if got.UpgradeURL != CNUpgrade || got.SiteURL != CNSite {
		t.Fatalf("got %+v", got)
	}
}

func TestAuthInstanceAllFields(t *testing.T) {
	eng := NewEngine()
	router, site, upgrade := "http://127.0.0.1:8080", CNSite, CNUpgrade
	compute := "cpu"
	hf, ms := "hf_abc", "ms_xyz"
	if err := eng.Auth(AuthUpdates{
		Router:             &router,
		SiteURL:            &site,
		UpgradeURL:         &upgrade,
		Compute:            &compute,
		HFToken:            &hf,
		ModelScopeAPIToken: &ms,
	}); err != nil {
		t.Fatal(err)
	}
	st := eng.AuthStatus()
	if st.Router != router || st.Compute != "cpu" || st.HFToken != "hf_abc" || st.SiteURL != CNSite {
		t.Fatalf("%+v", st)
	}
}

func TestAuthPartialMerge(t *testing.T) {
	eng := NewEngine()
	hf, router := "hf_one", "http://127.0.0.1:1"
	if err := eng.Auth(AuthUpdates{HFToken: &hf, Router: &router}); err != nil {
		t.Fatal(err)
	}
	comp := "cuda"
	if err := eng.Auth(AuthUpdates{Compute: &comp}); err != nil {
		t.Fatal(err)
	}
	st := eng.AuthStatus()
	if st.HFToken != "hf_one" || st.Router != router || st.Compute != "cuda" {
		t.Fatalf("%+v", st)
	}
}

func TestAuthInvalidEnumLeavesState(t *testing.T) {
	eng := NewEngine()
	comp := "cpu"
	if err := eng.Auth(AuthUpdates{Compute: &comp}); err != nil {
		t.Fatal(err)
	}
	bad := "gpu"
	if err := eng.Auth(AuthUpdates{Compute: &bad}); err == nil {
		t.Fatal("expected error")
	}
	if eng.AuthStatus().Compute != "cpu" {
		t.Fatalf("state changed: %+v", eng.AuthStatus())
	}
}

func TestAuthClearResetsInstance(t *testing.T) {
	eng := NewEngine()
	hf, comp := "hf_x", "cpu"
	if err := eng.Auth(AuthUpdates{HFToken: &hf, Compute: &comp}); err != nil {
		t.Fatal(err)
	}
	eng.AuthClear()
	st := eng.AuthStatus()
	if st.HFToken != "" || st.Compute != "auto" {
		t.Fatalf("%+v", st)
	}
}

func TestAuthFillsUrlsFromSiteTLD(t *testing.T) {
	eng := NewEngine()
	site := "https://ariacompute.cn"
	if err := eng.Auth(AuthUpdates{SiteURL: &site}); err != nil {
		t.Fatal(err)
	}
	if eng.AuthStatus().UpgradeURL != CNUpgrade {
		t.Fatalf("%+v", eng.AuthStatus())
	}
}

func TestAuthDoesNotWriteConfigYml(t *testing.T) {
	home := t.TempDir()
	t.Setenv("ARIA_COMPUTE_HOME", home)
	eng := NewEngine()
	router, site, hf := "http://127.0.0.1:8080", "https://ariacompute.com", "hf_x"
	if err := eng.Auth(AuthUpdates{Router: &router, SiteURL: &site, HFToken: &hf}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(home, "config.yml")); err == nil {
		t.Fatal("config.yml was written")
	}
}
