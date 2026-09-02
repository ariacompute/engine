package ariaengine

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultSetupConfig(t *testing.T) {
	cfg := DefaultSetupConfig()
	if cfg.Compute != "auto" || cfg.Router != "" {
		t.Fatalf("defaults: %+v", cfg)
	}
}

func TestApplySetupInvalidEnum(t *testing.T) {
	comp := "gpu"
	if _, err := ApplySetup(DefaultSetupConfig(), SetupUpdates{Compute: &comp}); err == nil {
		t.Fatal("expected invalid compute")
	}
}

func TestFillSetupUrlsFromCNSite(t *testing.T) {
	got := FillSetupUrls(SetupConfig{SiteURL: CNSite})
	if got.UpgradeURL != CNUpgrade || got.SiteURL != CNSite {
		t.Fatalf("got %+v", got)
	}
}

func TestAuthInstanceAllFields(t *testing.T) {
	eng := NewEngine()
	router, site, upgrade := "http://127.0.0.1:8080", CNSite, CNUpgrade
	compute := "cpu"
	hf, ms := "hf_abc", "ms_xyz"
	if err := eng.Setup(SetupUpdates{
		Router:             &router,
		SiteURL:            &site,
		UpgradeURL:         &upgrade,
		Compute:            &compute,
		HFToken:            &hf,
		ModelScopeAPIToken: &ms,
	}); err != nil {
		t.Fatal(err)
	}
	st := eng.SetupStatus()
	if st.Router != router || st.Compute != "cpu" || st.HFToken != "hf_abc" || st.SiteURL != CNSite {
		t.Fatalf("%+v", st)
	}
}

func TestAuthPartialMerge(t *testing.T) {
	eng := NewEngine()
	hf, router := "hf_one", "http://127.0.0.1:1"
	if err := eng.Setup(SetupUpdates{HFToken: &hf, Router: &router}); err != nil {
		t.Fatal(err)
	}
	comp := "cuda"
	if err := eng.Setup(SetupUpdates{Compute: &comp}); err != nil {
		t.Fatal(err)
	}
	st := eng.SetupStatus()
	if st.HFToken != "hf_one" || st.Router != router || st.Compute != "cuda" {
		t.Fatalf("%+v", st)
	}
}

func TestAuthInvalidEnumLeavesState(t *testing.T) {
	eng := NewEngine()
	comp := "cpu"
	if err := eng.Setup(SetupUpdates{Compute: &comp}); err != nil {
		t.Fatal(err)
	}
	bad := "gpu"
	if err := eng.Setup(SetupUpdates{Compute: &bad}); err == nil {
		t.Fatal("expected error")
	}
	if eng.SetupStatus().Compute != "cpu" {
		t.Fatalf("state changed: %+v", eng.SetupStatus())
	}
}

func TestSetupClearResetsInstance(t *testing.T) {
	eng := NewEngine()
	hf, comp := "hf_x", "cpu"
	if err := eng.Setup(SetupUpdates{HFToken: &hf, Compute: &comp}); err != nil {
		t.Fatal(err)
	}
	eng.SetupClear()
	st := eng.SetupStatus()
	if st.HFToken != "" || st.Compute != "auto" {
		t.Fatalf("%+v", st)
	}
}

func TestAuthFillsUrlsFromSiteTLD(t *testing.T) {
	eng := NewEngine()
	site := "https://ariacompute.cn"
	if err := eng.Setup(SetupUpdates{SiteURL: &site}); err != nil {
		t.Fatal(err)
	}
	if eng.SetupStatus().UpgradeURL != CNUpgrade {
		t.Fatalf("%+v", eng.SetupStatus())
	}
}

func TestAuthDoesNotWriteConfigYml(t *testing.T) {
	home := t.TempDir()
	t.Setenv("ARIA_COMPUTE_HOME", home)
	eng := NewEngine()
	router, site, hf := "http://127.0.0.1:8080", "https://ariacompute.com", "hf_x"
	if err := eng.Setup(SetupUpdates{Router: &router, SiteURL: &site, HFToken: &hf}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(home, "config.yml")); err == nil {
		t.Fatal("config.yml was written")
	}
}
