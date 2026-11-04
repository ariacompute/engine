use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::gateway::{preferred_hub, PublicHub};

/// Resolve `$HOME/.ariacompute` (overridable via `ARIA_COMPUTE_HOME`).
pub fn aria_home() -> io::Result<PathBuf> {
    let override_home = std::env::var("ARIA_COMPUTE_HOME").ok();
    resolve_aria_home(override_home.as_deref(), dirs::home_dir())
}

/// Pure resolution step (`ARIA_COMPUTE_HOME` wins, empty value falls through).
fn resolve_aria_home(override_home: Option<&str>, home: Option<PathBuf>) -> io::Result<PathBuf> {
    if let Some(dir) = override_home.filter(|h| !h.is_empty()) {
        return Ok(PathBuf::from(dir));
    }
    let home = home.ok_or_else(|| {
        io::Error::new(io::ErrorKind::NotFound, "could not resolve home directory")
    })?;
    Ok(home.join(".ariacompute"))
}

pub fn ensure_aria_home() -> io::Result<PathBuf> {
    let home = aria_home()?;
    fs::create_dir_all(&home)?;
    fs::create_dir_all(home.join("models"))?;
    fs::create_dir_all(home.join("lib"))?;
    Ok(home)
}

pub fn models_dir() -> io::Result<PathBuf> {
    Ok(aria_home()?.join("models"))
}

pub fn lib_dir() -> io::Result<PathBuf> {
    Ok(aria_home()?.join("lib"))
}

pub fn engine_yml_path() -> io::Result<PathBuf> {
    Ok(aria_home()?.join("engine.yml"))
}

pub fn legacy_config_path() -> io::Result<PathBuf> {
    Ok(aria_home()?.join("config.yml"))
}

/// Five-field engine.yml (no router).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AriaConfig {
    #[serde(default)]
    pub site_url: String,
    #[serde(default)]
    pub upgrade_url: String,
    #[serde(default = "default_compute")]
    pub compute: String,
    #[serde(default)]
    pub hf_token: String,
    #[serde(default)]
    pub modelscope_api_token: String,
}

fn default_compute() -> String {
    "auto".into()
}

impl Default for AriaConfig {
    fn default() -> Self {
        Self {
            site_url: String::new(),
            upgrade_url: String::new(),
            compute: default_compute(),
            hf_token: String::new(),
            modelscope_api_token: String::new(),
        }
    }
}

fn keep_or_replace(existing: &str, entered: &str) -> String {
    if entered.is_empty() {
        existing.to_string()
    } else {
        entered.to_string()
    }
}

/// `.com` → update HF token; `.cn` → update ModelScope token. The other field is left as-is.
pub fn apply_hub_token_input(existing: &AriaConfig, cn: bool, entered: &str) -> (String, String) {
    if cn {
        (
            existing.hf_token.clone(),
            keep_or_replace(&existing.modelscope_api_token, entered),
        )
    } else {
        (
            keep_or_replace(&existing.hf_token, entered),
            existing.modelscope_api_token.clone(),
        )
    }
}

/// Hub token for the preferred public hub of `site_url` (empty → None).
pub fn hub_token_for_site(cfg: &AriaConfig) -> Option<String> {
    let raw = match preferred_hub(&cfg.site_url) {
        PublicHub::HuggingFace => cfg.hf_token.as_str(),
        PublicHub::ModelScope => cfg.modelscope_api_token.as_str(),
    };
    let t = raw.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

pub fn parse_compute(s: &str) -> Result<String, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "auto" | "cpu" | "cuda" => Ok(s.trim().to_ascii_lowercase()),
        other => Err(format!("invalid compute {other:?}; expected auto|cpu|cuda")),
    }
}

pub fn load_config() -> io::Result<AriaConfig> {
    let path = engine_yml_path()?;
    if path.is_file() {
        let text = fs::read_to_string(&path)?;
        let cfg: AriaConfig = serde_yaml::from_str(&text)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        return Ok(cfg);
    }
    let legacy = legacy_config_path()?;
    if legacy.is_file() {
        let text = fs::read_to_string(&legacy)?;
        let cfg: AriaConfig = serde_yaml::from_str(&text)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        return Ok(cfg);
    }
    Ok(AriaConfig::default())
}

pub fn save_config(cfg: &AriaConfig) -> io::Result<()> {
    ensure_aria_home()?;
    let path = engine_yml_path()?;
    let tmp = path.with_extension("yml.tmp");
    let text =
        serde_yaml::to_string(cfg).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(&tmp, text)?;
    fs::rename(&tmp, &path)?;
    Ok(())
}

pub fn clear_config() -> io::Result<()> {
    let path = engine_yml_path()?;
    if path.is_file() {
        fs::remove_file(&path)?;
    }
    let legacy = legacy_config_path()?;
    if legacy.is_file() {
        fs::remove_file(&legacy)?;
    }
    Ok(())
}

pub fn model_cache_dir(model: &str) -> io::Result<PathBuf> {
    Ok(models_dir()?.join(model))
}

pub fn resolve_checkpoint(ref_: &str) -> PathBuf {
    let p = Path::new(ref_);
    if p.exists() || ref_.contains('/') || ref_.contains('\\') {
        return p.to_path_buf();
    }
    models_dir()
        .map(|d| d.join(ref_))
        .unwrap_or_else(|_| p.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use tempfile::tempdir;

    /// Run `f` with `ARIA_COMPUTE_HOME` pointing at a throwaway home.
    fn with_home<T>(f: impl FnOnce(&Path) -> T) -> T {
        let dir = tempdir().unwrap();
        std::env::set_var("ARIA_COMPUTE_HOME", dir.path());
        let out = f(dir.path());
        std::env::remove_var("ARIA_COMPUTE_HOME");
        out
    }

    #[test]
    fn apply_hub_token_input_intl_only_updates_hf() {
        let existing = AriaConfig {
            hf_token: "old_hf".into(),
            modelscope_api_token: "old_ms".into(),
            ..Default::default()
        };
        let (hf, ms) = apply_hub_token_input(&existing, false, "new_hf");
        assert_eq!(hf, "new_hf");
        assert_eq!(ms, "old_ms");
        let (hf, ms) = apply_hub_token_input(&existing, false, "");
        assert_eq!(hf, "old_hf");
        assert_eq!(ms, "old_ms");
    }

    #[test]
    fn apply_hub_token_input_cn_only_updates_modelscope() {
        let existing = AriaConfig {
            hf_token: "old_hf".into(),
            modelscope_api_token: "old_ms".into(),
            ..Default::default()
        };
        let (hf, ms) = apply_hub_token_input(&existing, true, "new_ms");
        assert_eq!(hf, "old_hf");
        assert_eq!(ms, "new_ms");
        let (hf, ms) = apply_hub_token_input(&existing, true, "");
        assert_eq!(hf, "old_hf");
        assert_eq!(ms, "old_ms");
    }

    #[test]
    fn parse_compute_ok() {
        assert_eq!(parse_compute("CUDA").unwrap(), "cuda");
        assert_eq!(parse_compute(" Cpu ").unwrap(), "cpu");
        assert_eq!(parse_compute("auto").unwrap(), "auto");
        assert!(parse_compute("gpu").is_err());
    }

    #[test]
    fn defaults_are_the_five_documented_fields() {
        let cfg = AriaConfig::default();
        assert_eq!(cfg.compute, "auto");
        assert!(cfg.site_url.is_empty());
        assert!(cfg.upgrade_url.is_empty());
        assert!(cfg.hf_token.is_empty());
        assert!(cfg.modelscope_api_token.is_empty());

        let parsed: AriaConfig = serde_yaml::from_str("site_url: https://ariacompute.com").unwrap();
        assert_eq!(parsed.compute, "auto");
        assert_eq!(parsed.site_url, "https://ariacompute.com");
    }

    #[test]
    #[serial]
    fn aria_home_honours_env_override() {
        let dir = tempdir().unwrap();
        std::env::set_var("ARIA_COMPUTE_HOME", dir.path());
        assert_eq!(aria_home().unwrap(), dir.path());
        assert_eq!(models_dir().unwrap(), dir.path().join("models"));
        assert_eq!(lib_dir().unwrap(), dir.path().join("lib"));
        assert_eq!(engine_yml_path().unwrap(), dir.path().join("engine.yml"));
        assert_eq!(legacy_config_path().unwrap(), dir.path().join("config.yml"));
        std::env::remove_var("ARIA_COMPUTE_HOME");
    }

    #[test]
    #[serial]
    fn aria_home_falls_back_to_dot_ariacompute() {
        std::env::set_var("ARIA_COMPUTE_HOME", "");
        let home = aria_home().unwrap();
        assert!(home.ends_with(".ariacompute"), "{}", home.display());
        std::env::remove_var("ARIA_COMPUTE_HOME");
    }

    #[test]
    fn aria_home_without_home_dir_is_an_error() {
        let err = resolve_aria_home(None, None).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        assert!(err.to_string().contains("could not resolve home directory"));

        // empty override falls through to $HOME/.ariacompute
        assert_eq!(
            resolve_aria_home(Some(""), Some(PathBuf::from("/home/u"))).unwrap(),
            PathBuf::from("/home/u/.ariacompute")
        );
        // non-empty override wins verbatim
        assert_eq!(
            resolve_aria_home(Some("/custom"), Some(PathBuf::from("/home/u"))).unwrap(),
            PathBuf::from("/custom")
        );
    }

    #[test]
    #[serial]
    fn ensure_aria_home_creates_models_and_lib() {
        with_home(|home| {
            let created = ensure_aria_home().unwrap();
            assert_eq!(created, home);
            assert!(home.join("models").is_dir());
            assert!(home.join("lib").is_dir());
        });
    }

    #[test]
    #[serial]
    fn save_and_load_round_trip() {
        with_home(|home| {
            let cfg = AriaConfig {
                site_url: "https://ariacompute.cn".into(),
                upgrade_url: "https://gitee.com/ariacompute".into(),
                compute: "cpu".into(),
                hf_token: "hf".into(),
                modelscope_api_token: "ms".into(),
            };
            save_config(&cfg).unwrap();
            assert!(home.join("engine.yml").is_file());
            assert_eq!(load_config().unwrap(), cfg);

            // engine.yml wins over the legacy config.yml
            fs::write(
                home.join("config.yml"),
                "site_url: https://legacy.example\ncompute: cuda\n",
            )
            .unwrap();
            assert_eq!(load_config().unwrap(), cfg);
        });
    }

    #[test]
    #[serial]
    fn load_config_reads_legacy_yaml_and_defaults() {
        with_home(|home| {
            assert_eq!(load_config().unwrap(), AriaConfig::default());
            fs::write(
                home.join("config.yml"),
                "site_url: https://legacy.example\ncompute: cuda\nhf_token: tok\n",
            )
            .unwrap();
            let cfg = load_config().unwrap();
            assert_eq!(cfg.site_url, "https://legacy.example");
            assert_eq!(cfg.compute, "cuda");
            assert_eq!(cfg.hf_token, "tok");
        });
    }

    #[test]
    #[serial]
    fn broken_config_yaml_is_invalid_data() {
        with_home(|home| {
            fs::write(home.join("engine.yml"), "compute: [1,\n").unwrap();
            let err = load_config().unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::InvalidData);

            fs::remove_file(home.join("engine.yml")).unwrap();
            fs::write(home.join("config.yml"), "compute: [1,\n").unwrap();
            assert_eq!(
                load_config().unwrap_err().kind(),
                io::ErrorKind::InvalidData
            );
        });
    }

    #[test]
    #[serial]
    fn clear_config_removes_engine_and_legacy_files() {
        with_home(|home| {
            write(home.join("engine.yml"), "compute: cpu\n");
            write(home.join("config.yml"), "compute: cpu\n");
            clear_config().unwrap();
            assert!(!home.join("engine.yml").exists());
            assert!(!home.join("config.yml").exists());
            // idempotent when nothing is left
            clear_config().unwrap();
        });
    }

    #[test]
    #[serial]
    fn model_cache_dir_is_under_models() {
        with_home(|home| {
            assert_eq!(
                model_cache_dir("afm-de").unwrap(),
                home.join("models").join("afm-de")
            );
        });
    }

    #[test]
    #[serial]
    fn resolve_checkpoint_keeps_paths_and_maps_bare_names() {
        with_home(|home| {
            assert_eq!(
                resolve_checkpoint("afm-de"),
                home.join("models").join("afm-de")
            );
            assert_eq!(
                resolve_checkpoint("vendor/afm-de"),
                Path::new("vendor/afm-de")
            );
            assert_eq!(
                resolve_checkpoint("vendor\\afm-de"),
                Path::new("vendor\\afm-de")
            );
            let existing = home.join("models").join("local");
            fs::create_dir_all(&existing).unwrap();
            assert_eq!(resolve_checkpoint(existing.to_str().unwrap()), existing);
        });
    }

    #[test]
    fn hub_token_for_site_picks_the_regional_secret() {
        let cfg = AriaConfig {
            site_url: "https://ariacompute.com".into(),
            hf_token: "  hf-tok  ".into(),
            modelscope_api_token: "ms-tok".into(),
            ..Default::default()
        };
        assert_eq!(hub_token_for_site(&cfg).as_deref(), Some("hf-tok"));

        let cn = AriaConfig {
            site_url: "https://ariacompute.cn".into(),
            ..cfg.clone()
        };
        assert_eq!(hub_token_for_site(&cn).as_deref(), Some("ms-tok"));

        let blank = AriaConfig {
            site_url: "https://ariacompute.com".into(),
            hf_token: "   ".into(),
            ..Default::default()
        };
        assert_eq!(hub_token_for_site(&blank), None);
    }

    fn write(path: PathBuf, body: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, body).unwrap();
    }
}
