//! `~/.ariacompute` paths and `engine.yml`.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AriaConfig {
    /// Optional aria-router management URL (empty = do not register).
    #[serde(default)]
    pub router: String,
    /// Router API key for `PUT /v1/router/providers` (`sk-aria_…` or `bfvk-…`).
    #[serde(default)]
    pub router_api_key: String,
    #[serde(default)]
    pub site_url: String,
    /// Org root for CLI/FFI upgrades (`…/ariacompute`); runtime appends `/engine`.
    #[serde(default)]
    pub upgrade_url: String,
    #[serde(default = "default_compute")]
    pub compute: String,
    /// Optional Hugging Face hub token (`.com` gated/private files).
    #[serde(default)]
    pub hf_token: String,
    /// Optional ModelScope hub token (`.cn` gated/private files).
    #[serde(default)]
    pub modelscope_api_token: String,
}

/// Accept local (`sk-aria_`) or OAuth (`bfvk-`) keys; association is done by aria-router.
pub fn validate_router_api_key(key: &str) -> Result<(), String> {
    let t = key.trim();
    if t.is_empty() {
        return Ok(());
    }
    if t.starts_with("sk-aria_") || t.starts_with("bfvk-") {
        return Ok(());
    }
    Err("router_api_key must start with sk-aria_ or bfvk-".into())
}

fn default_compute() -> String {
    "auto".into()
}

fn keep_or_replace(existing: &str, entered: &str) -> String {
    if entered.is_empty() {
        existing.to_string()
    } else {
        entered.to_string()
    }
}

/// Merge one regional hub-token prompt into config fields.
/// `cn=true` (`.cn` / ModelScope) updates `modelscope_api_token`; otherwise `hf_token`.
/// Empty input keeps the current value; the other field is left as-is.
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

impl Default for AriaConfig {
    fn default() -> Self {
        Self {
            router: String::new(),
            router_api_key: String::new(),
            site_url: String::new(),
            upgrade_url: String::new(),
            compute: default_compute(),
            hf_token: String::new(),
            modelscope_api_token: String::new(),
        }
    }
}

/// Resolve `$HOME/.ariacompute` (overridable via `ARIA_COMPUTE_HOME` for tests).
pub fn aria_home() -> io::Result<PathBuf> {
    if let Ok(override_home) = std::env::var("ARIA_COMPUTE_HOME") {
        if !override_home.is_empty() {
            return Ok(PathBuf::from(override_home));
        }
    }
    let home = dirs::home_dir().ok_or_else(|| {
        io::Error::new(io::ErrorKind::NotFound, "could not resolve home directory")
    })?;
    Ok(home.join(".ariacompute"))
}

pub fn config_path() -> io::Result<PathBuf> {
    Ok(aria_home()?.join("engine.yml"))
}

pub fn legacy_config_path() -> io::Result<PathBuf> {
    Ok(aria_home()?.join("config.yml"))
}

pub fn models_dir() -> io::Result<PathBuf> {
    Ok(aria_home()?.join("models"))
}

pub fn lib_dir() -> io::Result<PathBuf> {
    Ok(aria_home()?.join("lib"))
}

pub fn model_dir(model: &str) -> io::Result<PathBuf> {
    Ok(models_dir()?.join(model))
}

/// Catalog / CLI names omit codebook-share. Local dirs may be `*_q326_channel`.
pub(crate) fn bundle_cache_aliases(model: &str) -> Vec<String> {
    let name = model.trim().trim_end_matches('/');
    if name.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for stem in quant_stems(name) {
        for share in ["", "_channel", "_group"] {
            let candidate = format!("{stem}{share}");
            if candidate != name {
                out.push(candidate);
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

fn quant_stems(name: &str) -> Vec<String> {
    let lower = name.to_ascii_lowercase();
    let core = if let Some(base) = lower.strip_suffix("_channel") {
        &name[..base.len()]
    } else if let Some(base) = lower.strip_suffix("_group") {
        &name[..base.len()]
    } else {
        name
    };
    let core_lower = core.to_ascii_lowercase();
    let mut stems = vec![core.to_string()];
    if let Some(base) = core_lower.strip_suffix("_q326") {
        stems.push(format!("{}_q3.26", &core[..base.len()]));
    } else if let Some(base) = core_lower.strip_suffix("_q3.26") {
        stems.push(format!("{}_q326", &core[..base.len()]));
    }
    stems.sort();
    stems.dedup();
    stems
}

pub fn ensure_aria_home() -> io::Result<PathBuf> {
    let root = aria_home()?;
    fs::create_dir_all(root.join("models"))?;
    fs::create_dir_all(root.join("lib"))?;
    Ok(root)
}

pub fn load_config() -> io::Result<AriaConfig> {
    let primary = config_path()?;
    let path = if primary.exists() {
        primary
    } else {
        let legacy = legacy_config_path()?;
        if legacy.exists() {
            legacy
        } else {
            return Ok(AriaConfig::default());
        }
    };
    let raw = fs::read_to_string(&path)?;
    let cfg: AriaConfig =
        serde_yaml::from_str(&raw).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(cfg)
}

pub fn save_config(cfg: &AriaConfig) -> io::Result<()> {
    ensure_aria_home()?;
    let path = config_path()?;
    let raw =
        serde_yaml::to_string(cfg).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let tmp = path.with_extension("yml.tmp");
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(raw.as_bytes())?;
        f.sync_all()?;
    }
    fs::rename(&tmp, &path)?;
    Ok(())
}

pub fn clear_config() -> io::Result<()> {
    for path in [config_path()?, legacy_config_path()?] {
        if path.exists() {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}

pub fn resolve_model_path(model: &str) -> io::Result<PathBuf> {
    let as_path = Path::new(model);
    if as_path.exists() {
        return Ok(as_path.to_path_buf());
    }
    let mut names = vec![model.to_string()];
    names.extend(bundle_cache_aliases(model));
    let mut last = model_dir(model)?;
    for name in names {
        let cached = model_dir(&name)?;
        last = cached.clone();
        if cached.exists() {
            return Ok(cached);
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "model not found: {model} (tried path and {})",
            last.display()
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tempfile::tempdir;

    /// Tests that mutate `ARIA_COMPUTE_HOME` must not run in parallel:
    /// `load_config` / `save_config` read that env for the home path.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn restore_aria_home(prev: Option<String>) {
        match prev {
            Some(v) => std::env::set_var("ARIA_COMPUTE_HOME", v),
            None => std::env::remove_var("ARIA_COMPUTE_HOME"),
        }
    }

    #[test]
    fn config_roundtrip() {
        let _g = ENV_LOCK.lock().unwrap();
        let dir = tempdir().unwrap();
        let prev = std::env::var("ARIA_COMPUTE_HOME").ok();
        std::env::set_var("ARIA_COMPUTE_HOME", dir.path());
        let cfg = AriaConfig {
            router: "http://127.0.0.1:8080".into(),
            router_api_key: "sk-aria_test".into(),
            site_url: "https://ariacompute.com".into(),
            upgrade_url: "https://github.com/ariacompute".into(),
            compute: "cpu".into(),
            hf_token: "hf_test".into(),
            modelscope_api_token: "ms_test".into(),
        };
        save_config(&cfg).unwrap();
        let loaded = load_config().unwrap();
        assert_eq!(loaded, cfg);
        clear_config().unwrap();
        assert!(!config_path().unwrap().exists());
        restore_aria_home(prev);
    }

    #[test]
    fn unknown_serve_fields_ignored_on_load() {
        let raw = "router_api_key: sk-aria_local\nserve_api_key: bfvk-cloud\nserve_site: cn\n";
        let cfg: AriaConfig = serde_yaml::from_str(raw).unwrap();
        assert_eq!(cfg.router_api_key, "sk-aria_local");
    }

    #[test]
    fn key_prefix_validation() {
        assert!(validate_router_api_key("sk-aria_abc").is_ok());
        assert!(validate_router_api_key("bfvk-xyz").is_ok());
        assert!(validate_router_api_key("").is_ok());
        assert!(validate_router_api_key("other").is_err());
    }

    #[test]
    fn legacy_config_yml_still_loads() {
        let _g = ENV_LOCK.lock().unwrap();
        let dir = tempdir().unwrap();
        let prev = std::env::var("ARIA_COMPUTE_HOME").ok();
        std::env::set_var("ARIA_COMPUTE_HOME", dir.path());
        fs::write(
            dir.path().join("config.yml"),
            "hf_token: hf_legacy\nrouter: http://127.0.0.1:8090\n",
        )
        .unwrap();
        let loaded = load_config().unwrap();
        assert_eq!(loaded.hf_token, "hf_legacy");
        assert_eq!(loaded.router, "http://127.0.0.1:8090");
        assert!(!config_path().unwrap().exists());
        restore_aria_home(prev);
    }

    #[test]
    fn missing_upgrade_url_defaults_empty() {
        let raw = "site_url: https://ariacompute.com\n";
        let cfg: AriaConfig = serde_yaml::from_str(raw).unwrap();
        assert!(cfg.upgrade_url.is_empty());
        assert!(cfg.router.is_empty());
        assert_eq!(cfg.compute, "auto");
        assert!(cfg.hf_token.is_empty());
        assert!(cfg.modelscope_api_token.is_empty());
    }

    #[test]
    fn hub_tokens_parse_when_present() {
        let raw = "hf_token: hf_abc\nmodelscope_api_token: ms_xyz\n";
        let cfg: AriaConfig = serde_yaml::from_str(raw).unwrap();
        assert_eq!(cfg.hf_token, "hf_abc");
        assert_eq!(cfg.modelscope_api_token, "ms_xyz");
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
    fn router_field_parses() {
        let raw = "router: http://127.0.0.1:8080\n";
        let cfg: AriaConfig = serde_yaml::from_str(raw).unwrap();
        assert_eq!(cfg.router, "http://127.0.0.1:8080");
    }

    #[test]
    fn q326_aliases_include_channel() {
        let aliases = bundle_cache_aliases("gemma-3-1b-it_q326");
        assert!(aliases.contains(&"gemma-3-1b-it_q326_channel".into()));
        assert!(aliases.contains(&"gemma-3-1b-it_q3.26".into()));
        let back = bundle_cache_aliases("gemma-3-1b-it_q326_channel");
        assert!(back.contains(&"gemma-3-1b-it_q326".into()));
    }
}
