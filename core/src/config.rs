use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::gateway::{preferred_hub, PublicHub};

/// Resolve `$HOME/.ariacompute` (overridable via `ARIA_COMPUTE_HOME`).
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
        other => Err(format!(
            "invalid compute {other:?}; expected auto|cpu|cuda"
        )),
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
    let text = serde_yaml::to_string(cfg)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
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
    }

    #[test]
    fn parse_compute_ok() {
        assert_eq!(parse_compute("CUDA").unwrap(), "cuda");
        assert!(parse_compute("gpu").is_err());
    }
}
