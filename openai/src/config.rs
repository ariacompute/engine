//! `~/.ariacompute` paths and `config.yml`.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AriaConfig {
    #[serde(default)]
    pub cloud_api_key: String,
    #[serde(default)]
    pub cloud_url: String,
    #[serde(default)]
    pub site_url: String,
    /// Org root for CLI/FFI upgrades (`…/ariacompute`); runtime appends `/engine`.
    #[serde(default)]
    pub upgrade_url: String,
    #[serde(default = "default_hybrid_mode")]
    pub hybrid_mode: String,
    #[serde(default = "default_hybrid_execution")]
    pub hybrid_execution: String,
}

fn default_hybrid_mode() -> String {
    "balance".into()
}

fn default_hybrid_execution() -> String {
    "hybrid".into()
}

impl Default for AriaConfig {
    fn default() -> Self {
        Self {
            cloud_api_key: String::new(),
            cloud_url: String::new(),
            site_url: String::new(),
            upgrade_url: String::new(),
            hybrid_mode: default_hybrid_mode(),
            hybrid_execution: default_hybrid_execution(),
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

pub fn ensure_aria_home() -> io::Result<PathBuf> {
    let root = aria_home()?;
    fs::create_dir_all(root.join("models"))?;
    fs::create_dir_all(root.join("lib"))?;
    Ok(root)
}

pub fn load_config() -> io::Result<AriaConfig> {
    let path = config_path()?;
    if !path.exists() {
        return Ok(AriaConfig::default());
    }
    let raw = fs::read_to_string(&path)?;
    let cfg: AriaConfig = serde_yaml::from_str(&raw)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(cfg)
}

pub fn save_config(cfg: &AriaConfig) -> io::Result<()> {
    ensure_aria_home()?;
    let path = config_path()?;
    let raw = serde_yaml::to_string(cfg)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
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
    let path = config_path()?;
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

pub fn resolve_model_path(model: &str) -> io::Result<PathBuf> {
    let as_path = Path::new(model);
    if as_path.exists() {
        return Ok(as_path.to_path_buf());
    }
    let cached = model_dir(model)?;
    if cached.exists() {
        return Ok(cached);
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "model not found: {model} (tried path and {})",
            cached.display()
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn config_roundtrip() {
        let dir = tempdir().unwrap();
        std::env::set_var("ARIA_COMPUTE_HOME", dir.path());
        let cfg = AriaConfig {
            cloud_api_key: "sk-test".into(),
            cloud_url: "https://gateway.ariacompute.com".into(),
            site_url: "https://ariacompute.com".into(),
            upgrade_url: "https://github.com/ariacompute".into(),
            hybrid_mode: "cost".into(),
            hybrid_execution: "device".into(),
        };
        save_config(&cfg).unwrap();
        let loaded = load_config().unwrap();
        assert_eq!(loaded, cfg);
        clear_config().unwrap();
        assert!(!config_path().unwrap().exists());
        std::env::remove_var("ARIA_COMPUTE_HOME");
    }

    #[test]
    fn missing_upgrade_url_defaults_empty() {
        let raw = "cloud_api_key: x\ncloud_url: https://gateway.ariacompute.com\nsite_url: https://ariacompute.com\n";
        let cfg: AriaConfig = serde_yaml::from_str(raw).unwrap();
        assert!(cfg.upgrade_url.is_empty());
        assert_eq!(cfg.hybrid_mode, "balance");
    }
}
