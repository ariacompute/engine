use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

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

pub fn load_config() -> io::Result<AriaConfig> {
    let path = engine_yml_path()?;
    if !path.is_file() {
        return Ok(AriaConfig::default());
    }
    let text = fs::read_to_string(&path)?;
    let cfg: AriaConfig = serde_yaml::from_str(&text)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(cfg)
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
        fs::remove_file(path)?;
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
