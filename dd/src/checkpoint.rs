use std::fs;
use std::path::{Path, PathBuf};

use ariacompute_core::contract::{DD_BASE_MODEL, DD_BASE_REVISION, DD_CONFIG};
use ariacompute_core::error::{AfmError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DdConfig {
    #[serde(default = "default_base")]
    pub base_model: String,
    #[serde(default = "default_rev")]
    pub base_revision: String,
    #[serde(default)]
    pub prompt_version: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

fn default_base() -> String {
    DD_BASE_MODEL.into()
}
fn default_rev() -> String {
    DD_BASE_REVISION.into()
}

impl Default for DdConfig {
    fn default() -> Self {
        Self {
            base_model: default_base(),
            base_revision: default_rev(),
            prompt_version: Some("afm-d-decoder-v1".into()),
            extra: serde_json::Map::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecoderCheckpoint {
    pub root: Option<PathBuf>,
    pub config: DdConfig,
    pub adapter_dir: Option<PathBuf>,
}

impl DecoderCheckpoint {
    pub fn open(root: Option<impl AsRef<Path>>) -> Result<Self> {
        let root = root.map(|r| r.as_ref().to_path_buf());
        let mut config = DdConfig::default();
        let mut adapter_dir = None;
        if let Some(ref dir) = root {
            if !dir.is_dir() {
                return Err(AfmError::msg(format!(
                    "decoder checkpoint not a directory: {}",
                    dir.display()
                )));
            }
            let cfg_path = dir.join(DD_CONFIG);
            if cfg_path.is_file() {
                let text = fs::read_to_string(&cfg_path)?;
                config = serde_json::from_str(&text)?;
            }
            if dir.join("adapter_config.json").is_file() {
                adapter_dir = Some(dir.clone());
            } else if dir.join("adapter").join("adapter_config.json").is_file() {
                adapter_dir = Some(dir.join("adapter"));
            }
        }
        Ok(Self {
            root,
            config,
            adapter_dir,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn no_root_uses_contract_defaults() {
        let ckpt = DecoderCheckpoint::open(None::<&str>).unwrap();
        assert!(ckpt.root.is_none());
        assert!(ckpt.adapter_dir.is_none());
        assert_eq!(ckpt.config.base_model, DD_BASE_MODEL);
        assert_eq!(ckpt.config.base_revision, DD_BASE_REVISION);
        assert_eq!(
            ckpt.config.prompt_version.as_deref(),
            Some("afm-d-decoder-v1")
        );
    }

    #[test]
    fn empty_dir_yields_defaults() {
        let dir = tempdir().unwrap();
        let ckpt = DecoderCheckpoint::open(Some(dir.path())).unwrap();
        assert_eq!(ckpt.root.as_deref(), Some(dir.path()));
        assert!(ckpt.adapter_dir.is_none());
        assert_eq!(ckpt.config.base_model, DD_BASE_MODEL);
    }

    #[test]
    fn reads_dd_config_and_keeps_extra_fields() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join(DD_CONFIG),
            r#"{"base_model":"local/base","base_revision":"rev9","prompt_version":"v2","custom":1}"#,
        )
        .unwrap();
        let ckpt = DecoderCheckpoint::open(Some(dir.path())).unwrap();
        assert_eq!(ckpt.config.base_model, "local/base");
        assert_eq!(ckpt.config.base_revision, "rev9");
        assert_eq!(ckpt.config.prompt_version.as_deref(), Some("v2"));
        assert_eq!(ckpt.config.extra["custom"], 1);
    }

    #[test]
    fn detects_adapter_at_root_or_one_level_down() {
        let root = tempdir().unwrap();
        fs::write(root.path().join("adapter_config.json"), "{}").unwrap();
        let ckpt = DecoderCheckpoint::open(Some(root.path())).unwrap();
        assert_eq!(ckpt.adapter_dir.as_deref(), Some(root.path()));

        let nested = tempdir().unwrap();
        fs::create_dir_all(nested.path().join("adapter")).unwrap();
        fs::write(nested.path().join("adapter/adapter_config.json"), "{}").unwrap();
        let ckpt = DecoderCheckpoint::open(Some(nested.path())).unwrap();
        assert_eq!(ckpt.adapter_dir, Some(nested.path().join("adapter")));
    }

    #[test]
    fn rejects_missing_directory() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("nope");
        let err = DecoderCheckpoint::open(Some(&missing)).unwrap_err();
        assert!(err
            .to_string()
            .contains("decoder checkpoint not a directory"));
    }

    #[test]
    fn rejects_invalid_dd_config() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join(DD_CONFIG), "{oops").unwrap();
        let err = DecoderCheckpoint::open(Some(dir.path())).unwrap_err();
        assert!(matches!(err, AfmError::Json(_)));
    }
}
