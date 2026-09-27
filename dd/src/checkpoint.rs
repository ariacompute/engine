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
