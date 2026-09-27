use std::fs;
use std::path::{Path, PathBuf};

use ariacompute_core::contract::{CHECKPOINT_CONFIG, CHECKPOINT_WEIGHTS, HEAD_MAX_LEN, MAX_LEN};
use ariacompute_core::error::{AfmError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    #[serde(default = "default_encoder")]
    pub encoder: String,
    #[serde(default = "default_head_layers")]
    pub head_layers: usize,
    #[serde(default = "default_max_len")]
    pub max_len: usize,
    #[serde(default = "default_head_max_len")]
    pub head_max_len: usize,
    #[serde(default)]
    pub temperature: Option<Value>,
    #[serde(default)]
    pub decision_temperatures: Option<Value>,
    #[serde(default)]
    pub confidence_temperatures: Option<Value>,
    #[serde(default)]
    pub confidence_modes: Option<Value>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

fn default_encoder() -> String {
    ariacompute_core::contract::ENCODER_NAME.into()
}
fn default_head_layers() -> usize {
    2
}
fn default_max_len() -> usize {
    MAX_LEN
}
fn default_head_max_len() -> usize {
    HEAD_MAX_LEN
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            encoder: default_encoder(),
            head_layers: default_head_layers(),
            max_len: default_max_len(),
            head_max_len: default_head_max_len(),
            temperature: None,
            decision_temperatures: None,
            confidence_temperatures: None,
            confidence_modes: None,
            extra: serde_json::Map::new(),
        }
    }
}

pub fn load_agent_config(dir: &Path) -> Result<AgentConfig> {
    let path = dir.join(CHECKPOINT_CONFIG);
    if !path.is_file() {
        return Ok(AgentConfig::default());
    }
    let text = fs::read_to_string(&path)?;
    Ok(serde_json::from_str(&text)?)
}

#[derive(Debug, Clone)]
pub struct EncoderCheckpoint {
    pub root: PathBuf,
    pub config: AgentConfig,
    pub weights_path: PathBuf,
    pub tokenizer_dir: PathBuf,
    pub encoder_dir: PathBuf,
}

impl EncoderCheckpoint {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        if !root.is_dir() {
            return Err(AfmError::msg(format!(
                "encoder checkpoint not a directory: {}",
                root.display()
            )));
        }
        let config = load_agent_config(&root)?;
        let weights_path = root.join(CHECKPOINT_WEIGHTS);
        let tokenizer_dir = {
            let t = root.join("tokenizer");
            if t.is_dir() {
                t
            } else {
                root.clone()
            }
        };
        let encoder_dir = {
            let e = root.join("encoder");
            if e.is_dir() {
                e
            } else {
                root.clone()
            }
        };
        Ok(Self {
            root,
            config,
            weights_path,
            tokenizer_dir,
            encoder_dir,
        })
    }

    pub fn has_weights(&self) -> bool {
        self.weights_path.is_file()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn open_minimal_checkpoint() {
        let dir = tempdir().unwrap();
        let cfg = dir.path().join(CHECKPOINT_CONFIG);
        let mut f = fs::File::create(&cfg).unwrap();
        write!(f, r#"{{"encoder":"x","max_len":1024,"head_max_len":512}}"#).unwrap();
        let ckpt = EncoderCheckpoint::open(dir.path()).unwrap();
        assert_eq!(ckpt.config.max_len, 1024);
        assert!(!ckpt.has_weights());
    }
}
