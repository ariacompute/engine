//! Pinned AFM-D budgets (match `model/afm-d` contracts).

use serde::{Deserialize, Serialize};

pub const MAX_LEN: usize = 1024;
pub const HEAD_MAX_LEN: usize = 512;
pub const OPTION_DESC_MAX: usize = 96;
pub const MAX_CHOICES: usize = 255;
pub const MAX_OPTIONS_DD: usize = 16;
pub const DECISION_TEMPERATURE: f32 = 1.0;
pub const SHORTLIST_OPTION_THRESHOLD: usize = 40;
pub const DEFAULT_SHORTLIST_K: usize = 20;

pub const ENCODER_NAME: &str = "answerdotai/ModernBERT-large";
pub const BASE_LAYA: &str = "convaiinnovations/laya";
pub const DD_BASE_MODEL: &str = "openbmb/MiniCPM5-2B";
pub const DD_BASE_REVISION: &str = "12a3808a956f869c767195e9266b59c4d21d92e2";
pub const DD_DEFAULT_PORT: u16 = 8011;

pub const CHECKPOINT_CONFIG: &str = "rl_agent_config.json";
pub const CHECKPOINT_WEIGHTS: &str = "model.safetensors";
pub const DD_CONFIG: &str = "dd_config.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Track {
    Encoder,
    Decoder,
}

impl Track {
    pub fn as_str(self) -> &'static str {
        match self {
            Track::Encoder => "encoder",
            Track::Decoder => "decoder",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "encoder" | "afm_de" | "de" => Ok(Track::Encoder),
            "decoder" | "afm_dd" | "dd" => Ok(Track::Decoder),
            other => Err(format!("unknown track {other:?}; use encoder|decoder")),
        }
    }
}
