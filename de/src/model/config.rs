//! HF `config.json` → candle ModernBERT config (rope / layer-norm fallbacks).

use std::fs;
use std::path::Path;

use candle_transformers::models::modernbert::Config as ModernBertConfig;
use serde::Deserialize;

use ariacompute_core::error::{AfmError, Result};

/// Load encoder/config.json into candle ModernBERT config (maps HF rope_parameters).
pub fn load_modernbert_config(encoder_dir: &Path) -> Result<ModernBertConfig> {
    let path = encoder_dir.join("config.json");
    let text = fs::read_to_string(&path)
        .map_err(|e| AfmError::msg(format!("read {}: {e}", path.display())))?;
    let hf: HfModernBertConfig = serde_json::from_str(&text)?;
    Ok(hf.into_candle())
}

/// Remap Laya `encoder.*` weights to candle `model.*`; other keys pass through.
pub fn remap_encoder_key(key: &str) -> String {
    match key.strip_prefix("encoder.") {
        Some(rest) => format!("model.{rest}"),
        None => key.to_string(),
    }
}

#[derive(Debug, Deserialize)]
struct HfModernBertConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    #[serde(default = "default_eps")]
    layer_norm_eps: f64,
    #[serde(default = "default_eps")]
    norm_eps: f64,
    pad_token_id: u32,
    global_attn_every_n_layers: usize,
    local_attention: usize,
    #[serde(default)]
    global_rope_theta: Option<f64>,
    #[serde(default)]
    local_rope_theta: Option<f64>,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
}

#[derive(Debug, Deserialize)]
struct RopeParameters {
    #[serde(default)]
    full_attention: Option<RopeTheta>,
    #[serde(default)]
    sliding_attention: Option<RopeTheta>,
}

#[derive(Debug, Deserialize)]
struct RopeTheta {
    rope_theta: f64,
}

fn default_eps() -> f64 {
    1e-5
}

impl HfModernBertConfig {
    fn into_candle(self) -> ModernBertConfig {
        let global = self
            .global_rope_theta
            .or_else(|| {
                self.rope_parameters
                    .as_ref()
                    .and_then(|r| r.full_attention.as_ref())
                    .map(|t| t.rope_theta)
            })
            .unwrap_or(160_000.0);
        let local = self
            .local_rope_theta
            .or_else(|| {
                self.rope_parameters
                    .as_ref()
                    .and_then(|r| r.sliding_attention.as_ref())
                    .map(|t| t.rope_theta)
            })
            .unwrap_or(10_000.0);
        let eps = if self.layer_norm_eps > 0.0 {
            self.layer_norm_eps
        } else {
            self.norm_eps
        };
        ModernBertConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            intermediate_size: self.intermediate_size,
            max_position_embeddings: self.max_position_embeddings,
            layer_norm_eps: eps,
            pad_token_id: self.pad_token_id,
            global_attn_every_n_layers: self.global_attn_every_n_layers,
            global_rope_theta: global,
            local_attention: self.local_attention,
            local_rope_theta: local,
            classifier_config: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    fn base_config() -> serde_json::Value {
        json!({
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 64,
            "max_position_embeddings": 512,
            "pad_token_id": 0,
            "global_attn_every_n_layers": 3,
            "local_attention": 128,
            "rope_parameters": {
                "full_attention": {"rope_theta": 250000.0},
                "sliding_attention": {"rope_theta": 20000.0}
            }
        })
    }

    #[test]
    fn reads_rope_parameters_and_default_eps() {
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&base_config()).unwrap(),
        )
        .unwrap();
        let cfg = load_modernbert_config(dir.path()).unwrap();
        assert_eq!(cfg.vocab_size, 64);
        assert_eq!(cfg.hidden_size, 32);
        assert!((cfg.global_rope_theta - 250_000.0).abs() < 1e-6);
        assert!((cfg.local_rope_theta - 20_000.0).abs() < 1e-6);
        assert!((cfg.layer_norm_eps - 1e-5).abs() < 1e-12);
    }

    #[test]
    fn legacy_top_level_rope_theta_wins_over_rope_parameters() {
        let dir = tempdir().unwrap();
        let mut cfg = base_config();
        cfg["global_rope_theta"] = json!(1234.0);
        cfg["local_rope_theta"] = json!(5678.0);
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&cfg).unwrap(),
        )
        .unwrap();
        let out = load_modernbert_config(dir.path()).unwrap();
        assert!((out.global_rope_theta - 1234.0).abs() < 1e-6);
        assert!((out.local_rope_theta - 5678.0).abs() < 1e-6);
    }

    #[test]
    fn missing_rope_falls_back_to_modernbert_defaults() {
        let dir = tempdir().unwrap();
        let mut cfg = base_config();
        cfg.as_object_mut().unwrap().remove("rope_parameters");
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&cfg).unwrap(),
        )
        .unwrap();
        let out = load_modernbert_config(dir.path()).unwrap();
        assert!((out.global_rope_theta - 160_000.0).abs() < 1e-6);
        assert!((out.local_rope_theta - 10_000.0).abs() < 1e-6);
    }

    #[test]
    fn partial_rope_parameters_falls_back_per_slot() {
        let dir = tempdir().unwrap();
        let mut cfg = base_config();
        cfg["rope_parameters"] = json!({"full_attention": {"rope_theta": 9000.0}});
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&cfg).unwrap(),
        )
        .unwrap();
        let out = load_modernbert_config(dir.path()).unwrap();
        assert!((out.global_rope_theta - 9000.0).abs() < 1e-6);
        assert!((out.local_rope_theta - 10_000.0).abs() < 1e-6);

        // sliding_attention present, full_attention missing
        let dir2 = tempdir().unwrap();
        let mut cfg = base_config();
        cfg["rope_parameters"] = json!({"sliding_attention": {"rope_theta": 777.0}});
        std::fs::write(
            dir2.path().join("config.json"),
            serde_json::to_string(&cfg).unwrap(),
        )
        .unwrap();
        let out = load_modernbert_config(dir2.path()).unwrap();
        assert!((out.global_rope_theta - 160_000.0).abs() < 1e-6);
        assert!((out.local_rope_theta - 777.0).abs() < 1e-6);
    }

    #[test]
    fn zero_layer_norm_eps_falls_back_to_norm_eps() {
        let dir = tempdir().unwrap();
        let mut cfg = base_config();
        cfg["layer_norm_eps"] = json!(0.0);
        cfg["norm_eps"] = json!(1e-12);
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&cfg).unwrap(),
        )
        .unwrap();
        let out = load_modernbert_config(dir.path()).unwrap();
        assert!((out.layer_norm_eps - 1e-12).abs() < 1e-18);
    }

    #[test]
    fn missing_or_broken_config_is_an_error() {
        let dir = tempdir().unwrap();
        let err = load_modernbert_config(dir.path()).unwrap_err();
        assert!(err.to_string().contains("config.json"));

        std::fs::write(dir.path().join("config.json"), "{not json").unwrap();
        let err = load_modernbert_config(dir.path()).unwrap_err();
        matches!(err, AfmError::Json(_));
    }

    #[test]
    fn encoder_prefix_is_remapped() {
        assert_eq!(
            remap_encoder_key("encoder.layers.0.attn.weight"),
            "model.layers.0.attn.weight"
        );
        assert_eq!(remap_encoder_key("head.layers.0.x"), "head.layers.0.x");
    }
}
