//! MiniCPM5-2B (Llama) candle load + last-token letter logits.
//!
//! Excluded from the 100% line-coverage gate (`dd/src/forward.rs`): every branch
//! here needs the real MiniCPM5-2B checkpoint. Weight discovery/loading and LoRA
//! merge live in `dd/src/weights.rs`; tokenizer + letter selection live in
//! `dd/src/letters.rs`; both are fully covered. See `requirements.md` §4.

use std::fs;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
use tokenizers::Tokenizer;

use ariacompute_core::contract::MAX_TOKENS_DD;
use ariacompute_core::error::{AfmError, Result};

use crate::checkpoint::DecoderCheckpoint;
use crate::letters::{
    encode_prompt, last_token_logits, letter_probs, letter_token_ids, load_tokenizer,
};
use crate::prompt::prompt_text;
use crate::score::LetterScorer;
use crate::weights::{
    candle_err, collect_weight_files, load_weight_map, locate_base_dir, maybe_tie_lm_head,
    merge_lora_into, missing_weights,
};

pub struct MiniCpmForward {
    llama: Llama,
    cache: Cache,
    device: Device,
    tokenizer: Tokenizer,
    letter_ids: [u32; 16],
    bos_id: Option<u32>,
}

impl MiniCpmForward {
    pub fn load(ckpt: &DecoderCheckpoint) -> Result<Self> {
        let device = Device::Cpu;
        let base = locate_base_dir(ckpt)?;
        let tokenizer = load_tokenizer(ckpt, &base)?;
        let letter_ids = letter_token_ids(&tokenizer)?;
        let bos_id = tokenizer.token_to_id("<s>");

        let weight_files = collect_weight_files(&base)?;
        if weight_files.is_empty() {
            return Err(missing_weights(ckpt));
        }
        let mut tensors = load_weight_map(&weight_files, &device)?;
        maybe_tie_lm_head(&mut tensors)?;
        if let Some(adapter_dir) = &ckpt.adapter_dir {
            merge_lora_into(&mut tensors, adapter_dir, &device)?;
        }

        let cfg_text = fs::read_to_string(base.join("config.json")).map_err(|e| {
            AfmError::msg(format!("read {}: {e}", base.join("config.json").display()))
        })?;
        let llama_cfg: LlamaConfig = serde_json::from_str(&cfg_text)?;
        let mut config = llama_cfg.into_config(false);
        // MiniCPM5 lists 131072 positions; Cache preallocates that many — cap to the SemIf budget.
        if config.max_position_embeddings > MAX_TOKENS_DD {
            config.max_position_embeddings = MAX_TOKENS_DD;
        }

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let llama = Llama::load(vb, &config)
            .map_err(|e| AfmError::msg(format!("MiniCPM Llama::load: {e}")))?;
        let cache = Cache::new(false, DType::F32, &config, &device)
            .map_err(|e| AfmError::msg(format!("MiniCPM Cache: {e}")))?;
        Ok(Self {
            llama,
            cache,
            device,
            tokenizer,
            letter_ids,
            bos_id,
        })
    }
}

impl LetterScorer for MiniCpmForward {
    fn letter_logits(&mut self, row: &serde_json::Value) -> Result<(Vec<String>, Vec<f32>)> {
        let prompt = prompt_text(row)?;
        let ids = encode_prompt(&self.tokenizer, &prompt, self.bos_id)?;
        if ids.len() > MAX_TOKENS_DD {
            return Err(AfmError::msg(format!(
                "decoder prompt is {} tokens (max {MAX_TOKENS_DD})",
                ids.len()
            )));
        }
        let seq = ids.len();
        let x = Tensor::new(ids.as_slice(), &self.device)
            .map_err(candle_err)?
            .reshape((1, seq))
            .map_err(candle_err)?;
        let logits = self
            .llama
            .forward(&x, 0, &mut self.cache)
            .map_err(|e| AfmError::msg(format!("MiniCPM forward: {e}")))?;
        let last = last_token_logits(&logits)?;
        let options = row
            .get("options")
            .and_then(|v| v.as_array())
            .ok_or_else(|| AfmError::msg("semif row missing options"))?;
        letter_probs(&last, &self.letter_ids, options)
    }
}
