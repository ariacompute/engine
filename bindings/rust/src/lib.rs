//! Rust SDK: open AFM-D checkpoint and run System One.

use ariacompute_core::contract::Track;
use ariacompute_core::error::{AfmError, Result};
use ariacompute_core::packing::Record;
use ariacompute_core::systemone::{record_from_systemone_question, SystemOneRequest};
use ariacompute_dd::DecoderScorer;
use ariacompute_de::EncoderScorer;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

pub use ariacompute_core::config::{aria_home, load_config, save_config, AriaConfig};
pub use ariacompute_core::contract::Track as EngineTrack;

enum Inner {
    Encoder(Box<EncoderScorer>),
    Decoder(DecoderScorer),
}

pub struct Engine {
    inner: Inner,
    model_name: String,
}

impl Engine {
    pub fn open(checkpoint: impl AsRef<Path>, track: Track) -> Result<Self> {
        let ckpt = checkpoint.as_ref();
        let inner = match track {
            Track::Encoder => Inner::Encoder(Box::new(EncoderScorer::open(ckpt)?)),
            Track::Decoder => Inner::Decoder(DecoderScorer::open(Some(ckpt))?),
        };
        Ok(Self {
            inner,
            model_name: ckpt
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| track.as_str().into()),
        })
    }

    pub fn systemone(&self, request: &SystemOneRequest) -> Result<Value> {
        let mut answers = Map::new();
        for (qid, question) in &request.questions {
            let record = record_from_systemone_question(qid, &request.state, question)?;
            answers.insert(qid.clone(), self.decide_record(&record)?);
        }
        Ok(json!({
            "answers": answers,
            "model": self.model_name,
        }))
    }

    pub fn decide_record(&self, record: &Record) -> Result<Value> {
        match &self.inner {
            Inner::Encoder(s) => s.score_record(record),
            Inner::Decoder(s) => {
                let out = s.score_record(record)?;
                Ok(out.get("systemone").cloned().unwrap_or(out))
            }
        }
    }

    /// Encoder golden path without full weights.
    pub fn decide_with_logits(&self, record: &Record, logits: &[f32]) -> Result<Value> {
        match &self.inner {
            Inner::Encoder(s) => s.score_record_with_logits(record, logits),
            Inner::Decoder(_) => Err(AfmError::msg("decide_with_logits is encoder-only")),
        }
    }

    pub fn decide_with_semif_out(&self, record: &Record, out: &Value) -> Result<Value> {
        match &self.inner {
            Inner::Decoder(s) => {
                let scored = s.score_from_semif_out(record, out)?;
                Ok(scored.get("systemone").cloned().unwrap_or(scored))
            }
            Inner::Encoder(_) => Err(AfmError::msg("decide_with_semif_out is decoder-only")),
        }
    }
}

pub fn resolve_checkpoint(name: &str) -> PathBuf {
    ariacompute_core::config::resolve_checkpoint(name)
}
