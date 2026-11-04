//! Encoder checkpoint model: pure HF config mapping + candle forward.
//!
//! `config` holds everything that can be validated without weights; `forward`
//! holds the tensor graph (excluded from the 100% line-coverage gate because it
//! needs the real ModernBERT checkpoint — see `requirements.md` §4).

pub mod config;
pub mod forward;

pub use config::{load_modernbert_config, remap_encoder_key};
pub use forward::{cpu, DecisionHead, DecisionModel, SharedDecisionModel};
