//! AFM-D Encoder (`afm_de`): Laya-layout checkpoint load, packing, typed decode.

pub mod checkpoint;
pub mod model;
pub mod score;
pub mod shortlist;
pub mod tokenizer;

pub use checkpoint::{load_agent_config, AgentConfig, EncoderCheckpoint};
pub use model::DecisionModel;
pub use score::{score_record_logits, EncoderScorer};
pub use shortlist::maybe_shortlist;
