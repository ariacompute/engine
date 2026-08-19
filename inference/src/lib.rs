//! Aria inference: bundle loader, family registry, Session.

mod bundle;
mod chat;
mod family;
pub mod fixture;
pub mod multimodal;
mod pack;
mod session;
mod tensor_names;
mod tokenizer;

pub use aria_kernel::EngineError;
pub use bundle::{dequantize, load_bundle, Bundle, ModelConfig, QuantTensor, TensorData};
pub use family::{
    arch_class_representatives, family_phase, graph_hook, infer_family_path, lookup_family,
    require_runnable, require_stage_a, require_stage_b, ArchClass, Family, FamilyPhase,
    FAMILY_REGISTRY,
};
pub use multimodal::{action_head, asr_transcribe_pcm16le, rag_pack_context, vision_encode};
pub use session::{
    confidence_from_logits, GenerateOpts, Generation, Session, SessionBuilder,
};
pub use chat::{apply_chat_template, ChatTurn};
pub use tokenizer::{decode_placeholders, encode_naive, BundleTokenizer};
