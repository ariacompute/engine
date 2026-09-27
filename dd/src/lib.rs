//! AFM-D Decoder (`afm_dd`): SemIf row build + first-token scoring map.

pub mod checkpoint;
pub mod score;
pub mod semif;

pub use checkpoint::DecoderCheckpoint;
pub use score::{probs_from_semif_out, DecoderScorer};
pub use semif::{record_to_semif_row, LETTERS};
