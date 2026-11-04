//! AFM-D Decoder (`afm_dd`): SemIf row build + first-token scoring map.

pub mod checkpoint;
pub mod forward;
pub mod letters;
pub mod prompt;
pub mod pyjson;
pub mod score;
pub mod semif;
pub mod weights;

pub use checkpoint::DecoderCheckpoint;
pub use score::{probs_from_semif_out, DecoderScorer, LetterScorer};
pub use semif::{record_to_semif_row, LETTERS};
pub use weights::{locate_base_dir, missing_weights};
