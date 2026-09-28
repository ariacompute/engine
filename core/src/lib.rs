//! AFM-D shared types, System One contract, config, and packing helpers.

pub mod config;
pub mod contract;
pub mod error;
pub mod gateway;
pub mod packing;
pub mod systemone;
pub mod typed;

pub use config::{
    apply_hub_token_input, aria_home, hub_token_for_site, load_config, parse_compute, save_config,
    AriaConfig,
};
pub use contract::{Track, HEAD_MAX_LEN, MAX_LEN, OPTION_DESC_MAX};
pub use error::{AfmError, Result};
pub use gateway::{preferred_hub, GatewayPair, PublicHub};
pub use systemone::{
    answer_from_probs, record_from_systemone_question, SystemOneAnswer, SystemOneRequest,
    SystemOneResponse,
};
pub use typed::{softmax, typed_answer, Task};
