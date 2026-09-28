//! AFM-D engine CLI library (setup + upgrade + HTTP serve helpers).

pub mod download;
pub mod serve;
pub mod setup;
pub mod upgrade;

pub use serve::{build_router, ServeOpts};
pub use setup::cmd_setup;
