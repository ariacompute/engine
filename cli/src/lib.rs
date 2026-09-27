//! AFM-D engine CLI library (upgrade + HTTP serve helpers).

pub mod download;
pub mod serve;
pub mod upgrade;

pub use serve::{build_router, ServeOpts};
