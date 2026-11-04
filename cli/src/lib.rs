//! AFM-D engine CLI library (setup + upgrade + HTTP serve helpers).

pub mod app;
pub mod download;
pub mod serve;
pub mod setup;
pub mod upgrade;

#[cfg(test)]
pub(crate) mod test_env;

pub use app::{Cli, Command};
pub use serve::{build_router, ServeOpts};
pub use setup::cmd_setup;
