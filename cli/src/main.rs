use aria_cli::app::{self, Cli};
use clap::Parser;
use std::process;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    if let Err(err) = app::run(cli).await {
        eprintln!("error: {err:#}");
        process::exit(1);
    }
}
