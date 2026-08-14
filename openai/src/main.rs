use aria_hybrid::{CloudClient, ExecutionMode, ParetoMode, Router};
use aria_openai::{app, build_state};
use std::env;
use std::net::SocketAddr;
use std::process;

fn parse_mode(raw: &str) -> ParetoMode {
    match raw.to_ascii_lowercase().as_str() {
        "cost" => ParetoMode::Cost,
        "intelligence" | "intel" => ParetoMode::Intelligence,
        _ => ParetoMode::Balance,
    }
}

fn print_usage() {
    eprintln!(
        "\
aria-engine — Aria Compute OpenAI-compatible inference server

Usage:
  aria-engine serve --model <bundle_dir> [--bind host:port]
  aria-engine -h | --help | help

Options:
  serve                 Start HTTP server (OpenAI-compatible)
  --model <bundle_dir>  Aria quant bundle directory (required)
  --bind <host:port>    Listen address (default: 127.0.0.1:8080)
  -h, --help, help      Show this help and exit

Environment (hybrid cloud):
  ARIA_HYBRID_CLOUD_URL       Cloud base URL (default: https://gateway.ariacompute.com)
  ARIA_HYBRID_CLOUD_API_KEY   Bearer token for cloud calls
  ARIA_HYBRID_MODE            cost | balance | intelligence (default: balance)
  ARIA_HYBRID_EXECUTION       hybrid | device | cloud (default: hybrid)
"
    );
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = None;
    let mut bind = "127.0.0.1:8080".to_string();
    let mut args = env::args().skip(1).peekable();
    if args.peek().is_none() {
        print_usage();
        process::exit(2);
    }
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" | "help" => {
                print_usage();
                process::exit(0);
            }
            "serve" => {}
            "--model" => model = args.next(),
            "--bind" => bind = args.next().unwrap_or(bind),
            other => {
                eprintln!("unknown arg: {other}");
                print_usage();
                process::exit(2);
            }
        }
    }
    let model = match model {
        Some(m) => m,
        None => {
            eprintln!("error: --model <bundle_dir> is required");
            print_usage();
            process::exit(2);
        }
    };
    let cloud_base = env::var("ARIA_HYBRID_CLOUD_URL")
        .unwrap_or_else(|_| "https://gateway.ariacompute.com".into());
    let mode = env::var("ARIA_HYBRID_MODE")
        .ok()
        .map(|s| parse_mode(&s))
        .unwrap_or_default();
    let execution = match env::var("ARIA_HYBRID_EXECUTION") {
        Ok(s) => ExecutionMode::parse(&s)?,
        Err(_) => ExecutionMode::Hybrid,
    };
    let router = Router::new()?
        .with_mode(mode)
        .with_execution(execution);
    let state = build_state(&model, router, CloudClient::from_env(cloud_base))?;
    let app = app(state);
    let addr: SocketAddr = bind.parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("aria-openai listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
