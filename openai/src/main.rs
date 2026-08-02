use aria_hybrid::{CloudClient, Router};
use aria_openai::{app, build_state};
use std::env;
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = None;
    let mut bind = "127.0.0.1:8080".to_string();
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "serve" => {}
            "--model" => model = args.next(),
            "--bind" => bind = args.next().unwrap_or(bind),
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }
    let model = model.ok_or("usage: aria-openai serve --model <bundle_dir> [--bind host:port]")?;
    let cloud_base =
        env::var("ARIA_HYBRID_CLOUD_URL").unwrap_or_else(|_| "http://127.0.0.1:9".into());
    let threshold: f32 = env::var("ARIA_HYBRID_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let mut router = Router::new(threshold)?;
    if env::var("ARIA_ON_DEVICE_ONLY").ok().as_deref() == Some("1") {
        router.on_device_only = true;
    }
    let state = build_state(&model, router, CloudClient::from_env(cloud_base))?;
    let app = app(state);
    let addr: SocketAddr = bind.parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("aria-openai listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
