use aria_hybrid::{CloudClient, ExecutionMode, ParetoMode, Router};
use aria_kernel::ComputePref;
use aria_openai::config::{self, AriaConfig};
use aria_openai::download;
use aria_openai::gateway_detect;
use aria_openai::upgrade;
use aria_openai::{app, build_state_with_hybrid_opts, HybridRoutingOpts};
use std::env;
use std::io::{self, BufRead, Write};
use std::net::SocketAddr;
use std::process;

/// Embedded at compile time; release builds set `ARIA_ENGINE_VERSION` from the git tag.
const ENGINE_VERSION: &str = env!("ARIA_ENGINE_VERSION");

fn parse_mode(raw: &str) -> Result<ParetoMode, String> {
    match raw.to_ascii_lowercase().as_str() {
        "cost" => Ok(ParetoMode::Cost),
        "balance" => Ok(ParetoMode::Balance),
        "intelligence" | "intel" => Ok(ParetoMode::Intelligence),
        other => Err(format!(
            "hybrid_mode must be cost|balance|intelligence, got {other:?}"
        )),
    }
}

fn print_usage() {
    eprintln!(
        "\
aria-engine — Aria Compute inference engine

Usage:
  aria-engine auth [--status|--clear]
  aria-engine download <model>
  aria-engine list
  aria-engine clean [model]
  aria-engine upgrade [version]
  aria-engine serve <model> [--bind host:port] [--hybrid-mode MODE] [--hybrid-execution MODE]
                         [--hybrid-semantic on|off] [--compute auto|cpu|cuda] [--profile]
  aria-engine -h | --help | help
  aria-engine -v | --version | version

Cache:
  ~/.ariacompute/config.yml
  ~/.ariacompute/models/<model>/
  ~/.ariacompute/lib/   (libaria_ffi from upgrade)

auth                 Prompt for API key + hybrid prefs; detect .com/.cn from key
  --status           Show config status (key redacted)
  --clear            Remove config.yml
download <model>     Probe dashboard / Hugging Face / ModelScope; fetch best source
list                 Query site catalog; mark each bundle downloaded / not downloaded
clean [model]        Remove one cached model or all
upgrade [version]    Replace this CLI + libaria_ffi from GitHub/Gitee (via upgrade_url)
serve <model>        Start OpenAI-compatible HTTP server
  --bind             Listen address (default: 127.0.0.1:8080)
  --hybrid-mode      cost | balance | intelligence (overrides config for this process)
  --hybrid-execution hybrid | device | cloud (overrides config for this process)
  --hybrid-semantic  on | off (semantic routing layer; overrides config for this process)
  --compute          auto | cpu | cuda (local GEMM; orthogonal to hybrid_execution)
  --profile          record load/generate timings; GET /v1/engine/profile
"
    );
}

fn print_version() {
    println!("aria-engine {ENGINE_VERSION}");
}

fn prompt(label: &str) -> io::Result<String> {
    eprint!("{label}");
    io::stderr().flush()?;
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line)?;
    Ok(line.trim().to_string())
}

fn prompt_choice(label: &str, allowed: &[&str], default: &str) -> io::Result<String> {
    let joined = allowed.join("|");
    loop {
        let raw = prompt(&format!("{label} [{joined}] (default: {default}): "))?;
        if raw.is_empty() {
            return Ok(default.to_string());
        }
        let lower = raw.to_ascii_lowercase();
        if allowed.iter().any(|a| *a == lower) {
            return Ok(lower);
        }
        eprintln!("invalid choice: {raw}");
    }
}

async fn cmd_auth(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.iter().any(|a| a == "--status") {
        let cfg = config::load_config()?;
        let key = if cfg.cloud_api_key.is_empty() {
            "(not set)".into()
        } else if cfg.cloud_api_key.len() <= 8 {
            "********".into()
        } else {
            format!(
                "{}…{}",
                &cfg.cloud_api_key[..4],
                &cfg.cloud_api_key[cfg.cloud_api_key.len() - 4..]
            )
        };
        println!("cloud_api_key: {key}");
        println!(
            "cloud_url: {}",
            if cfg.cloud_url.is_empty() {
                "(not set)"
            } else {
                &cfg.cloud_url
            }
        );
        println!(
            "site_url: {}",
            if cfg.site_url.is_empty() {
                "(not set)"
            } else {
                &cfg.site_url
            }
        );
        println!(
            "upgrade_url: {}",
            if cfg.upgrade_url.is_empty() {
                "(not set)"
            } else {
                &cfg.upgrade_url
            }
        );
        println!("hybrid_mode: {}", cfg.hybrid_mode);
        println!("hybrid_execution: {}", cfg.hybrid_execution);
        println!(
            "hybrid_semantic: {} (timeout={}ms cache={})",
            cfg.hybrid_semantic, cfg.hybrid_semantic_timeout_ms, cfg.hybrid_semantic_cache_size
        );
        println!("compute: {}", cfg.compute);
        println!("config: {}", config::config_path()?.display());
        println!("lib: {}", config::lib_dir()?.display());
        return Ok(());
    }
    if args.iter().any(|a| a == "--clear") {
        config::clear_config()?;
        println!("cleared {}", config::config_path()?.display());
        return Ok(());
    }

    let api_key = prompt("API key (sk-… / bfvk-…): ")?;
    if api_key.is_empty() {
        return Err("API key required".into());
    }
    eprintln!("detecting gateway / site from API key…");
    let pair = gateway_detect::detect_gateway_and_site(&api_key).await;
    eprintln!(
        "using cloud_url={} site_url={} upgrade_url={}",
        pair.cloud_url,
        pair.site_url,
        pair.upgrade_url()
    );
    let hybrid_mode = prompt_choice(
        "hybrid_mode",
        &["cost", "balance", "intelligence"],
        "balance",
    )?;
    let hybrid_execution =
        prompt_choice("hybrid_execution", &["hybrid", "device", "cloud"], "hybrid")?;
    let compute = prompt_choice("compute", &["auto", "cpu", "cuda"], "auto")?;

    let cfg = AriaConfig {
        cloud_api_key: api_key,
        cloud_url: pair.cloud_url.to_string(),
        site_url: pair.site_url.to_string(),
        upgrade_url: pair.upgrade_url().to_string(),
        hybrid_mode,
        hybrid_execution,
        compute,
        ..AriaConfig::default()
    };
    config::save_config(&cfg)?;
    println!("wrote {}", config::config_path()?.display());
    Ok(())
}

async fn load_config_reconciled() -> Result<AriaConfig, Box<dyn std::error::Error>> {
    let mut cfg = config::load_config()?;
    if gateway_detect::reconcile_config_urls(&mut cfg).await {
        config::save_config(&cfg)?;
        eprintln!(
            "updated region: cloud_url={} site_url={}",
            cfg.cloud_url, cfg.site_url
        );
    }
    Ok(cfg)
}

async fn cmd_download(model: &str) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = load_config_reconciled().await?;
    let path = download::download_model(model, &cfg).await?;
    println!("{}", path.display());
    Ok(())
}

async fn cmd_list() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = load_config_reconciled().await?;
    let models = download::list_models_with_catalog(&cfg).await?;
    if models.is_empty() {
        println!("(no models in catalog)");
        return Ok(());
    }
    let width = models.iter().map(|m| m.name.len()).max().unwrap_or(0);
    for m in &models {
        println!("{:<width$}  {}", m.name, m.status, width = width);
    }
    Ok(())
}

fn cmd_clean(model: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    download::clean_models(model)?;
    match model {
        Some(m) => println!("cleaned {m}"),
        None => println!("cleaned all models under {}", config::models_dir()?.display()),
    }
    Ok(())
}

async fn cmd_serve(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut model = None;
    let mut bind = "127.0.0.1:8080".to_string();
    let mut mode_override = None;
    let mut exec_override = None;
    let mut semantic_override = None;
    let mut compute_override = None;
    let mut profile = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--bind" => {
                i += 1;
                bind = args
                    .get(i)
                    .cloned()
                    .ok_or("--bind requires host:port")?;
            }
            "--hybrid-mode" => {
                i += 1;
                mode_override = Some(
                    args.get(i)
                        .cloned()
                        .ok_or("--hybrid-mode requires a value")?,
                );
            }
            "--hybrid-execution" => {
                i += 1;
                exec_override = Some(
                    args.get(i)
                        .cloned()
                        .ok_or("--hybrid-execution requires a value")?,
                );
            }
            "--hybrid-semantic" => {
                i += 1;
                semantic_override = Some(
                    args.get(i)
                        .cloned()
                        .ok_or("--hybrid-semantic requires on|off")?,
                );
            }
            "--compute" => {
                i += 1;
                compute_override = Some(
                    args.get(i)
                        .cloned()
                        .ok_or("--compute requires auto|cpu|cuda")?,
                );
            }
            "--profile" => {
                profile = true;
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown flag: {other}").into());
            }
            other => {
                if model.is_some() {
                    return Err(format!("unexpected argument: {other}").into());
                }
                model = Some(other.to_string());
            }
        }
        i += 1;
    }
    let model = model.ok_or("serve requires <model>")?;
    let model_path = config::resolve_model_path(&model)?;

    let cfg = load_config_reconciled().await.unwrap_or_default();
    let mode = parse_mode(
        mode_override
            .as_deref()
            .unwrap_or(cfg.hybrid_mode.as_str()),
    )?;
    let execution = ExecutionMode::parse(
        exec_override
            .as_deref()
            .unwrap_or(cfg.hybrid_execution.as_str()),
    )?;
    let compute = ComputePref::parse(
        compute_override
            .as_deref()
            .unwrap_or(cfg.compute.as_str()),
    )?;
    let semantic_enabled = match semantic_override.as_deref() {
        Some("on") => true,
        Some("off") => false,
        Some(other) => {
            return Err(format!("--hybrid-semantic must be on|off, got {other:?}").into());
        }
        None => cfg.hybrid_semantic,
    };
    let cloud_url = if cfg.cloud_url.is_empty() {
        "https://gateway.ariacompute.com".to_string()
    } else {
        cfg.cloud_url.clone()
    };
    let router = Router::new()?
        .with_mode(mode)
        .with_execution(execution);
    let state = build_state_with_hybrid_opts(
        model_path.to_str().ok_or("invalid model path")?,
        router,
        CloudClient::new(cloud_url, cfg.cloud_api_key.clone()),
        compute,
        profile,
        HybridRoutingOpts {
            semantic_enabled,
            semantic_timeout_ms: cfg.hybrid_semantic_timeout_ms,
            semantic_cache_size: cfg.hybrid_semantic_cache_size,
        },
    )?;
    let compute_label = state
        .session
        .lock()
        .map(|s| s.compute_label().to_string())
        .unwrap_or_else(|_| "unknown".into());
    let app = app(state);
    let addr: SocketAddr = bind.parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!(
        "aria-openai listening on http://{addr} (model={} execution={:?} mode={:?} semantic={} compute={})",
        model_path.display(),
        execution,
        mode,
        semantic_enabled,
        compute_label
    );
    axum::serve(listener, app).await?;
    Ok(())
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        print_usage();
        process::exit(2);
    }
    let result = match args[0].as_str() {
        "-h" | "--help" | "help" => {
            print_usage();
            Ok(())
        }
        "-v" | "--version" | "version" => {
            print_version();
            Ok(())
        }
        "auth" => cmd_auth(&args[1..]).await,
        "download" => {
            let model = args.get(1).map(|s| s.as_str()).unwrap_or("");
            if model.is_empty() {
                Err("download requires <model>".into())
            } else {
                cmd_download(model).await
            }
        }
        "list" => cmd_list().await,
        "clean" => cmd_clean(args.get(1).map(|s| s.as_str())),
        "upgrade" => {
            let version = args.get(1).map(|s| s.as_str());
            upgrade::run(version, ENGINE_VERSION)
                .await
                .map_err(|e| e.into())
        }
        "serve" => cmd_serve(&args[1..]).await,
        other => Err(format!("unknown command: {other}").into()),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        process::exit(1);
    }
}
