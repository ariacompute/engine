use aria_kernel::ComputePref;
use aria_openai::check;
use aria_openai::config::{self, AriaConfig};
use aria_openai::download;
use aria_openai::gateway_detect;
use aria_openai::upgrade;
use aria_openai::{app, build_state_with_opts, register_with_router};
use std::env;
use std::io::{self, BufRead, Write};
use std::net::SocketAddr;
use std::process;

/// Embedded at compile time; release builds set `ARIA_ENGINE_VERSION` from the git tag.
const ENGINE_VERSION: &str = env!("ARIA_ENGINE_VERSION");

fn print_usage() {
    eprintln!(
        "\
aria-engine — Aria Compute inference engine

Usage:
  aria-engine auth [--status|--clear]
  aria-engine download <model>
  aria-engine list
  aria-engine check [model]
  aria-engine clean [model]
  aria-engine upgrade [version]
  aria-engine serve <model> [--bind host:port] [--router URL] [--compute auto|cpu|cuda] [--profile]
  aria-engine -h | --help | help
  aria-engine -v | --version | version

Cache:
  ~/.ariacompute/config.yml
  ~/.ariacompute/models/<model>/
  ~/.ariacompute/lib/   (libaria_ffi from upgrade)

auth                 Prompt for compute, hub token, optional router URL; no API key required
  --status           Show config status (keys redacted)
  --clear            Remove config.yml
download <model>     Fetch from the regional public hub
list                 Scan local ~/.ariacompute/models
check [model]        Compare local bundle files (count, names, SHA-256) to regional hub
clean [model]        Remove one cached model or all
upgrade [version]    Replace this CLI + libaria_ffi from GitHub/Gitee (via upgrade_url)
serve <model>        Start OpenAI-compatible HTTP server
  --bind             Listen address (default: 127.0.0.1:8080)
  --router           aria-router management URL (process override; does not write config.yml)
  --compute          auto | cpu | cuda (local GEMM)
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

/// `.com` → Hugging Face token; `.cn` → ModelScope token. The other field is left as-is.
fn prompt_regional_hub_token(
    pair: gateway_detect::GatewayPair,
    existing: &AriaConfig,
) -> io::Result<(String, String)> {
    let cn = pair == gateway_detect::GatewayPair::CN;
    let entered = if cn {
        prompt("modelscope_api_token (ModelScope, optional): ")?
    } else {
        prompt("hf_token (Hugging Face, optional): ")?
    };
    Ok(config::apply_hub_token_input(existing, cn, &entered))
}

fn redact_secret(value: &str) -> String {
    if value.is_empty() {
        "(not set)".into()
    } else if value.len() <= 8 {
        "********".into()
    } else {
        format!("{}…{}", &value[..4], &value[value.len() - 4..])
    }
}

async fn cmd_auth(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.iter().any(|a| a == "--status") {
        let cfg = config::load_config()?;
        println!(
            "router: {}",
            if cfg.router.is_empty() {
                "(not set)"
            } else {
                &cfg.router
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
        println!("compute: {}", cfg.compute);
        println!("hf_token: {}", redact_secret(&cfg.hf_token));
        println!(
            "modelscope_api_token: {}",
            redact_secret(&cfg.modelscope_api_token)
        );
        println!("config: {}", config::config_path()?.display());
        println!("lib: {}", config::lib_dir()?.display());
        return Ok(());
    }
    if args.iter().any(|a| a == "--clear") {
        config::clear_config()?;
        println!("cleared {}", config::config_path()?.display());
        return Ok(());
    }

    let existing = config::load_config().unwrap_or_default();
    let site_url = prompt(&format!(
        "site_url (default: {}): ",
        if existing.site_url.is_empty() {
            "https://ariacompute.com"
        } else {
            &existing.site_url
        }
    ))?;
    let site_url = if site_url.is_empty() {
        if existing.site_url.is_empty() {
            "https://ariacompute.com".into()
        } else {
            existing.site_url.clone()
        }
    } else {
        site_url
    };
    let pair = gateway_detect::GatewayPair::from_url(&site_url)
        .unwrap_or(gateway_detect::GatewayPair::INTL);
    let upgrade_url = prompt(&format!(
        "upgrade_url (default: {}): ",
        if existing.upgrade_url.is_empty() {
            pair.upgrade_url()
        } else {
            &existing.upgrade_url
        }
    ))?;
    let upgrade_url = if upgrade_url.is_empty() {
        if existing.upgrade_url.is_empty() {
            pair.upgrade_url().to_string()
        } else {
            existing.upgrade_url.clone()
        }
    } else {
        upgrade_url
    };
    let router = prompt(&format!(
        "router URL (optional, default: {}): ",
        if existing.router.is_empty() {
            "(none)"
        } else {
            &existing.router
        }
    ))?;
    let router = if router.is_empty() {
        existing.router.clone()
    } else {
        router
    };
    let compute = prompt_choice("compute", &["auto", "cpu", "cuda"], "auto")?;
    let (hf_token, modelscope_api_token) = prompt_regional_hub_token(pair, &existing)?;

    let cfg = AriaConfig {
        router,
        site_url,
        upgrade_url,
        compute,
        hf_token,
        modelscope_api_token,
    };
    config::save_config(&cfg)?;
    println!("wrote {}", config::config_path()?.display());
    Ok(())
}

async fn load_config_reconciled() -> Result<AriaConfig, Box<dyn std::error::Error>> {
    Ok(config::load_config()?)
}

async fn cmd_download(model: &str) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = load_config_reconciled().await?;
    let path = download::download_model(model, &cfg).await?;
    println!("{}", path.display());
    Ok(())
}

async fn cmd_list() -> Result<(), Box<dyn std::error::Error>> {
    let models = download::list_models()?;
    if models.is_empty() {
        println!("(no local models under ~/.ariacompute/models)");
        return Ok(());
    }
    for m in &models {
        println!("{m}");
    }
    Ok(())
}

async fn cmd_check(model: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = load_config_reconciled().await?;
    let ok = match model {
        Some(m) => check::check_model(m, &cfg).await?,
        None => check::check_all(&cfg).await?,
    };
    if ok {
        Ok(())
    } else {
        Err("check failed".into())
    }
}

fn cmd_clean(model: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    download::clean_models(model)?;
    match model {
        Some(m) => println!("cleaned {m}"),
        None => println!(
            "cleaned all models under {}",
            config::models_dir()?.display()
        ),
    }
    Ok(())
}

async fn cmd_serve(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut model = None;
    let mut bind = "127.0.0.1:8080".to_string();
    let mut router_override = None;
    let mut compute_override = None;
    let mut profile = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--bind" => {
                i += 1;
                bind = args.get(i).cloned().ok_or("--bind requires host:port")?;
            }
            "--router" => {
                i += 1;
                router_override = Some(args.get(i).cloned().ok_or("--router requires a URL")?);
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
    let compute = ComputePref::parse(compute_override.as_deref().unwrap_or(cfg.compute.as_str()))?;
    let state = build_state_with_opts(
        model_path.to_str().ok_or("invalid model path")?,
        compute,
        profile,
    )?;
    let model_id = state.model_id.clone();
    let compute_label = state
        .session
        .lock()
        .map(|s| s.compute_label().to_string())
        .unwrap_or_else(|_| "unknown".into());
    let app = app(state);
    let addr: SocketAddr = bind.parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let router_url = router_override
        .filter(|s| !s.is_empty())
        .or_else(|| {
            if cfg.router.is_empty() {
                None
            } else {
                Some(cfg.router.clone())
            }
        });
    if let Some(url) = router_url {
        eprintln!("registering provider {model_id} at {url}");
        register_with_router(&url, &bind, &model_id)
            .await
            .map_err(|e| format!("{e}"))?;
    }
    eprintln!(
        "aria-openai listening on http://{addr} (model={} compute={})",
        model_path.display(),
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
        "check" => cmd_check(args.get(1).map(|s| s.as_str())).await,
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
