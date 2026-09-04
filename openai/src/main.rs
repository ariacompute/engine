use aria_kernel::ComputePref;
use aria_openai::check;
use aria_openai::config::{self, AriaConfig};
use aria_openai::download;
use aria_openai::gateway_detect;
use aria_openai::upgrade;
use aria_openai::{app, build_state_with_opts, register_with_router};
use clap::{ArgAction, Parser, Subcommand};
use std::io::{self, BufRead, Write};
use std::net::SocketAddr;
use std::process;

/// Embedded at compile time; release builds set `ARIA_ENGINE_VERSION` from the git tag.
const ENGINE_VERSION: &str = env!("ARIA_ENGINE_VERSION");

#[derive(Parser)]
#[command(
    name = "aria-engine",
    about = "Aria Compute inference engine CLI",
    version = ENGINE_VERSION,
    arg_required_else_help = true,
    disable_version_flag = true
)]
struct Cli {
    /// Print version
    #[arg(short = 'v', long = "version", action = ArgAction::Version)]
    _version: (),
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Write engine.yml (hub, compute, optional router)
    Setup {
        /// Show config status (keys redacted)
        #[arg(long)]
        status: bool,
        /// Remove engine.yml (and leftover config.yml)
        #[arg(long)]
        clear: bool,
        /// Site URL (.com or .cn)
        #[arg(long)]
        site_url: Option<String>,
        /// Releases org root (GitHub/Gitee)
        #[arg(long)]
        upgrade_url: Option<String>,
        /// aria-router management URL
        #[arg(long)]
        router: Option<String>,
        /// aria-router API key (sk-aria_… or serve sk-bf-…)
        #[arg(long)]
        router_api_key: Option<String>,
        /// Local GEMM preference: auto | cpu | cuda
        #[arg(long)]
        compute: Option<String>,
    },
    /// Fetch a model from the regional public hub
    Download {
        model: String,
    },
    /// Scan local ~/.ariacompute/models
    List,
    /// Compare local bundle files to the regional hub
    Check {
        model: Option<String>,
    },
    /// Remove one cached model or all
    Clean {
        model: Option<String>,
    },
    /// Replace this CLI + libaria-engine_ffi from Releases
    Upgrade {
        version: Option<String>,
    },
    /// Start OpenAI-compatible HTTP server
    Serve {
        model: String,
        /// Listen address
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: String,
        /// aria-router management URL (process override)
        #[arg(long)]
        router: Option<String>,
        /// aria-router API key (process override)
        #[arg(long)]
        router_api_key: Option<String>,
        /// auto | cpu | cuda (local GEMM)
        #[arg(long)]
        compute: Option<String>,
        /// Record load/generate timings; GET /v1/engine/profile
        #[arg(long)]
        profile: bool,
    },
    /// Print version
    Version,
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

async fn cmd_setup(
    status: bool,
    clear: bool,
    site_url_flag: Option<String>,
    upgrade_url_flag: Option<String>,
    router_flag: Option<String>,
    router_api_key_flag: Option<String>,
    compute_flag: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    if status {
        let cfg = config::load_config()?;
        println!(
            "router: {}",
            if cfg.router.is_empty() {
                "(not set)"
            } else {
                &cfg.router
            }
        );
        println!("router_api_key: {}", redact_secret(&cfg.router_api_key));
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
    if clear {
        config::clear_config()?;
        println!("cleared {}", config::config_path()?.display());
        return Ok(());
    }

    let existing = config::load_config().unwrap_or_default();

    let site_url = site_url_flag.unwrap_or_else(|| {
        prompt(&format!(
            "site_url (default: {}): ",
            if existing.site_url.is_empty() {
                "https://ariacompute.com"
            } else {
                &existing.site_url
            }
        ))
        .unwrap_or_default()
    });
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
    let upgrade_url = upgrade_url_flag.unwrap_or_else(|| {
        prompt(&format!(
            "upgrade_url (default: {}): ",
            if existing.upgrade_url.is_empty() {
                pair.upgrade_url()
            } else {
                &existing.upgrade_url
            }
        ))
        .unwrap_or_default()
    });
    let upgrade_url = if upgrade_url.is_empty() {
        if existing.upgrade_url.is_empty() {
            pair.upgrade_url().to_string()
        } else {
            existing.upgrade_url.clone()
        }
    } else {
        upgrade_url
    };

    let router = router_flag.unwrap_or_else(|| {
        prompt(&format!(
            "router URL (optional, default: {}): ",
            if existing.router.is_empty() {
                "(none)"
            } else {
                &existing.router
            }
        ))
        .unwrap_or_default()
    });
    let router = if router.is_empty() {
        existing.router.clone()
    } else {
        router
    };

    let key_entered = router_api_key_flag.unwrap_or_else(|| {
        prompt(&format!(
            "router API key (optional, default: {}): ",
            if existing.router_api_key.is_empty() {
                "(none)"
            } else {
                "(set)"
            }
        ))
        .unwrap_or_default()
    });
    let router_api_key = if key_entered.is_empty() {
        existing.router_api_key.clone()
    } else {
        key_entered
    };
    config::validate_router_api_key(&router_api_key)?;

    let compute = compute_flag.unwrap_or_else(|| {
        prompt_choice("compute", &["auto", "cpu", "cuda"], "auto").unwrap_or_else(|_| "auto".into())
    });
    let (hf_token, modelscope_api_token) = prompt_regional_hub_token(pair, &existing)?;

    let cfg = AriaConfig {
        router,
        router_api_key,
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

async fn cmd_serve(
    model: String,
    bind: String,
    router_override: Option<String>,
    router_api_key_override: Option<String>,
    compute_override: Option<String>,
    profile: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = config::resolve_model_path(&model)?;

    let cfg = load_config_reconciled().await.unwrap_or_default();
    if let Some(ref secret) = router_api_key_override {
        config::validate_router_api_key(secret)?;
    }
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
    let router_api_key = router_api_key_override
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| cfg.router_api_key.clone());
    if let Some(url) = router_url {
        eprintln!("registering provider {model_id} at {url}");
        register_with_router(&url, &bind, &model_id, &router_api_key)
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
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Setup {
            status,
            clear,
            site_url,
            upgrade_url,
            router,
            router_api_key,
            compute,
        } => {
            cmd_setup(
                status,
                clear,
                site_url,
                upgrade_url,
                router,
                router_api_key,
                compute,
            )
            .await
        }
        Command::Download { model } => cmd_download(&model).await,
        Command::List => cmd_list().await,
        Command::Check { model } => cmd_check(model.as_deref()).await,
        Command::Clean { model } => cmd_clean(model.as_deref()),
        Command::Upgrade { version } => upgrade::run(version.as_deref(), ENGINE_VERSION)
            .await
            .map_err(|e| e.into()),
        Command::Serve {
            model,
            bind,
            router,
            router_api_key,
            compute,
            profile,
        } => cmd_serve(model, bind, router, router_api_key, compute, profile).await,
        Command::Version => {
            println!("aria-engine {ENGINE_VERSION}");
            Ok(())
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        process::exit(1);
    }
}
