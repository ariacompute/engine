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

Credentials (two sections — do not mix):
  [1/2] Local (router registration)  — router URL; API keys sk-aria_… only
  [2/2] OAuth (Aria Compute)         — serve_site + bfvk-… (stored only)

Usage:
  aria-engine setup [--status|--clear]
    Local flags:  --router URL --router-api-key sk-aria_…
    OAuth flags:  --serve-site com|cn --serve-api-key bfvk-…
  aria-engine download <model>
  aria-engine list
  aria-engine check [model]
  aria-engine clean [model]
  aria-engine upgrade [version]
  aria-engine serve <model> [--bind host:port] [--router URL] [--router-api-key sk-aria_…] [--compute auto|cpu|cuda] [--profile]
  aria-engine -h | --help | help
  aria-engine -v | --version | version

Cache:
  ~/.ariacompute/engine.yml
  ~/.ariacompute/models/<model>/
  ~/.ariacompute/lib/   (libaria-engine_ffi from upgrade)

setup                Sectioned Local vs OAuth prompts; hub/compute after
  --status           Show config status (grouped; keys redacted)
  --clear            Remove engine.yml (and leftover config.yml)
download <model>     Fetch from the regional public hub
list                 Scan local ~/.ariacompute/models
check [model]        Compare local bundle files (count, names, SHA-256) to regional hub
clean [model]        Remove one cached model or all
upgrade [version]    Replace this CLI + libaria-engine_ffi from GitHub/Gitee (via upgrade_url)
serve <model>        Start OpenAI-compatible HTTP server
  --bind             Listen address (default: 127.0.0.1:8080)
  --router           aria-router management URL (process override; does not write engine.yml)
  --router-api-key   Local sk-aria_… only for provider registration Bearer (process override)
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

fn take_flag(args: &mut Vec<String>, name: &str) -> Option<String> {
    if let Some(i) = args.iter().position(|a| a == name) {
        args.remove(i);
        if i < args.len() {
            return Some(args.remove(i));
        }
    }
    None
}

fn prompt_yn(label: &str, default_yes: bool) -> io::Result<bool> {
    let hint = if default_yes { "Y/n" } else { "y/N" };
    let raw = prompt(&format!("{label} [{hint}]: "))?;
    if raw.is_empty() {
        return Ok(default_yes);
    }
    Ok(matches!(raw.to_ascii_lowercase().as_str(), "y" | "yes"))
}

async fn cmd_setup(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = args.to_vec();
    if args.iter().any(|a| a == "--status") {
        let cfg = config::load_config()?;
        println!("Local (router registration):");
        println!(
            "  router: {}",
            if cfg.router.is_empty() {
                "(not set)"
            } else {
                &cfg.router
            }
        );
        println!(
            "  router_api_key: {}",
            redact_secret(&cfg.router_api_key)
        );
        println!("OAuth (Aria Compute):");
        println!(
            "  serve_site: {}",
            if cfg.serve_site.is_empty() {
                "(not set)"
            } else {
                &cfg.serve_site
            }
        );
        println!(
            "  serve_api_key: {}",
            redact_secret(&cfg.serve_api_key)
        );
        println!("Hub / compute:");
        println!(
            "  site_url: {}",
            if cfg.site_url.is_empty() {
                "(not set)"
            } else {
                &cfg.site_url
            }
        );
        println!(
            "  upgrade_url: {}",
            if cfg.upgrade_url.is_empty() {
                "(not set)"
            } else {
                &cfg.upgrade_url
            }
        );
        println!("  compute: {}", cfg.compute);
        println!("  hf_token: {}", redact_secret(&cfg.hf_token));
        println!(
            "  modelscope_api_token: {}",
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

    eprintln!("── [1/2] Local (router registration) ──────────────────────");
    eprintln!("  Register this engine on aria-router with a LOCAL key.");
    eprintln!("  Source: Router Dashboard → Keys (sk-aria_…).");
    eprintln!("  Do NOT paste OAuth keys (bfvk-) here.");
    eprintln!();

    let site_url = take_flag(&mut args, "--site-url").unwrap_or_else(|| {
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
    let upgrade_url = take_flag(&mut args, "--upgrade-url").unwrap_or_else(|| {
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

    let router = take_flag(&mut args, "--router").unwrap_or_else(|| {
        prompt(&format!(
            "router URL (mgmt, optional, default: {}): ",
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

    let router_api_key = take_flag(&mut args, "--router-api-key").unwrap_or_else(|| {
        prompt(&format!(
            "local router API key sk-aria_… (optional, default: {}): ",
            if existing.router_api_key.is_empty() {
                "(none)"
            } else {
                "(set)"
            }
        ))
        .unwrap_or_default()
    });
    let router_api_key = if router_api_key.is_empty() {
        existing.router_api_key.clone()
    } else {
        router_api_key
    };
    config::validate_router_api_key(&router_api_key).map_err(|e| e)?;

    eprintln!();
    eprintln!("── [2/2] OAuth (Aria Compute) ───────────────────────");
    eprintln!("  Optional. Same cloud key as Router Dashboard → Account.");
    eprintln!("  Prefix bfvk-… only. Not used for PUT /providers this release.");
    eprintln!();

    let flag_serve_site = take_flag(&mut args, "--serve-site");
    let flag_serve_key = take_flag(&mut args, "--serve-api-key");
    let configure_oauth = if flag_serve_site.is_some() || flag_serve_key.is_some() {
        true
    } else {
        prompt_yn("configure OAuth API key?", false)?
    };

    let (serve_site, serve_api_key) = if configure_oauth {
        let site_raw = flag_serve_site.unwrap_or_else(|| {
            let choice = prompt(&format!(
                "Serve site [1] https://ariacompute.com  [2] https://ariacompute.cn (default: {}): ",
                if existing.serve_site.is_empty() {
                    "1"
                } else {
                    &existing.serve_site
                }
            ))
            .unwrap_or_default();
            if choice.is_empty() {
                if existing.serve_site.is_empty() {
                    "1".into()
                } else {
                    existing.serve_site.clone()
                }
            } else {
                choice
            }
        });
        let serve_site = config::normalize_serve_site(&site_raw);
        let key_raw = flag_serve_key.unwrap_or_else(|| {
            prompt(&format!(
                "Serve API key (bfvk-…, default: {}): ",
                if existing.serve_api_key.is_empty() {
                    "(none)"
                } else {
                    "(set)"
                }
            ))
            .unwrap_or_default()
        });
        let serve_api_key = if key_raw.is_empty() {
            existing.serve_api_key.clone()
        } else {
            key_raw
        };
        config::validate_serve_api_key(&serve_api_key).map_err(|e| e)?;
        (serve_site, serve_api_key)
    } else {
        (existing.serve_site.clone(), existing.serve_api_key.clone())
    };

    eprintln!();
    eprintln!("── Hub / compute ──────────────────────────────────────────");
    let compute = take_flag(&mut args, "--compute").unwrap_or_else(|| {
        prompt_choice("compute", &["auto", "cpu", "cuda"], "auto").unwrap_or_else(|_| "auto".into())
    });
    let (hf_token, modelscope_api_token) = prompt_regional_hub_token(pair, &existing)?;

    let cfg = AriaConfig {
        router,
        router_api_key,
        serve_site,
        serve_api_key,
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
    let mut router_api_key_override = None;
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
            "--router-api-key" => {
                i += 1;
                let secret = args
                    .get(i)
                    .cloned()
                    .ok_or("--router-api-key requires a secret")?;
                config::validate_router_api_key(&secret).map_err(|e| e)?;
                router_api_key_override = Some(secret);
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
        "setup" => cmd_setup(&args[1..]).await,
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
