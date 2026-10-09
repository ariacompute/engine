//! `aria-engine setup` — five-field engine.yml (no router).

use ariacompute_core::config::{self, apply_hub_token_input, parse_compute, AriaConfig};
use ariacompute_core::gateway::GatewayPair;
use std::io::{self, BufRead, Write};

fn prompt(label: &str) -> io::Result<String> {
    eprint!("{label}");
    io::stderr().flush()?;
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line)?;
    Ok(line.trim().to_string())
}

/// Interactive secret: echo `*` per character (empty allowed).
fn prompt_secret(label: &str) -> io::Result<String> {
    let prompt = label.trim_end_matches([' ', ':']);
    dialoguer::Password::new()
        .with_prompt(prompt)
        .allow_empty_password(true)
        .interact()
        .map_err(io::Error::other)
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

fn redact_secret(value: &str) -> String {
    if value.is_empty() {
        "(not set)".into()
    } else if value.len() <= 8 {
        "********".into()
    } else {
        format!("{}…{}", &value[..4], &value[value.len() - 4..])
    }
}

fn prompt_regional_hub_token(
    pair: GatewayPair,
    existing: &AriaConfig,
) -> io::Result<(String, String)> {
    let cn = pair.is_cn();
    let entered = if cn {
        prompt_secret("modelscope_api_token (ModelScope, optional): ")?
    } else {
        prompt_secret("hf_token (Hugging Face, optional): ")?
    };
    Ok(apply_hub_token_input(existing, cn, &entered))
}

/// Run `aria-engine setup` (status / clear / interactive or flags).
pub fn cmd_setup(
    status: bool,
    clear: bool,
    site_url_flag: Option<String>,
    upgrade_url_flag: Option<String>,
    compute_flag: Option<String>,
) -> anyhow::Result<()> {
    if clear {
        config::clear_config()?;
        println!("cleared {}", config::engine_yml_path()?.display());
        return Ok(());
    }
    if status {
        let cfg = config::load_config()?;
        println!(
            "site_url: {}",
            if cfg.site_url.is_empty() {
                "(empty)"
            } else {
                &cfg.site_url
            }
        );
        println!(
            "upgrade_url: {}",
            if cfg.upgrade_url.is_empty() {
                "(empty)"
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
        println!("config: {}", config::engine_yml_path()?.display());
        println!("lib: {}", config::lib_dir()?.display());
        return Ok(());
    }

    let existing = config::load_config().unwrap_or_default();
    let detect = GatewayPair::detect_default();

    let site_url = match site_url_flag {
        Some(s) => s,
        None => {
            let default_shown = if existing.site_url.is_empty() {
                detect.site_url()
            } else {
                existing.site_url.as_str()
            };
            let s = prompt(&format!("site_url (default: {default_shown}): "))?;
            if s.is_empty() {
                if existing.site_url.is_empty() {
                    detect.site_url().to_string()
                } else {
                    existing.site_url.clone()
                }
            } else {
                s
            }
        }
    };

    let pair = GatewayPair::from_url(&site_url);

    let upgrade_url = match upgrade_url_flag {
        Some(s) => s,
        None => {
            let default_shown = if existing.upgrade_url.is_empty() {
                pair.upgrade_url()
            } else {
                existing.upgrade_url.as_str()
            };
            let s = prompt(&format!("upgrade_url (default: {default_shown}): "))?;
            if s.is_empty() {
                if existing.upgrade_url.is_empty() {
                    pair.upgrade_url().to_string()
                } else {
                    existing.upgrade_url.clone()
                }
            } else {
                s
            }
        }
    };

    let compute = match compute_flag {
        Some(s) => parse_compute(&s).map_err(anyhow::Error::msg)?,
        None => {
            let default = if existing.compute.is_empty() {
                "auto"
            } else {
                existing.compute.as_str()
            };
            prompt_choice("compute", &["auto", "cpu", "cuda"], default)?
        }
    };

    let (hf_token, modelscope_api_token) = prompt_regional_hub_token(pair, &existing)?;

    let cfg = AriaConfig {
        site_url,
        upgrade_url,
        compute,
        hf_token,
        modelscope_api_token,
    };
    config::save_config(&cfg)?;
    println!("wrote {}", config::engine_yml_path()?.display());
    Ok(())
}
