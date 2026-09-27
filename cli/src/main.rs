use aria_cli::download;
use aria_cli::serve::{run_serve, ServeOpts};
use aria_cli::upgrade;
use ariacompute_core::config::{self, AriaConfig};
use ariacompute_core::contract::{Track, DD_DEFAULT_PORT};
use ariacompute_core::gateway::GatewayPair;
use ariacompute_core::packing::Record;
use ariacompute_core::systemone::record_from_systemone_question;
use ariacompute_dd::DecoderScorer;
use ariacompute_de::EncoderScorer;
use clap::{ArgAction, Parser, Subcommand};
use serde_json::Value;
use std::io::{self, BufRead, Read, Write};
use std::path::PathBuf;
use std::process;

const ENGINE_VERSION: &str = env!("ARIA_ENGINE_VERSION");

#[derive(Parser)]
#[command(
    name = "aria-engine",
    about = "AFM-D typed-decision engine CLI (encoder afm_de / decoder afm_dd)",
    version = ENGINE_VERSION,
    arg_required_else_help = true,
    disable_version_flag = true
)]
struct Cli {
    #[arg(short = 'v', long = "version", action = ArgAction::Version)]
    _version: (),
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Write engine.yml (site_url, upgrade_url, compute)
    Setup {
        #[arg(long)]
        status: bool,
        #[arg(long)]
        clear: bool,
        #[arg(long)]
        site_url: Option<String>,
        #[arg(long)]
        upgrade_url: Option<String>,
        #[arg(long)]
        compute: Option<String>,
    },
    /// Fetch afm-de / afm-dd (or Hub id) into ~/.ariacompute/models
    Download { model: String },
    /// List cached models
    List,
    /// Check local checkpoint layout
    Check { model: Option<String> },
    /// Remove one cached model or all
    Clean { model: Option<String> },
    /// Replace this CLI + libaria-engine_ffi from Releases
    Upgrade { version: Option<String> },
    /// Start System One HTTP server
    Serve {
        /// encoder | decoder
        #[arg(long, default_value = "decoder")]
        track: String,
        /// Checkpoint directory or cache name
        #[arg(long)]
        checkpoint: String,
        #[arg(long)]
        bind: Option<String>,
        #[arg(long)]
        model_name: Option<String>,
    },
    /// One-shot System One JSON (stdin or --file)
    Decide {
        #[arg(long, default_value = "encoder")]
        track: String,
        #[arg(long)]
        checkpoint: String,
        #[arg(long)]
        file: Option<PathBuf>,
        /// Raw logits JSON array (encoder golden path without weights)
        #[arg(long)]
        logits: Option<String>,
        /// SemIf out JSON (decoder golden path)
        #[arg(long)]
        semif_out: Option<String>,
    },
    Version,
}

fn prompt(label: &str) -> io::Result<String> {
    eprint!("{label}");
    io::stderr().flush()?;
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line)?;
    Ok(line.trim().to_string())
}

fn cmd_setup(
    status: bool,
    clear: bool,
    site_url: Option<String>,
    upgrade_url: Option<String>,
    compute: Option<String>,
) -> anyhow::Result<()> {
    if clear {
        config::clear_config()?;
        println!("cleared engine.yml");
        return Ok(());
    }
    if status {
        let cfg = config::load_config()?;
        println!("site_url: {}", if cfg.site_url.is_empty() { "(empty)" } else { &cfg.site_url });
        println!(
            "upgrade_url: {}",
            if cfg.upgrade_url.is_empty() {
                "(empty)"
            } else {
                &cfg.upgrade_url
            }
        );
        println!("compute: {}", cfg.compute);
        return Ok(());
    }
    let existing = config::load_config().unwrap_or_default();
    let pair = GatewayPair::detect_default();
    let site_url = match site_url {
        Some(s) => s,
        None => {
            let s = prompt(&format!("site_url (default: {}): ", pair.site_url()))?;
            if s.is_empty() {
                if existing.site_url.is_empty() {
                    pair.site_url().to_string()
                } else {
                    existing.site_url.clone()
                }
            } else {
                s
            }
        }
    };
    let upgrade_url = match upgrade_url {
        Some(s) => s,
        None => {
            let s = prompt(&format!(
                "upgrade_url (default: {}): ",
                if existing.upgrade_url.is_empty() {
                    pair.upgrade_url()
                } else {
                    &existing.upgrade_url
                }
            ))?;
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
    let compute = compute.unwrap_or_else(|| existing.compute.clone());
    let cfg = AriaConfig {
        site_url,
        upgrade_url,
        compute,
        hf_token: existing.hf_token,
        modelscope_api_token: existing.modelscope_api_token,
    };
    config::save_config(&cfg)?;
    println!("wrote {}", config::engine_yml_path()?.display());
    Ok(())
}

fn resolve_ckpt(checkpoint: &str) -> PathBuf {
    config::resolve_checkpoint(checkpoint)
}

async fn cmd_decide(
    track: &str,
    checkpoint: &str,
    file: Option<PathBuf>,
    logits: Option<String>,
    semif_out: Option<String>,
) -> anyhow::Result<()> {
    let track = Track::parse(track).map_err(anyhow::Error::msg)?;
    let ckpt = resolve_ckpt(checkpoint);
    let raw = if let Some(p) = file {
        std::fs::read_to_string(p)?
    } else {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf)?;
        buf
    };
    let body: Value = serde_json::from_str(&raw)?;
    // Accept either a single record or System One request.
    let records: Vec<Record> = if body.get("questions").is_some() {
        let state = body.get("state").cloned().unwrap_or(Value::Null);
        let qs = body
            .get("questions")
            .and_then(|q| q.as_object())
            .ok_or_else(|| anyhow::anyhow!("questions must be object"))?;
        let mut out = Vec::new();
        for (qid, q) in qs {
            out.push(record_from_systemone_question(qid, &state, q)?);
        }
        out
    } else {
        vec![Record::from_value(&body)?]
    };

    match track {
        Track::Encoder => {
            let scorer = EncoderScorer::open(&ckpt)?;
            for rec in &records {
                let ans = if let Some(ref logits_s) = logits {
                    let logits: Vec<f32> = serde_json::from_str(logits_s)?;
                    scorer.score_record_with_logits(rec, &logits)?
                } else {
                    scorer.score_record(rec)?
                };
                println!("{}", serde_json::to_string_pretty(&ans)?);
            }
        }
        Track::Decoder => {
            let scorer = DecoderScorer::open(Some(&ckpt))?;
            for rec in &records {
                let ans = if let Some(ref semif) = semif_out {
                    let out: Value = serde_json::from_str(semif)?;
                    scorer.score_from_semif_out(rec, &out)?
                } else {
                    scorer.score_record(rec)?
                };
                println!("{}", serde_json::to_string_pretty(&ans)?);
            }
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    let result = match cli.command {
        Command::Setup {
            status,
            clear,
            site_url,
            upgrade_url,
            compute,
        } => cmd_setup(status, clear, site_url, upgrade_url, compute),
        Command::Download { model } => download::download_model(&model)
            .await
            .map(|_| ())
            .map_err(Into::into),
        Command::List => {
            for m in download::list_models().unwrap_or_default() {
                println!("{m}");
            }
            Ok(())
        }
        Command::Check { model } => download::check_model(model.as_deref()).map_err(Into::into),
        Command::Clean { model } => download::clean_model(model.as_deref()).map_err(Into::into),
        Command::Upgrade { version } => upgrade::run(version.as_deref(), ENGINE_VERSION)
            .await
            .map_err(Into::into),
        Command::Serve {
            track,
            checkpoint,
            bind,
            model_name,
        } => {
            let track = Track::parse(&track).expect("track");
            let default_bind = match track {
                Track::Decoder => format!("127.0.0.1:{DD_DEFAULT_PORT}"),
                Track::Encoder => "127.0.0.1:8010".into(),
            };
            let opts = ServeOpts {
                track,
                checkpoint: resolve_ckpt(&checkpoint),
                model_name: model_name.unwrap_or_else(|| match track {
                    Track::Encoder => "afm-de".into(),
                    Track::Decoder => "afm-dd".into(),
                }),
                bind: bind.unwrap_or(default_bind),
            };
            run_serve(opts).await
        }
        Command::Decide {
            track,
            checkpoint,
            file,
            logits,
            semif_out,
        } => cmd_decide(&track, &checkpoint, file, logits, semif_out).await,
        Command::Version => {
            println!("aria-engine {ENGINE_VERSION}");
            Ok(())
        }
    };

    if let Err(err) = result {
        eprintln!("error: {err:#}");
        process::exit(1);
    }
}
