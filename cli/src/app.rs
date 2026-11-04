//! CLI surface + subcommand dispatch (kept out of `main.rs` so it is testable).

use crate::download;
use crate::serve::{run_serve, ServeOpts};
use crate::setup::cmd_setup;
use crate::upgrade;
use ariacompute_core::config::{self, parse_compute};
use ariacompute_core::contract::{Track, DD_DEFAULT_PORT};
use ariacompute_core::packing::Record;
use ariacompute_core::systemone::record_from_systemone_question;
use ariacompute_dd::DecoderScorer;
use ariacompute_de::EncoderScorer;
use clap::{ArgAction, Parser, Subcommand};
use serde_json::Value;
use std::io::{self, Read};
use std::path::PathBuf;

pub const ENGINE_VERSION: &str = env!("ARIA_ENGINE_VERSION");

#[derive(Parser)]
#[command(
    name = "aria-engine",
    about = "AFM typed-decision engine CLI (encoder afm_de / decoder afm_dd)",
    version = ENGINE_VERSION,
    arg_required_else_help = true,
    disable_version_flag = true
)]
pub struct Cli {
    #[arg(short = 'v', long = "version", action = ArgAction::Version)]
    pub version: (),
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Write engine.yml (site_url, upgrade_url, compute, hub tokens)
    Setup {
        /// Show config status (secrets redacted)
        #[arg(long)]
        status: bool,
        /// Remove engine.yml (and legacy config.yml)
        #[arg(long)]
        clear: bool,
        #[arg(long)]
        site_url: Option<String>,
        #[arg(long)]
        upgrade_url: Option<String>,
        /// auto | cpu | cuda
        #[arg(long)]
        compute: Option<String>,
    },
    /// Fetch AFM model into ~/.ariacompute/models
    Download {
        model: String,
    },
    /// List local + Hub org models (downloaded / not downloaded)
    List,
    /// Check local checkpoint layout
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
    /// Start System One HTTP server
    Serve {
        /// encoder | decoder
        #[arg(long, default_value = "encoder")]
        track: String,
        /// Checkpoint directory or cache name (default: ~/.ariacompute/models/<model-name>)
        #[arg(long)]
        checkpoint: Option<String>,
        #[arg(long)]
        bind: Option<String>,
        /// Model id / cache name (default: afm-de | afm-dd from --track); used as checkpoint when --checkpoint omitted
        #[arg(long)]
        model_name: Option<String>,
        /// Override engine.yml compute: auto | cpu | cuda
        #[arg(long)]
        compute: Option<String>,
    },
    /// One-shot System One JSON (stdin or --file)
    Decide {
        /// encoder | decoder
        #[arg(long, default_value = "encoder")]
        track: String,
        /// Checkpoint directory or cache name (default: ~/.ariacompute/models/<model-name>)
        #[arg(long)]
        checkpoint: Option<String>,
        /// Model id / cache name (default: afm-de | afm-dd from --track); used as checkpoint when --checkpoint omitted
        #[arg(long)]
        model_name: Option<String>,
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

fn resolve_ckpt(checkpoint: &str) -> PathBuf {
    config::resolve_checkpoint(checkpoint)
}

/// Resolve checkpoint from `--checkpoint` and/or `--model-name` (+ track default).
fn resolve_model_checkpoint(
    track: Track,
    checkpoint: Option<&str>,
    model_name: &str,
) -> anyhow::Result<PathBuf> {
    let ref_ = checkpoint.unwrap_or(model_name);
    let path = resolve_ckpt(ref_);
    if !path.is_dir() {
        anyhow::bail!(
            "checkpoint not found at {} (pass --checkpoint PATH or download with `aria-engine download {model_name}`)",
            path.display()
        );
    }
    // Soft layout hint by track (still allow open() to do the real validation).
    let encoder_like =
        path.join("model.safetensors").is_file() || path.join("rl_agent_config.json").is_file();
    let decoder_like =
        path.join("adapter_config.json").is_file() || path.join("dd_config.json").is_file();
    match track {
        Track::Encoder if !encoder_like && decoder_like => anyhow::bail!(
            "{} looks like a decoder checkpoint; use --track decoder or a different --model-name",
            path.display()
        ),
        Track::Decoder if !decoder_like && encoder_like => anyhow::bail!(
            "{} looks like an encoder checkpoint; use --track encoder or a different --model-name",
            path.display()
        ),
        _ => {}
    }
    Ok(path)
}

fn default_model_name(track: Track) -> &'static str {
    match track {
        Track::Encoder => "afm-de",
        Track::Decoder => "afm-dd",
    }
}

/// Default listen address per track (decoder keeps the documented 8011 port).
fn default_bind(track: Track) -> String {
    match track {
        Track::Decoder => format!("127.0.0.1:{DD_DEFAULT_PORT}"),
        Track::Encoder => "127.0.0.1:8010".into(),
    }
}

async fn cmd_decide(
    track: &str,
    checkpoint: Option<&str>,
    model_name: Option<&str>,
    file: Option<PathBuf>,
    logits: Option<String>,
    semif_out: Option<String>,
) -> anyhow::Result<()> {
    let track = Track::parse(track).map_err(anyhow::Error::msg)?;
    let model_name = model_name
        .map(str::to_string)
        .unwrap_or_else(|| default_model_name(track).into());
    let ckpt = resolve_model_checkpoint(track, checkpoint, &model_name)?;
    let raw = if let Some(p) = file {
        std::fs::read_to_string(p)?
    } else {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf)?;
        buf
    };
    let body: Value = serde_json::from_str(&raw)?;
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

pub async fn run(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
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
        Command::List => download::cmd_list().await,
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
            compute,
        } => {
            let track = Track::parse(&track).map_err(anyhow::Error::msg)?;
            let cfg = config::load_config().unwrap_or_default();
            let compute = match compute {
                Some(s) => parse_compute(&s).map_err(anyhow::Error::msg)?,
                None => parse_compute(&cfg.compute).unwrap_or_else(|_| "auto".into()),
            };
            let model_name = model_name.unwrap_or_else(|| default_model_name(track).into());
            let checkpoint = resolve_model_checkpoint(track, checkpoint.as_deref(), &model_name)?;
            let opts = ServeOpts {
                track,
                checkpoint,
                model_name,
                bind: bind.unwrap_or_else(|| default_bind(track)),
                compute,
            };
            run_serve(opts).await
        }
        Command::Decide {
            track,
            checkpoint,
            model_name,
            file,
            logits,
            semif_out,
        } => {
            cmd_decide(
                &track,
                checkpoint.as_deref(),
                model_name.as_deref(),
                file,
                logits,
                semif_out,
            )
            .await
        }
        Command::Version => {
            println!("aria-engine {ENGINE_VERSION}");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn default_bind_matches_the_documented_ports() {
        assert_eq!(default_bind(Track::Encoder), "127.0.0.1:8010");
        assert_eq!(
            default_bind(Track::Decoder),
            format!("127.0.0.1:{DD_DEFAULT_PORT}")
        );
        assert_eq!(default_model_name(Track::Encoder), "afm-de");
        assert_eq!(default_model_name(Track::Decoder), "afm-dd");
    }

    #[test]
    fn resolve_model_checkpoint_reports_layout_mismatches() {
        let dir = tempdir().unwrap();
        let enc = dir.path().join("enc");
        std::fs::create_dir_all(&enc).unwrap();
        std::fs::write(enc.join("model.safetensors"), b"x").unwrap();
        let dec = dir.path().join("dec");
        std::fs::create_dir_all(&dec).unwrap();
        std::fs::write(dec.join("adapter_config.json"), "{}").unwrap();

        assert_eq!(
            resolve_model_checkpoint(Track::Encoder, Some(enc.to_str().unwrap()), "afm-de")
                .unwrap(),
            enc
        );
        assert_eq!(
            resolve_model_checkpoint(Track::Decoder, Some(dec.to_str().unwrap()), "afm-dd")
                .unwrap(),
            dec
        );
        let err = resolve_model_checkpoint(Track::Encoder, Some(dec.to_str().unwrap()), "afm-de")
            .unwrap_err();
        assert!(err.to_string().contains("looks like a decoder checkpoint"));
        let err = resolve_model_checkpoint(Track::Decoder, Some(enc.to_str().unwrap()), "afm-dd")
            .unwrap_err();
        assert!(err.to_string().contains("looks like an encoder checkpoint"));
        let err = resolve_model_checkpoint(Track::Encoder, None, "afm-not-downloaded").unwrap_err();
        assert!(err.to_string().contains("checkpoint not found"));
    }
}
