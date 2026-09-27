//! Download AFM-D checkpoints into `~/.ariacompute/models/{name}`.

use ariacompute_core::config::{ensure_aria_home, model_cache_dir, models_dir};
use std::fs;
use std::io;
use std::path::PathBuf;
use std::process::Command;

/// Resolve well-known AFM model ids to Hub repos.
pub fn resolve_hub_repo(model: &str) -> (String, Option<&'static str>) {
    match model {
        "afm-de" | "afm_de" | "encoder" => ("ariacompute/afm-de".into(), None),
        "afm-dd" | "afm_dd" | "decoder" => ("ariacompute/afm-dd".into(), None),
        other => (other.to_string(), None),
    }
}

pub async fn download_model(model: &str) -> io::Result<PathBuf> {
    ensure_aria_home()?;
    let dest = model_cache_dir(model)?;
    fs::create_dir_all(&dest)?;
    let (repo, _rev) = resolve_hub_repo(model);

    // Prefer huggingface-cli when available; otherwise print instructions.
    let status = Command::new("huggingface-cli")
        .args([
            "download",
            &repo,
            "--local-dir",
            dest.to_str().unwrap_or("."),
        ])
        .status();
    match status {
        Ok(s) if s.success() => {
            println!("downloaded {repo} → {}", dest.display());
            Ok(dest)
        }
        Ok(s) => Err(io::Error::other(format!(
            "huggingface-cli failed with {s}"
        ))),
        Err(_) => {
            // Fallback: try `hf` CLI
            let status2 = Command::new("hf")
                .args([
                    "download",
                    &repo,
                    "--local-dir",
                    dest.to_str().unwrap_or("."),
                ])
                .status();
            match status2 {
                Ok(s) if s.success() => {
                    println!("downloaded {repo} → {}", dest.display());
                    Ok(dest)
                }
                _ => {
                    println!(
                        "install huggingface_hub CLI, then:\n  hf download {repo} --local-dir {}",
                        dest.display()
                    );
                    Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        "huggingface-cli / hf not found",
                    ))
                }
            }
        }
    }
}

pub fn list_models() -> io::Result<Vec<String>> {
    let dir = models_dir()?;
    if !dir.is_dir() {
        return Ok(vec![]);
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            out.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
    out.sort();
    Ok(out)
}

pub fn clean_model(model: Option<&str>) -> io::Result<()> {
    match model {
        Some(m) => {
            let path = model_cache_dir(m)?;
            if path.is_dir() {
                fs::remove_dir_all(path)?;
                println!("removed {m}");
            } else {
                println!("not found: {m}");
            }
        }
        None => {
            let dir = models_dir()?;
            if dir.is_dir() {
                for entry in fs::read_dir(&dir)? {
                    let entry = entry?;
                    if entry.file_type()?.is_dir() {
                        fs::remove_dir_all(entry.path())?;
                    }
                }
            }
            println!("cleared models cache");
        }
    }
    Ok(())
}

pub fn check_model(model: Option<&str>) -> io::Result<()> {
    let names = match model {
        Some(m) => vec![m.to_string()],
        None => list_models()?,
    };
    for name in names {
        let path = model_cache_dir(&name)?;
        let de = path.join("model.safetensors").is_file()
            || path.join("rl_agent_config.json").is_file();
        let dd = path.join("adapter_config.json").is_file()
            || path.join("dd_config.json").is_file();
        println!(
            "{name}: path={} encoder_like={de} decoder_like={dd}",
            path.display()
        );
    }
    Ok(())
}
