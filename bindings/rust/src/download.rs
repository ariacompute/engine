//! Dashboard-only model auto-download for the Rust SDK.
//!
//! Mirrors the `dashboard` branch of `openai/src/download.rs`: resolve the
//! `slug`/`quant` from the model name, request the meta URL with a bearer
//! token, stream the zip, validate the zip magic, extract (flattening a single
//! top-level subdir), and verify the resulting bundle.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DownloadError {
    #[error("invalid model name: {0}")]
    InvalidModelName(String),
    #[error("dashboard request failed: {0}")]
    Request(String),
    #[error("download stream failed: {0}")]
    Stream(String),
    #[error("invalid zip archive: {0}")]
    BadZip(String),
    #[error("extraction failed: {0}")]
    Extract(String),
    #[error("invalid bundle after download: {0}")]
    InvalidBundle(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

const DEFAULT_SITE: &str = "https://ariacompute.com";

fn aria_home() -> Result<PathBuf, DownloadError> {
    if let Ok(override_home) = std::env::var("ARIA_COMPUTE_HOME") {
        if !override_home.is_empty() {
            return Ok(PathBuf::from(override_home));
        }
    }
    let home = if cfg!(windows) {
        std::env::var("USERPROFILE").map_err(|_| {
            DownloadError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "could not resolve home directory",
            ))
        })?
    } else {
        std::env::var("HOME").map_err(|_| {
            DownloadError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "could not resolve home directory",
            ))
        })?
    };
    Ok(PathBuf::from(home).join(".ariacompute"))
}

fn models_dir() -> Result<PathBuf, DownloadError> {
    Ok(aria_home()?.join("models"))
}

/// Parse a model name such as `gemma-4-e2b-it_q4` into `(slug, quant)`.
/// Quant follows the `_q4`/`_q8`/`_q326`/`_q3.26` suffix; defaults to `int4`.
/// Optional codebook-share `_channel` / `_group` (e.g. `*_q326_channel`) is ignored.
fn parse_bundle_name(model: &str) -> Result<(String, String), DownloadError> {
    if model.is_empty() || model.contains('/') || model.contains('\\') {
        return Err(DownloadError::InvalidModelName(model.to_string()));
    }
    let (slug, quant) = if let Some(idx) = model.rfind("_q") {
        let (slug, suffix) = model.split_at(idx);
        let mut suffix = &suffix[2..];
        if let Some(core) = suffix.strip_suffix("_channel") {
            suffix = core;
        } else if let Some(core) = suffix.strip_suffix("_group") {
            suffix = core;
        }
        let quant = match suffix {
            "4" => "int4",
            "8" => "int8",
            "326" | "3.26" => "int326",
            other => {
                return Err(DownloadError::InvalidModelName(format!(
                    "unknown quant suffix _q{other}"
                )))
            }
        };
        (slug.to_string(), quant.to_string())
    } else {
        (model.to_string(), "int4".to_string())
    };
    if slug.is_empty() {
        return Err(DownloadError::InvalidModelName(model.to_string()));
    }
    Ok((slug, quant))
}

fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

#[derive(serde::Deserialize)]
struct DashboardMeta {
    url: String,
}

fn meta_url(site: &str, slug: &str, quant: &str) -> String {
    let sdk = "v1.0";
    format!(
        "{}/api/dashboard/models/{}/download?quant={}&sdk={}&format=json",
        site.trim_end_matches('/'),
        url_encode(slug),
        url_encode(quant),
        url_encode(sdk),
    )
}

fn is_valid_bundle(dir: &Path) -> bool {
    let weight = dir.join("weight.bin");
    let config = dir.join("config.json");
    if !weight.is_file() || !config.is_file() {
        return false;
    }
    let Ok(raw) = std::fs::read_to_string(&config) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return false;
    };
    v.get("format")
        .and_then(|x| x.as_str())
        .map(|f| f == "aria-quant-bundle")
        .unwrap_or(false)
}

/// Flatten a single top-level subdir (when config.json sits inside it).
fn flatten_single_subdir(dir: &Path) -> std::io::Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    if entries.len() != 1 {
        return Ok(());
    }
    let only = &entries[0];
    if !only.is_dir() {
        return Ok(());
    }
    if !only.join("config.json").is_file() {
        return Ok(());
    }
    let tmp = dir.join(format!(".flatten_{}", std::process::id()));
    std::fs::rename(only, &tmp)?;
    for entry in std::fs::read_dir(&tmp)? {
        let entry = entry?;
        std::fs::rename(entry.path(), dir.join(entry.file_name()))?;
    }
    std::fs::remove_dir_all(&tmp)?;
    entries.clear();
    Ok(())
}

fn extract_zip(data: &[u8], dest: &Path) -> Result<(), DownloadError> {
    if data.len() < 4 || &data[0..2] != b"PK" {
        return Err(DownloadError::BadZip("missing PK magic".into()));
    }
    std::fs::create_dir_all(dest)?;
    let cursor = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| DownloadError::BadZip(e.to_string()))?;
    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| DownloadError::Extract(e.to_string()))?;
        let name = match file.enclosed_name() {
            Some(n) => n.to_path_buf(),
            None => continue,
        };
        let out_path = dest.join(&name);
        if file.is_dir() {
            std::fs::create_dir_all(&out_path)?;
        } else {
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut buf = Vec::with_capacity(file.size() as usize);
            file.read_to_end(&mut buf)?;
            let mut out = std::fs::File::create(&out_path)?;
            out.write_all(&buf)?;
        }
    }
    flatten_single_subdir(dest).map_err(DownloadError::Io)?;
    Ok(())
}

fn atomic_replace(src: &Path, dst: &Path) -> std::io::Result<()> {
    if dst.exists() {
        std::fs::remove_dir_all(dst)?;
    }
    std::fs::rename(src, dst)
}

/// Download `model` from the Dashboard private source into
/// `~/.ariacompute/models/{model}`, then return that directory.
///
/// If a valid bundle already exists at the cache path, the download is skipped.
pub fn download_model(
    model: &str,
    token: &str,
    site: Option<&str>,
) -> Result<PathBuf, DownloadError> {
    let (slug, quant) = parse_bundle_name(model)?;
    let site = site.unwrap_or(DEFAULT_SITE);
    let cache = models_dir()?.join(model);

    if cache.exists() && is_valid_bundle(&cache) {
        return Ok(cache);
    }

    let url = meta_url(site, &slug, &quant);
    let agent = ureq::AgentBuilder::new().build();
    let meta: DashboardMeta = agent
        .get(&url)
        .set("Authorization", &format!("Bearer {token}"))
        .call()
        .map_err(|e| DownloadError::Request(e.to_string()))?
        .into_json::<DashboardMeta>()
        .map_err(|e| DownloadError::Request(e.to_string()))?;
    if meta.url.is_empty() {
        return Err(DownloadError::Request(
            "dashboard meta returned empty url".into(),
        ));
    }

    let mut reader = agent
        .get(&meta.url)
        .set("Authorization", &format!("Bearer {token}"))
        .call()
        .map_err(|e| DownloadError::Stream(e.to_string()))?
        .into_reader();
    let mut data = Vec::new();
    reader
        .read_to_end(&mut data)
        .map_err(|e| DownloadError::Stream(e.to_string()))?;

    let staging = models_dir()?.join(format!(".{}.partial", model));
    if staging.exists() {
        std::fs::remove_dir_all(&staging)?;
    }
    extract_zip(&data, &staging)?;
    if !is_valid_bundle(&staging) {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(DownloadError::InvalidBundle(
            "downloaded archive did not contain a valid aria-quant-bundle".into(),
        ));
    }
    atomic_replace(&staging, &cache)?;
    Ok(cache)
}
