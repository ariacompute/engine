//! Download AFM-D checkpoints into `~/.ariacompute/models/{name}`.
//!
//! Hub choice follows `engine.yml` `site_url` (`.cn` → ModelScope, else Hugging Face).
//! Uses hub **HTTP APIs** (tree/list + resolve URLs) — not huggingface-cli / modelscope CLI.
//! Tokens from setup (`hf_token` / `modelscope_api_token`) as Bearer auth.
//! Failed downloads do not leave list/check-visible cache entries.
//!
//! `afm-dd` pulls the PEFT package **and** MiniCPM5-2B safetensors into `base/`
//! (engine candle does not use GGUF).

use ariacompute_core::config::{self, ensure_aria_home, model_cache_dir, models_dir, AriaConfig};
use ariacompute_core::contract::{DD_BASE_MODEL, DD_BASE_REVISION};
use ariacompute_core::gateway::{preferred_hub, PublicHub};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::Value;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

const LIST_TIMEOUT: Duration = Duration::from_secs(30);
/// Overall transfer budget per file (large safetensors over proxy can exceed 10+ minutes).
const FETCH_TIMEOUT: Duration = Duration::from_secs(4 * 60 * 60);
const FETCH_RETRIES: u32 = 3;
/// ModelScope mirror of the pinned MiniCPM5-2B base (`openbmb/…` on HF).
const DD_BASE_MS_REPO: &str = "OpenBMB/MiniCPM5-2B";
const DD_BASE_SUBDIR: &str = "base";

#[derive(Debug, Clone, PartialEq, Eq)]
struct RemoteFile {
    /// Repo-relative path (may include `/`, e.g. `tokenizer/vocab.json`).
    path: String,
}

/// Normalize cache / CLI aliases to a short product name when known.
fn short_product_name(model: &str) -> &str {
    match model {
        "afm-de" | "afm_de" | "encoder" => "afm-de",
        "afm-dd" | "afm_dd" | "decoder" => "afm-dd",
        other => other,
    }
}

fn is_afm_dd(model: &str) -> bool {
    short_product_name(model) == "afm-dd"
        || model.ends_with("/afm-dd")
        || model.ends_with("/afm_dd")
}

/// Resolve well-known AFM model ids to Hub repo ids for the given hub.
pub fn resolve_hub_repo(model: &str, hub: PublicHub) -> String {
    let name = short_product_name(model);
    if name.contains('/') {
        return name.to_string();
    }
    match hub {
        PublicHub::HuggingFace => format!("ariacompute/{name}"),
        PublicHub::ModelScope => format!("AriaCompute/{name}"),
    }
}

fn looks_like_encoder_checkpoint(path: &Path) -> bool {
    path.is_dir()
        && (path.join("model.safetensors").is_file() || path.join("rl_agent_config.json").is_file())
}

fn has_decoder_adapter(path: &Path) -> bool {
    path.is_dir()
        && (path.join("adapter_config.json").is_file()
            || path.join("dd_config.json").is_file()
            || path.join("adapter").join("adapter_config.json").is_file())
}

fn has_llama_safetensors(dir: &Path) -> bool {
    if !dir.is_dir() || !dir.join("config.json").is_file() {
        return false;
    }
    if dir.join("model.safetensors").is_file() {
        return true;
    }
    if dir.join("model.safetensors.index.json").is_file() {
        return true;
    }
    if let Ok(rd) = fs::read_dir(dir) {
        for ent in rd.flatten() {
            let name = ent.file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".safetensors")
                && !name.starts_with("adapter")
                && !name.contains("lora")
            {
                return true;
            }
        }
    }
    false
}

/// MiniCPM safetensors under the decoder cache (`base/`, `merged/`, or root).
pub fn has_decoder_base_weights(path: &Path) -> bool {
    has_llama_safetensors(path)
        || has_llama_safetensors(&path.join(DD_BASE_SUBDIR))
        || has_llama_safetensors(&path.join("merged"))
}

/// True when the directory is a **complete** encoder or decoder checkpoint.
/// Decoder requires PEFT markers **and** MiniCPM5-2B safetensors (not GGUF alone).
pub fn looks_like_checkpoint(path: &Path) -> bool {
    looks_like_encoder_checkpoint(path)
        || (has_decoder_adapter(path) && has_decoder_base_weights(path))
}

fn remove_incomplete_cache(path: &Path) {
    // Keep adapter-only trees so a follow-up download can fill `base/` without re-pulling PEFT.
    if path.is_dir() && !looks_like_encoder_checkpoint(path) && !has_decoder_adapter(path) {
        let _ = fs::remove_dir_all(path);
    }
}

fn hub_token(hub: PublicHub, cfg: &AriaConfig) -> Option<String> {
    let raw = match hub {
        PublicHub::HuggingFace => cfg.hf_token.as_str(),
        PublicHub::ModelScope => cfg.modelscope_api_token.as_str(),
    };
    let t = raw.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

fn hub_token_field(hub: PublicHub) -> &'static str {
    match hub {
        PublicHub::HuggingFace => "hf_token",
        PublicHub::ModelScope => "modelscope_api_token",
    }
}

fn skip_hub_path(path: &str) -> bool {
    let base = path.rsplit('/').next().unwrap_or(path);
    base.starts_with('.') || base == ".gitattributes" || base == ".gitignore"
}

/// Skip GGUF and other non-candle assets from the afm-dd product repo.
fn keep_afm_dd_product_file(path: &str) -> bool {
    if skip_hub_path(path) {
        return false;
    }
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".gguf") || lower.starts_with("gguf/") || lower.contains("/gguf/") {
        return false;
    }
    true
}

/// MiniCPM base: safetensors + configs/tokenizer only (skip README / images / GGUF).
fn keep_decoder_base_file(path: &str) -> bool {
    if skip_hub_path(path) {
        return false;
    }
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".gguf") || lower.starts_with("gguf/") {
        return false;
    }
    lower.ends_with(".safetensors")
        || lower.ends_with(".safetensors.index.json")
        || lower.ends_with("config.json")
        || lower.contains("tokenizer")
        || lower.ends_with("special_tokens_map.json")
        || lower.ends_with("generation_config.json")
        || lower.contains("chat_template")
}

fn io_err<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::other(e.to_string())
}

fn http_client(timeout: Duration) -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .timeout(timeout)
        .connect_timeout(Duration::from_secs(30))
        .pool_idle_timeout(Duration::from_secs(90))
        .redirect(reqwest::redirect::Policy::limited(10))
        .cookie_store(true)
        .user_agent(concat!("aria-engine/", env!("CARGO_PKG_VERSION")))
        .build()
}

fn apply_auth(
    mut req: reqwest::RequestBuilder,
    hub: PublicHub,
    cfg: &AriaConfig,
) -> reqwest::RequestBuilder {
    if let Some(token) = hub_token(hub, cfg) {
        match hub {
            // Hugging Face: Bearer
            PublicHub::HuggingFace => req = req.bearer_auth(token),
            // ModelScope accepts Bearer; also send SDK-style `token …` for older gateways.
            PublicHub::ModelScope => {
                req = req
                    .header(reqwest::header::AUTHORIZATION, format!("Bearer {token}"))
                    .header("X-ModelScope-Authorization", format!("token {token}"));
            }
        }
    }
    req
}

fn default_revision(hub: PublicHub) -> &'static str {
    match hub {
        PublicHub::HuggingFace => "main",
        PublicHub::ModelScope => "master",
    }
}

fn resolve_file_urls(hub: PublicHub, repo: &str, path: &str, revision: &str) -> Vec<String> {
    let path = path.trim_start_matches('/');
    let rev = if revision.is_empty() {
        default_revision(hub)
    } else {
        revision
    };
    match hub {
        PublicHub::HuggingFace => {
            vec![format!(
                "https://huggingface.co/{repo}/resolve/{rev}/{path}"
            )]
        }
        PublicHub::ModelScope => {
            // Prefer official API download (stable for nested paths + large OSS objects),
            // then legacy /resolve/ URLs as fallback.
            let fp = urlencoding::encode(path);
            let rev_enc = urlencoding::encode(rev);
            vec![
                format!(
                    "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision={rev_enc}&FilePath={fp}"
                ),
                format!(
                    "https://modelscope.cn/api/v1/models/{repo}/repo?Revision={rev_enc}&FilePath={fp}"
                ),
                format!("https://www.modelscope.cn/models/{repo}/resolve/{rev}/{path}"),
                format!("https://modelscope.cn/models/{repo}/resolve/{rev}/{path}"),
            ]
        }
    }
}

pub async fn download_model(model: &str) -> io::Result<PathBuf> {
    ensure_aria_home()?;
    let cfg = config::load_config().unwrap_or_default();
    download_model_with_config(model, &cfg).await
}

pub async fn download_model_with_config(model: &str, cfg: &AriaConfig) -> io::Result<PathBuf> {
    ensure_aria_home()?;
    let dest = model_cache_dir(model)?;
    if looks_like_checkpoint(&dest) {
        eprintln!("download: already present at {}", dest.display());
        return Ok(dest);
    }

    let hub = preferred_hub(&cfg.site_url);

    // Adapter-only afm-dd from an older download: fill MiniCPM safetensors into `base/`.
    if is_afm_dd(model) && has_decoder_adapter(&dest) && !has_decoder_base_weights(&dest) {
        eprintln!(
            "download: afm-dd adapter present; fetching MiniCPM5-2B safetensors → {}/{DD_BASE_SUBDIR}",
            dest.display()
        );
        fetch_decoder_base(hub, &dest.join(DD_BASE_SUBDIR), cfg).await?;
        if looks_like_checkpoint(&dest) {
            println!(
                "downloaded MiniCPM5-2B base via {} → {}/{DD_BASE_SUBDIR}",
                hub.as_str(),
                dest.display()
            );
            return Ok(dest);
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "MiniCPM base download finished but no safetensors under {}/{DD_BASE_SUBDIR}",
                dest.display()
            ),
        ));
    }

    remove_incomplete_cache(&dest);

    let repo = resolve_hub_repo(model, hub);
    let staging = dest.with_extension("partial");
    let _ = fs::remove_dir_all(&staging);

    let result = async {
        let rev = default_revision(hub);
        if is_afm_dd(model) {
            fetch_repo(hub, &repo, &staging, cfg, rev, keep_afm_dd_product_file).await?;
            fetch_decoder_base(hub, &staging.join(DD_BASE_SUBDIR), cfg).await?;
        } else {
            fetch_repo(hub, &repo, &staging, cfg, rev, |p| !skip_hub_path(p)).await?;
        }
        Ok::<(), io::Error>(())
    }
    .await;

    match result {
        Ok(()) => {
            if !looks_like_checkpoint(&staging) {
                let _ = fs::remove_dir_all(&staging);
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "download finished but incomplete checkpoint under {} \
(decoder needs PEFT + MiniCPM safetensors in {DD_BASE_SUBDIR}/)",
                        staging.display()
                    ),
                ));
            }
            atomic_replace(&staging, &dest)?;
            println!(
                "downloaded {repo} via {} → {}",
                hub.as_str(),
                dest.display()
            );
            Ok(dest)
        }
        Err(e) => {
            let _ = fs::remove_dir_all(&staging);
            remove_incomplete_cache(&dest);
            Err(e)
        }
    }
}

async fn fetch_decoder_base(hub: PublicHub, dest: &Path, cfg: &AriaConfig) -> io::Result<()> {
    let (repo, revisions): (String, Vec<&str>) = match hub {
        PublicHub::HuggingFace => (DD_BASE_MODEL.to_string(), vec![DD_BASE_REVISION, "main"]),
        PublicHub::ModelScope => (
            DD_BASE_MS_REPO.to_string(),
            vec![DD_BASE_REVISION, "master"],
        ),
    };
    let mut last = None;
    for rev in revisions {
        eprintln!(
            "download: MiniCPM base {} @ {rev} → {}",
            repo,
            dest.display()
        );
        match fetch_repo(hub, &repo, dest, cfg, rev, keep_decoder_base_file).await {
            Ok(()) if has_llama_safetensors(dest) => return Ok(()),
            Ok(()) => {
                last = Some(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{repo}@{rev}: listed files but no safetensors"),
                ));
            }
            Err(e) => last = Some(e),
        }
    }
    Err(last.unwrap_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("failed to fetch MiniCPM base {repo}"),
        )
    }))
}

async fn fetch_repo(
    hub: PublicHub,
    repo: &str,
    staging: &Path,
    cfg: &AriaConfig,
    revision: &str,
    keep: impl Fn(&str) -> bool,
) -> io::Result<()> {
    let list_client = http_client(LIST_TIMEOUT).map_err(io_err)?;
    let files = list_hub_files(&list_client, hub, repo, cfg, revision).await?;
    let mut files: Vec<RemoteFile> = files.into_iter().filter(|f| keep(&f.path)).collect();
    if files.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("{}: empty file list for {repo}@{revision}", hub.as_str()),
        ));
    }
    fs::create_dir_all(staging)?;
    let fetch_client = http_client(FETCH_TIMEOUT).map_err(io_err)?;
    // Small metadata first; large weight files last (clearer progress / fewer partial stalls).
    files.sort_by_key(|f| {
        let lower = f.path.to_ascii_lowercase();
        let heavy = lower.ends_with(".safetensors")
            || lower.ends_with(".bin")
            || lower.ends_with(".gguf")
            || lower.ends_with(".pt");
        (heavy, f.path.clone())
    });
    eprintln!(
        "download: {} ({repo}@{revision}) — {} files",
        hub.as_str(),
        files.len()
    );
    for file in &files {
        fetch_one_file(&fetch_client, hub, repo, &file.path, staging, cfg, revision).await?;
    }
    Ok(())
}

async fn fetch_one_file(
    client: &reqwest::Client,
    hub: PublicHub,
    repo: &str,
    path: &str,
    staging: &Path,
    cfg: &AriaConfig,
    revision: &str,
) -> io::Result<()> {
    let out = staging.join(path);
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    let label = format!("download {path}");
    let mut last_err = None;
    for attempt in 1..=FETCH_RETRIES {
        for url in resolve_file_urls(hub, repo, path, revision) {
            let req = apply_auth(client.get(&url), hub, cfg);
            match req.send().await {
                Ok(r) if r.status().is_success() => {
                    match stream_response_to_file(r, &out, &label).await {
                        Ok(_) => return Ok(()),
                        Err(e) => {
                            let _ = fs::remove_file(&out);
                            last_err = Some(format!("body: {e}"));
                        }
                    }
                }
                Ok(r) if r.status().as_u16() == 401 || r.status().as_u16() == 403 => {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        format!(
                            "auth failed HTTP {}; run `aria-engine setup` to set {}",
                            r.status(),
                            hub_token_field(hub)
                        ),
                    ));
                }
                Ok(r) => {
                    last_err = Some(format!("HTTP {} for {url}", r.status()));
                }
                Err(e) => {
                    last_err = Some(format!("send: {e}"));
                }
            }
        }
        if attempt < FETCH_RETRIES {
            let backoff = Duration::from_secs(2u64.pow(attempt));
            eprintln!(
                "download: retry {attempt}/{FETCH_RETRIES} for {path} after {}s ({})",
                backoff.as_secs(),
                last_err.as_deref().unwrap_or("error")
            );
            tokio::time::sleep(backoff).await;
        }
    }
    Err(io::Error::other(format!(
        "{}: failed to fetch {path}: {}",
        hub.as_str(),
        last_err.unwrap_or_else(|| "unreachable".into())
    )))
}

async fn list_hub_files(
    client: &reqwest::Client,
    hub: PublicHub,
    repo: &str,
    cfg: &AriaConfig,
    revision: &str,
) -> io::Result<Vec<RemoteFile>> {
    let rev = if revision.is_empty() {
        default_revision(hub)
    } else {
        revision
    };
    match hub {
        PublicHub::HuggingFace => {
            let rev_enc = urlencoding::encode(rev);
            let url =
                format!("https://huggingface.co/api/models/{repo}/tree/{rev_enc}?recursive=true");
            let json = get_json_paginated(client, hub, cfg, &url).await?;
            Ok(parse_hf_tree(&json))
        }
        PublicHub::ModelScope => {
            let rev_enc = urlencoding::encode(rev);
            let urls = [
                format!(
                    "https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision={rev_enc}&Recursive=true"
                ),
                format!(
                    "https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision={rev_enc}&Recursive=true"
                ),
                // Some gateways want an explicit Root=/
                format!(
                    "https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision={rev_enc}&Recursive=true&Root="
                ),
            ];
            let mut last = None;
            for url in urls {
                match get_json(client, hub, cfg, &url).await {
                    Ok(json) => {
                        let files = parse_ms_files(&json);
                        if !files.is_empty() {
                            return Ok(files);
                        }
                        last = Some(io::Error::new(
                            io::ErrorKind::NotFound,
                            "empty ModelScope listing",
                        ));
                    }
                    Err(e) => last = Some(e),
                }
            }
            Err(last.unwrap_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, "ModelScope listing failed")
            }))
        }
    }
}

fn parse_hf_tree(json: &Value) -> Vec<RemoteFile> {
    let entries = match json {
        Value::Array(a) => a.as_slice(),
        Value::Object(o) => o
            .get("items")
            .or_else(|| o.get("tree"))
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[]),
        _ => &[],
    };
    let mut out = Vec::new();
    for entry in entries {
        let ty = entry
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if ty == "directory" || ty == "tree" || ty == "folder" {
            continue;
        }
        if !ty.is_empty() && ty != "file" && ty != "blob" && ty != "unknown" {
            continue;
        }
        let path = entry
            .get("path")
            .or_else(|| entry.get("rfilename"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if path.is_empty() || skip_hub_path(path) {
            continue;
        }
        out.push(RemoteFile {
            path: path.to_string(),
        });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out.dedup_by(|a, b| a.path == b.path);
    out
}

fn ms_file_array(json: &Value) -> &[Value] {
    if let Some(arr) = json.as_array() {
        return arr;
    }
    let data = json.get("Data").or_else(|| json.get("data"));
    match data {
        Some(Value::Array(a)) => a.as_slice(),
        Some(Value::Object(o)) => o
            .get("Files")
            .or_else(|| o.get("files"))
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[]),
        _ => json
            .get("Files")
            .or_else(|| json.get("files"))
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[]),
    }
}

fn parse_ms_files(json: &Value) -> Vec<RemoteFile> {
    let files = ms_file_array(json);
    let mut out = Vec::new();
    for entry in files {
        let ty = entry
            .get("Type")
            .or_else(|| entry.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("blob")
            .to_ascii_lowercase();
        if ty == "tree" || ty == "directory" || ty == "folder" {
            continue;
        }
        let path = entry
            .get("Path")
            .or_else(|| entry.get("path"))
            .or_else(|| entry.get("Name"))
            .or_else(|| entry.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if path.is_empty() || skip_hub_path(path) {
            continue;
        }
        out.push(RemoteFile {
            path: path.to_string(),
        });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out.dedup_by(|a, b| a.path == b.path);
    out
}

async fn get_json(
    client: &reqwest::Client,
    hub: PublicHub,
    cfg: &AriaConfig,
    url: &str,
) -> io::Result<Value> {
    let req = apply_auth(
        client.get(url).header("Accept", "application/json"),
        hub,
        cfg,
    );
    let resp = req.send().await.map_err(io_err)?;
    let status = resp.status();
    if status.as_u16() == 401 || status.as_u16() == 403 {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "auth failed HTTP {}; run `aria-engine setup` to set {}",
                status,
                hub_token_field(hub)
            ),
        ));
    }
    if !status.is_success() {
        return Err(io::Error::other(format!("hub HTTP {status} for {url}")));
    }
    resp.json::<Value>().await.map_err(io_err)
}

async fn get_json_paginated(
    client: &reqwest::Client,
    hub: PublicHub,
    cfg: &AriaConfig,
    start_url: &str,
) -> io::Result<Value> {
    let mut url = start_url.to_string();
    let mut all = Vec::new();
    for _ in 0..32 {
        let req = apply_auth(
            client.get(&url).header("Accept", "application/json"),
            hub,
            cfg,
        );
        let resp = req.send().await.map_err(io_err)?;
        let status = resp.status();
        if status.as_u16() == 401 || status.as_u16() == 403 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "auth failed HTTP {}; run `aria-engine setup` to set {}",
                    status,
                    hub_token_field(hub)
                ),
            ));
        }
        if !status.is_success() {
            return Err(io::Error::other(format!("hub HTTP {status} for {url}")));
        }
        let next = next_link(resp.headers());
        let json = resp.json::<Value>().await.map_err(io_err)?;
        match json {
            Value::Array(mut a) => all.append(&mut a),
            other => {
                if all.is_empty() {
                    return Ok(other);
                }
                break;
            }
        }
        match next {
            Some(n) => url = n,
            None => break,
        }
    }
    Ok(Value::Array(all))
}

fn next_link(headers: &reqwest::header::HeaderMap) -> Option<String> {
    let link = headers.get(reqwest::header::LINK)?.to_str().ok()?;
    for part in link.split(',') {
        if part.contains("rel=\"next\"") || part.contains("rel=next") {
            let start = part.find('<')?;
            let end = part.find('>')?;
            if end > start + 1 {
                return Some(part[start + 1..end].to_string());
            }
        }
    }
    None
}

async fn stream_response_to_file(
    resp: reqwest::Response,
    path: &Path,
    label: &str,
) -> io::Result<u64> {
    let total = resp.content_length();
    let show = !label.is_empty() && io::stderr().is_terminal();
    let pb = if !show {
        ProgressBar::hidden()
    } else if let Some(n) = total {
        let pb = ProgressBar::new(n);
        pb.set_style(
            ProgressStyle::with_template(
                "{msg} [{bar:40.green/bright.black}] {bytes}/{total_bytes} ({bytes_per_sec}, ETA {eta})",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("█▉▊▋▌▍▎▏ "),
        );
        pb.set_message(label.to_string());
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template("{msg} {spinner:.green} {bytes} ({bytes_per_sec})")
                .unwrap_or_else(|_| ProgressStyle::default_spinner()),
        );
        pb.set_message(label.to_string());
        pb.enable_steady_tick(Duration::from_millis(100));
        pb
    };

    let mut file = fs::File::create(path)?;
    let mut stream = resp.bytes_stream();
    let mut downloaded = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(io_err)?;
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;
        pb.set_position(downloaded);
    }
    file.flush()?;
    pb.finish_and_clear();
    Ok(downloaded)
}

fn atomic_replace(staging: &Path, dest: &Path) -> io::Result<()> {
    if dest.exists() {
        fs::remove_dir_all(dest)?;
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::rename(staging, dest)?;
    Ok(())
}

pub fn list_models() -> io::Result<Vec<String>> {
    let dir = models_dir()?;
    if !dir.is_dir() {
        return Ok(vec![]);
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let path = entry.path();
        if !looks_like_checkpoint(&path) {
            remove_incomplete_cache(&path);
            continue;
        }
        out.push(entry.file_name().to_string_lossy().into_owned());
    }
    out.sort();
    Ok(out)
}

/// One row for `aria-engine list` (local ∪ regional Hub org catalog).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListedModel {
    /// Cache / download name (e.g. `afm-de`).
    pub name: String,
    /// Full Hub id when known (e.g. `ariacompute/afm-de`).
    pub hub_id: Option<String>,
    pub downloaded: bool,
}

impl ListedModel {
    pub fn status(&self) -> &'static str {
        if self.downloaded {
            "downloaded"
        } else {
            "not downloaded"
        }
    }
}

fn hub_org(hub: PublicHub) -> &'static str {
    match hub {
        PublicHub::HuggingFace => "ariacompute",
        PublicHub::ModelScope => "AriaCompute",
    }
}

/// Short cache name for an org model id (`ariacompute/afm-de` → `afm-de`).
fn short_model_name(hub_id: &str, org: &str) -> String {
    let prefix = format!("{org}/");
    if let Some(rest) = hub_id.strip_prefix(&prefix) {
        if !rest.is_empty() && !rest.contains('/') {
            return rest.to_string();
        }
    }
    hub_id.rsplit('/').next().unwrap_or(hub_id).to_string()
}

fn is_downloaded_locally(name: &str, hub_id: Option<&str>) -> bool {
    let Ok(home) = models_dir() else {
        return false;
    };
    let candidates = [
        Some(name),
        hub_id,
        hub_id.and_then(|id| id.rsplit('/').next()),
    ];
    for c in candidates.into_iter().flatten() {
        if looks_like_checkpoint(&home.join(c)) {
            return true;
        }
    }
    false
}

/// List local complete checkpoints ∪ models under the regional Hub org.
pub async fn list_models_with_hub(cfg: &AriaConfig) -> io::Result<Vec<ListedModel>> {
    ensure_aria_home()?;
    let hub = preferred_hub(&cfg.site_url);
    let org = hub_org(hub);
    let local = list_models().unwrap_or_default();
    let mut local_set: std::collections::BTreeSet<String> = local.into_iter().collect();

    let remote = match list_hub_org_models(hub, org, cfg).await {
        Ok(ids) => ids,
        Err(e) => {
            eprintln!(
                "list: hub catalog unavailable ({}): {e}; showing local only",
                hub.as_str()
            );
            Vec::new()
        }
    };

    let mut rows: Vec<ListedModel> = Vec::new();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    for hub_id in remote {
        let name = short_model_name(&hub_id, org);
        let downloaded = is_downloaded_locally(&name, Some(&hub_id));
        if downloaded {
            local_set.remove(&name);
            local_set.remove(&hub_id);
        }
        seen.insert(name.clone());
        rows.push(ListedModel {
            name,
            hub_id: Some(hub_id),
            downloaded,
        });
    }

    // Local-only checkpoints (not in Hub org catalog).
    for name in local_set {
        if seen.contains(&name) {
            continue;
        }
        rows.push(ListedModel {
            name,
            hub_id: None,
            downloaded: true,
        });
    }

    rows.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(rows)
}

async fn list_hub_org_models(
    hub: PublicHub,
    org: &str,
    cfg: &AriaConfig,
) -> io::Result<Vec<String>> {
    let client = http_client(LIST_TIMEOUT).map_err(io_err)?;
    match hub {
        PublicHub::HuggingFace => list_hf_org_models(&client, org, cfg).await,
        PublicHub::ModelScope => list_ms_org_models(&client, org, cfg).await,
    }
}

async fn list_hf_org_models(
    client: &reqwest::Client,
    org: &str,
    cfg: &AriaConfig,
) -> io::Result<Vec<String>> {
    let start = format!(
        "https://huggingface.co/api/models?author={org}&limit=100&sort=lastModified&direction=-1"
    );
    let json = get_json_paginated(client, PublicHub::HuggingFace, cfg, &start).await?;
    let mut out = Vec::new();
    let entries = match &json {
        Value::Array(a) => a.as_slice(),
        _ => &[],
    };
    for entry in entries {
        let id = entry
            .get("id")
            .or_else(|| entry.get("modelId"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if id.is_empty() {
            continue;
        }
        let prefix = format!("{org}/");
        if id.starts_with(&prefix) {
            out.push(id.to_string());
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

async fn list_ms_org_models(
    client: &reqwest::Client,
    org: &str,
    cfg: &AriaConfig,
) -> io::Result<Vec<String>> {
    let mut out = Vec::new();
    let mut page: u32 = 1;
    loop {
        let url = format!(
            "https://www.modelscope.cn/openapi/v1/models?owner={org}&page_size=50&page_number={page}"
        );
        let json = get_json(client, PublicHub::ModelScope, cfg, &url).await?;
        let data = json.get("data").cloned().unwrap_or(Value::Null);
        let models = data
            .get("models")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        if models.is_empty() {
            break;
        }
        for entry in &models {
            let id = entry
                .get("id")
                .or_else(|| entry.get("Path"))
                .or_else(|| entry.get("path"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if id.is_empty() {
                continue;
            }
            let full = if id.contains('/') {
                id.to_string()
            } else {
                format!("{org}/{id}")
            };
            let prefix = format!("{org}/");
            if full.starts_with(&prefix) {
                out.push(full);
            }
        }
        let total = data
            .get("total_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        if (out.len() as u64) >= total || models.len() < 50 {
            break;
        }
        page += 1;
        if page > 50 {
            break;
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

/// Print `aria-engine list` rows.
pub async fn cmd_list() -> anyhow::Result<()> {
    let cfg = config::load_config().unwrap_or_default();
    let rows = list_models_with_hub(&cfg).await?;
    if rows.is_empty() {
        println!("(no models)");
        return Ok(());
    }
    let width = rows.iter().map(|r| r.name.len()).max().unwrap_or(0);
    for row in rows {
        println!("{:<width$}  {}", row.name, row.status(), width = width);
    }
    Ok(())
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
    match model {
        Some(m) => {
            let path = model_cache_dir(m)?;
            if !looks_like_checkpoint(&path) {
                remove_incomplete_cache(&path);
                println!("{m}: not found (no complete checkpoint)");
                return Ok(());
            }
            let de = looks_like_encoder_checkpoint(&path);
            let dd_adapter = has_decoder_adapter(&path);
            let dd_base = has_decoder_base_weights(&path);
            println!(
                "{m}: path={} encoder_like={de} decoder_adapter={dd_adapter} decoder_base={dd_base} complete={}",
                path.display(),
                looks_like_checkpoint(&path)
            );
        }
        None => {
            let names = list_models()?;
            if names.is_empty() {
                println!("(no models)");
                return Ok(());
            }
            for name in names {
                let path = model_cache_dir(&name)?;
                let de = looks_like_encoder_checkpoint(&path);
                let dd_adapter = has_decoder_adapter(&path);
                let dd_base = has_decoder_base_weights(&path);
                println!(
                    "{name}: path={} encoder_like={de} decoder_adapter={dd_adapter} decoder_base={dd_base} complete={}",
                    path.display(),
                    looks_like_checkpoint(&path)
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn resolve_repo_by_hub() {
        assert_eq!(
            resolve_hub_repo("afm-de", PublicHub::HuggingFace),
            "ariacompute/afm-de"
        );
        assert_eq!(
            resolve_hub_repo("encoder", PublicHub::ModelScope),
            "AriaCompute/afm-de"
        );
        assert_eq!(
            resolve_hub_repo("org/custom", PublicHub::HuggingFace),
            "org/custom"
        );
    }

    #[test]
    fn resolve_urls() {
        assert_eq!(
            resolve_file_urls(
                PublicHub::HuggingFace,
                "ariacompute/afm-de",
                "model.safetensors",
                "main",
            ),
            vec![
                "https://huggingface.co/ariacompute/afm-de/resolve/main/model.safetensors"
                    .to_string()
            ]
        );
        assert_eq!(
            resolve_file_urls(
                PublicHub::HuggingFace,
                "openbmb/MiniCPM5-2B",
                "config.json",
                DD_BASE_REVISION,
            ),
            vec![format!(
                "https://huggingface.co/openbmb/MiniCPM5-2B/resolve/{DD_BASE_REVISION}/config.json"
            )]
        );
        let ms = resolve_file_urls(
            PublicHub::ModelScope,
            "AriaCompute/afm-de",
            "tokenizer/vocab.json",
            "master",
        );
        assert!(
            ms[0].starts_with(
                "https://www.modelscope.cn/api/v1/models/AriaCompute/afm-de/repo?Revision=master&FilePath="
            ),
            "expected API FilePath URL, got {}",
            ms[0]
        );
        assert!(ms[0].contains("tokenizer%2Fvocab.json"));
        assert!(ms
            .iter()
            .any(|u| u.contains("/resolve/master/tokenizer/vocab.json")));
    }

    #[test]
    fn afm_dd_skips_gguf_keeps_adapter() {
        assert!(keep_afm_dd_product_file("adapter_config.json"));
        assert!(keep_afm_dd_product_file("adapter_model.safetensors"));
        assert!(!keep_afm_dd_product_file("gguf/afm-dd-2b-Q8_0.gguf"));
        assert!(!keep_afm_dd_product_file("model.gguf"));
    }

    #[test]
    fn decoder_base_keeps_safetensors() {
        assert!(keep_decoder_base_file("model.safetensors"));
        assert!(keep_decoder_base_file("model-00001-of-00002.safetensors"));
        assert!(keep_decoder_base_file("model.safetensors.index.json"));
        assert!(keep_decoder_base_file("config.json"));
        assert!(keep_decoder_base_file("tokenizer.json"));
        assert!(!keep_decoder_base_file("README.md"));
        assert!(!keep_decoder_base_file("gguf/x.gguf"));
    }

    #[test]
    fn parse_hf_keeps_nested_paths() {
        let json = json!([
            {"type": "file", "path": "model.safetensors"},
            {"type": "directory", "path": "tokenizer"},
            {"type": "file", "path": "tokenizer/vocab.json"},
            {"type": "file", "path": ".gitattributes"},
        ]);
        let files = parse_hf_tree(&json);
        assert_eq!(
            files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec!["model.safetensors", "tokenizer/vocab.json"]
        );
    }

    #[test]
    fn parse_ms_files_paths() {
        let json = json!({
            "Data": {
                "Files": [
                    {"Type": "blob", "Path": "rl_agent_config.json"},
                    {"Type": "tree", "Path": "encoder"},
                    {"Type": "blob", "Path": "encoder/config.json"},
                ]
            }
        });
        let files = parse_ms_files(&json);
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].path, "encoder/config.json");
        assert_eq!(files[1].path, "rl_agent_config.json");
    }

    #[test]
    fn short_name_strips_org() {
        assert_eq!(
            short_model_name("ariacompute/afm-de", "ariacompute"),
            "afm-de"
        );
        assert_eq!(
            short_model_name("AriaCompute/afm-dd", "AriaCompute"),
            "afm-dd"
        );
        assert_eq!(short_model_name("other/x", "ariacompute"), "x");
    }

    #[test]
    fn listed_status_labels() {
        assert_eq!(
            ListedModel {
                name: "afm-de".into(),
                hub_id: None,
                downloaded: true,
            }
            .status(),
            "downloaded"
        );
        assert_eq!(
            ListedModel {
                name: "afm-dd".into(),
                hub_id: Some("ariacompute/afm-dd".into()),
                downloaded: false,
            }
            .status(),
            "not downloaded"
        );
    }

    #[test]
    fn incomplete_dir_is_not_checkpoint() {
        let tmp = tempdir().unwrap();
        let p = tmp.path().join("afm-de");
        fs::create_dir_all(&p).unwrap();
        assert!(!looks_like_checkpoint(&p));
        fs::write(p.join("rl_agent_config.json"), "{}").unwrap();
        assert!(looks_like_checkpoint(&p));
    }

    #[test]
    fn decoder_needs_base_safetensors() {
        let tmp = tempdir().unwrap();
        let p = tmp.path().join("afm-dd");
        fs::create_dir_all(&p).unwrap();
        fs::write(p.join("dd_config.json"), "{}").unwrap();
        fs::write(p.join("adapter_config.json"), "{}").unwrap();
        assert!(has_decoder_adapter(&p));
        assert!(!looks_like_checkpoint(&p));
        let base = p.join(DD_BASE_SUBDIR);
        fs::create_dir_all(&base).unwrap();
        fs::write(
            base.join("config.json"),
            "{\"architectures\":[\"LlamaForCausalLM\"]}",
        )
        .unwrap();
        fs::write(base.join("model.safetensors"), b"fake").unwrap();
        assert!(has_decoder_base_weights(&p));
        assert!(looks_like_checkpoint(&p));
    }

    #[test]
    fn remove_incomplete_leaves_complete() {
        let tmp = tempdir().unwrap();
        let incomplete = tmp.path().join("bad");
        fs::create_dir_all(&incomplete).unwrap();
        remove_incomplete_cache(&incomplete);
        assert!(!incomplete.exists());

        // Adapter-only is incomplete for serve, but kept so download can fill base/.
        let adapter_only = tmp.path().join("adapter_only");
        fs::create_dir_all(&adapter_only).unwrap();
        fs::write(adapter_only.join("dd_config.json"), "{}").unwrap();
        remove_incomplete_cache(&adapter_only);
        assert!(adapter_only.exists());
        assert!(!looks_like_checkpoint(&adapter_only));
    }

    #[test]
    fn hub_token_from_config_not_env() {
        std::env::set_var("HF_TOKEN", "env-hf");
        std::env::set_var("MODELSCOPE_API_TOKEN", "env-ms");
        let empty = AriaConfig::default();
        assert_eq!(hub_token(PublicHub::HuggingFace, &empty), None);
        assert_eq!(hub_token(PublicHub::ModelScope, &empty), None);
        let cfg = AriaConfig {
            hf_token: "hf_yml".into(),
            modelscope_api_token: "ms_yml".into(),
            ..AriaConfig::default()
        };
        assert_eq!(
            hub_token(PublicHub::HuggingFace, &cfg).as_deref(),
            Some("hf_yml")
        );
        assert_eq!(
            hub_token(PublicHub::ModelScope, &cfg).as_deref(),
            Some("ms_yml")
        );
        std::env::remove_var("HF_TOKEN");
        std::env::remove_var("MODELSCOPE_API_TOKEN");
    }
}
