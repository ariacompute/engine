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

/// Hub endpoint bases (`ARIA_HF_ENDPOINT` / `ARIA_MS_ENDPOINT` allow mirrors + offline tests).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubEndpoints {
    pub hf: String,
    pub ms: String,
    pub ms_alt: String,
}

impl Default for HubEndpoints {
    fn default() -> Self {
        Self {
            hf: "https://huggingface.co".into(),
            ms: "https://www.modelscope.cn".into(),
            ms_alt: "https://modelscope.cn".into(),
        }
    }
}

fn env_endpoint(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().trim_end_matches('/').to_string())
        .filter(|v| !v.is_empty())
}

impl HubEndpoints {
    /// Production endpoints, honouring the mirror overrides.
    pub fn from_env() -> Self {
        let mut out = Self::default();
        if let Some(hf) = env_endpoint("ARIA_HF_ENDPOINT") {
            out.hf = hf;
        }
        if let Some(ms) = env_endpoint("ARIA_MS_ENDPOINT") {
            out.ms_alt = ms.clone();
            out.ms = ms;
        }
        if let Some(ms_alt) = env_endpoint("ARIA_MS_ENDPOINT_ALT") {
            out.ms_alt = ms_alt;
        }
        out
    }
}

/// Retry policy for individual file transfers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    pub attempts: u32,
    /// Base backoff; attempt `n` waits `backoff * 2^(n-1)`.
    pub backoff: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            attempts: FETCH_RETRIES,
            backoff: Duration::from_secs(1),
        }
    }
}

impl RetryPolicy {
    /// Single attempt with no sleeping (tests / offline fixtures).
    pub fn none() -> Self {
        Self {
            attempts: 1,
            backoff: Duration::ZERO,
        }
    }
}

/// Everything the network paths need to be pointed somewhere else.
#[derive(Debug, Clone, Default)]
pub struct DownloadCtx {
    pub endpoints: HubEndpoints,
    pub retry: RetryPolicy,
}

impl DownloadCtx {
    pub fn from_env() -> Self {
        Self {
            endpoints: HubEndpoints::from_env(),
            retry: RetryPolicy::default(),
        }
    }
}

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

/// Create the parent directory of `path` (no-op at a filesystem root).
fn ensure_parent(path: &Path) -> io::Result<()> {
    match path.parent() {
        Some(parent) => fs::create_dir_all(parent),
        None => Ok(()),
    }
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
    // Sharded checkpoints: any non-adapter `.safetensors` shard counts.
    let entries = fs::read_dir(dir).into_iter().flatten().flatten();
    for ent in entries {
        let name = ent.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".safetensors") && !name.starts_with("adapter") && !name.contains("lora")
        {
            return true;
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

fn resolve_file_urls(
    endpoints: &HubEndpoints,
    hub: PublicHub,
    repo: &str,
    path: &str,
    revision: &str,
) -> Vec<String> {
    let path = path.trim_start_matches('/');
    let rev = if revision.is_empty() {
        default_revision(hub)
    } else {
        revision
    };
    match hub {
        PublicHub::HuggingFace => {
            vec![format!("{}/{repo}/resolve/{rev}/{path}", endpoints.hf)]
        }
        PublicHub::ModelScope => {
            // Prefer official API download (stable for nested paths + large OSS objects),
            // then legacy /resolve/ URLs as fallback.
            let fp = urlencoding::encode(path);
            let rev_enc = urlencoding::encode(rev);
            vec![
                format!(
                    "{}/api/v1/models/{repo}/repo?Revision={rev_enc}&FilePath={fp}",
                    endpoints.ms
                ),
                format!(
                    "{}/api/v1/models/{repo}/repo?Revision={rev_enc}&FilePath={fp}",
                    endpoints.ms_alt
                ),
                format!("{}/models/{repo}/resolve/{rev}/{path}", endpoints.ms),
                format!("{}/models/{repo}/resolve/{rev}/{path}", endpoints.ms_alt),
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
    download_model_with_config_at(model, cfg, &DownloadCtx::from_env()).await
}

/// `download_model_with_config` with explicit hub endpoints + retry policy.
pub async fn download_model_with_config_at(
    model: &str,
    cfg: &AriaConfig,
    ctx: &DownloadCtx,
) -> io::Result<PathBuf> {
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
        fetch_decoder_base(ctx, hub, &dest.join(DD_BASE_SUBDIR), cfg).await?;
        // `fetch_decoder_base` only returns Ok once `base/` holds llama safetensors,
        // which together with the adapter above makes the checkpoint complete.
        println!(
            "downloaded MiniCPM5-2B base via {} → {}/{DD_BASE_SUBDIR}",
            hub.as_str(),
            dest.display()
        );
        return Ok(dest);
    }

    remove_incomplete_cache(&dest);

    let repo = resolve_hub_repo(model, hub);
    let staging = dest.with_extension("partial");
    let _ = fs::remove_dir_all(&staging);

    let result = async {
        let rev = default_revision(hub);
        if is_afm_dd(model) {
            fetch_repo(
                ctx,
                hub,
                &repo,
                &staging,
                cfg,
                rev,
                keep_afm_dd_product_file,
            )
            .await?;
            fetch_decoder_base(ctx, hub, &staging.join(DD_BASE_SUBDIR), cfg).await?;
        } else {
            fetch_repo(ctx, hub, &repo, &staging, cfg, rev, |p| !skip_hub_path(p)).await?;
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

async fn fetch_decoder_base(
    ctx: &DownloadCtx,
    hub: PublicHub,
    dest: &Path,
    cfg: &AriaConfig,
) -> io::Result<()> {
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
        match fetch_repo(ctx, hub, &repo, dest, cfg, rev, keep_decoder_base_file).await {
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
    // `revisions` always has at least one entry, so the loop above sets `last`.
    Err(last.expect("decoder base revision list is never empty"))
}

async fn fetch_repo(
    ctx: &DownloadCtx,
    hub: PublicHub,
    repo: &str,
    staging: &Path,
    cfg: &AriaConfig,
    revision: &str,
    keep: impl Fn(&str) -> bool,
) -> io::Result<()> {
    let list_client = http_client(LIST_TIMEOUT).map_err(io_err)?;
    let files = list_hub_files(ctx, &list_client, hub, repo, cfg, revision).await?;
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
        fetch_one_file(
            ctx,
            &fetch_client,
            HubFile {
                hub,
                repo,
                path: &file.path,
                revision,
            },
            staging,
            cfg,
        )
        .await?;
    }
    Ok(())
}

/// One file to pull from a Hub repo.
#[derive(Debug, Clone, Copy)]
struct HubFile<'a> {
    hub: PublicHub,
    repo: &'a str,
    path: &'a str,
    revision: &'a str,
}

async fn fetch_one_file(
    ctx: &DownloadCtx,
    client: &reqwest::Client,
    file: HubFile<'_>,
    staging: &Path,
    cfg: &AriaConfig,
) -> io::Result<()> {
    let out = staging.join(file.path);
    ensure_parent(&out)?;
    let label = format!("download {}", file.path);
    let mut last_err = None;
    for attempt in 1..=ctx.retry.attempts.max(1) {
        for url in resolve_file_urls(
            &ctx.endpoints,
            file.hub,
            file.repo,
            file.path,
            file.revision,
        ) {
            let req = apply_auth(client.get(&url), file.hub, cfg);
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
                            hub_token_field(file.hub)
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
        if attempt < ctx.retry.attempts {
            let backoff = ctx.retry.backoff * 2u32.pow(attempt - 1);
            eprintln!(
                "download: retry {attempt}/{} for {} after {}s ({})",
                ctx.retry.attempts,
                file.path,
                backoff.as_secs(),
                last_err.as_deref().unwrap_or("error")
            );
            if !backoff.is_zero() {
                tokio::time::sleep(backoff).await;
            }
        }
    }
    Err(io::Error::other(format!(
        "{}: failed to fetch {}: {}",
        file.hub.as_str(),
        file.path,
        last_err.unwrap_or_else(|| "unreachable".into())
    )))
}

async fn list_hub_files(
    ctx: &DownloadCtx,
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
            let url = format!(
                "{}/api/models/{repo}/tree/{rev_enc}?recursive=true",
                ctx.endpoints.hf
            );
            let json = get_json_paginated(client, hub, cfg, &url).await?;
            Ok(parse_hf_tree(&json))
        }
        PublicHub::ModelScope => {
            let rev_enc = urlencoding::encode(rev);
            let urls = [
                format!(
                    "{}/api/v1/models/{repo}/repo/files?Revision={rev_enc}&Recursive=true",
                    ctx.endpoints.ms
                ),
                format!(
                    "{}/api/v1/models/{repo}/repo/files?Revision={rev_enc}&Recursive=true",
                    ctx.endpoints.ms_alt
                ),
                // Some gateways want an explicit Root=/
                format!(
                    "{}/api/v1/models/{repo}/repo/files?Revision={rev_enc}&Recursive=true&Root=",
                    ctx.endpoints.ms
                ),
                format!(
                    "{}/api/v1/models/{repo}/repo/files?Revision={rev_enc}&Recursive=true&Root=",
                    ctx.endpoints.ms_alt
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
            // `urls` is never empty, so every iteration above sets `last`.
            Err(last.expect("ModelScope listing always tries at least one URL"))
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
    let show = !label.is_empty() && io::stderr().is_terminal();
    write_response_stream(resp, path, label, show).await
}

/// Bar for a known size, spinner for a chunked body, hidden when not a terminal.
fn progress_bar(total: Option<u64>, label: &str, show: bool) -> ProgressBar {
    if !show {
        return ProgressBar::hidden();
    }
    if let Some(n) = total {
        let pb = ProgressBar::new(n);
        pb.set_style(
            ProgressStyle::with_template(
                "{msg} [{bar:40.green/bright.black}] {bytes}/{total_bytes} ({bytes_per_sec}, ETA {eta})",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("█▉▊▋▌▍▎▏ "),
        );
        pb.set_message(label.to_string());
        return pb;
    }
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{msg} {spinner:.green} {bytes} ({bytes_per_sec})")
            .unwrap_or_else(|_| ProgressStyle::default_spinner()),
    );
    pb.set_message(label.to_string());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

async fn write_response_stream(
    resp: reqwest::Response,
    path: &Path,
    label: &str,
    show: bool,
) -> io::Result<u64> {
    let pb = progress_bar(resp.content_length(), label, show);

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
    ensure_parent(dest)?;
    fs::rename(staging, dest)?;
    Ok(())
}

pub fn list_models() -> io::Result<Vec<String>> {
    let dir = models_dir()?;
    if !dir.is_dir() {
        return Ok(vec![]);
    }
    let mut out = Vec::new();
    // loose files in `models/` are not checkpoints; skip them without failing
    let entries = fs::read_dir(dir)?
        .collect::<io::Result<Vec<_>>>()?
        .into_iter()
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false));
    for entry in entries {
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

fn is_downloaded_locally(models: &Path, name: &str, hub_id: Option<&str>) -> bool {
    let candidates = [
        Some(name),
        hub_id,
        hub_id.and_then(|id| id.rsplit('/').next()),
    ];
    for c in candidates.into_iter().flatten() {
        if looks_like_checkpoint(&models.join(c)) {
            return true;
        }
    }
    false
}

/// List local complete checkpoints ∪ models under the regional Hub org.
pub async fn list_models_with_hub(cfg: &AriaConfig) -> io::Result<Vec<ListedModel>> {
    list_models_with_hub_at(cfg, &DownloadCtx::from_env()).await
}

/// `list_models_with_hub` with explicit hub endpoints.
pub async fn list_models_with_hub_at(
    cfg: &AriaConfig,
    ctx: &DownloadCtx,
) -> io::Result<Vec<ListedModel>> {
    ensure_aria_home()?;
    let hub = preferred_hub(&cfg.site_url);
    let org = hub_org(hub);
    let local = list_models().unwrap_or_default();
    let mut local_set: std::collections::BTreeSet<String> = local.into_iter().collect();

    let remote = match list_hub_org_models(ctx, hub, org, cfg).await {
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
    let models = models_dir()?;

    for hub_id in remote {
        let name = short_model_name(&hub_id, org);
        let downloaded = is_downloaded_locally(&models, &name, Some(&hub_id));
        // A downloaded catalog entry is reported once, from the catalog row; drop
        // the local twin so it is not listed twice.
        if downloaded {
            local_set.remove(&name);
            local_set.remove(&hub_id);
        }
        rows.push(ListedModel {
            name,
            hub_id: Some(hub_id),
            downloaded,
        });
    }

    // Local-only checkpoints (not in the Hub org catalog).
    for name in local_set {
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
    ctx: &DownloadCtx,
    hub: PublicHub,
    org: &str,
    cfg: &AriaConfig,
) -> io::Result<Vec<String>> {
    let client = http_client(LIST_TIMEOUT).map_err(io_err)?;
    match hub {
        PublicHub::HuggingFace => list_hf_org_models(ctx, &client, org, cfg).await,
        PublicHub::ModelScope => list_ms_org_models(ctx, &client, org, cfg).await,
    }
}

async fn list_hf_org_models(
    ctx: &DownloadCtx,
    client: &reqwest::Client,
    org: &str,
    cfg: &AriaConfig,
) -> io::Result<Vec<String>> {
    let start = format!(
        "{}/api/models?author={org}&limit=100&sort=lastModified&direction=-1",
        ctx.endpoints.hf
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
    ctx: &DownloadCtx,
    client: &reqwest::Client,
    org: &str,
    cfg: &AriaConfig,
) -> io::Result<Vec<String>> {
    let mut out = Vec::new();
    let mut page: u32 = 1;
    loop {
        let url = format!(
            "{}/openapi/v1/models?owner={org}&page_size=50&page_number={page}",
            ctx.endpoints.ms
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
    cmd_list_at(&DownloadCtx::from_env()).await
}

/// `cmd_list` with explicit hub endpoints.
pub async fn cmd_list_at(ctx: &DownloadCtx) -> anyhow::Result<()> {
    let cfg = config::load_config().unwrap_or_default();
    let rows = list_models_with_hub_at(&cfg, ctx).await?;
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
            // a missing models dir yields no entries instead of an error
            let entries = fs::read_dir(&dir).into_iter().flatten().flatten();
            for entry in entries {
                if entry.file_type()?.is_dir() {
                    fs::remove_dir_all(entry.path())?;
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
    use crate::test_env::with_home;
    use serde_json::json;
    use serial_test::serial;
    use tempfile::tempdir;
    use wiremock::matchers::{header, method, path as mock_path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn hf_cfg() -> AriaConfig {
        AriaConfig {
            site_url: "https://ariacompute.com".into(),
            ..AriaConfig::default()
        }
    }

    fn ms_cfg() -> AriaConfig {
        AriaConfig {
            site_url: "https://ariacompute.cn".into(),
            modelscope_api_token: "ms-token".into(),
            ..AriaConfig::default()
        }
    }

    fn ctx_for(server: &MockServer) -> DownloadCtx {
        DownloadCtx {
            endpoints: HubEndpoints {
                hf: server.uri(),
                ms: server.uri(),
                ms_alt: server.uri(),
            },
            retry: RetryPolicy::none(),
        }
    }

    fn write_checkpoint(dir: &Path) {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join("rl_agent_config.json"), "{}").unwrap();
    }

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
        let ep = HubEndpoints::default();
        assert_eq!(
            resolve_file_urls(
                &ep,
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
                &ep,
                PublicHub::HuggingFace,
                "openbmb/MiniCPM5-2B",
                "config.json",
                DD_BASE_REVISION,
            ),
            vec![format!(
                "https://huggingface.co/openbmb/MiniCPM5-2B/resolve/{DD_BASE_REVISION}/config.json"
            )]
        );
        // empty revision falls back to the hub default branch
        assert_eq!(
            resolve_file_urls(&ep, PublicHub::HuggingFace, "r/x", "a.json", ""),
            vec!["https://huggingface.co/r/x/resolve/main/a.json".to_string()]
        );
        let ms = resolve_file_urls(
            &ep,
            PublicHub::ModelScope,
            "AriaCompute/afm-de",
            "/tokenizer/vocab.json",
            "",
        );
        assert!(ms[0].starts_with(
            "https://www.modelscope.cn/api/v1/models/AriaCompute/afm-de/repo?Revision=master&FilePath="
        ));
        assert!(ms[0].contains("tokenizer%2Fvocab.json"));
        assert!(ms
            .iter()
            .any(|u| u.contains("/resolve/master/tokenizer/vocab.json")));
    }

    #[test]
    #[serial]
    fn endpoints_can_be_overridden_by_env() {
        std::env::set_var("ARIA_HF_ENDPOINT", "http://127.0.0.1:9/hf/");
        std::env::set_var("ARIA_MS_ENDPOINT", "http://127.0.0.1:9/ms");
        let ep = HubEndpoints::from_env();
        assert_eq!(ep.hf, "http://127.0.0.1:9/hf");
        assert_eq!(ep.ms, "http://127.0.0.1:9/ms");
        assert_eq!(ep.ms_alt, "http://127.0.0.1:9/ms");

        std::env::set_var("ARIA_MS_ENDPOINT_ALT", "http://127.0.0.1:9/ms-alt");
        assert_eq!(HubEndpoints::from_env().ms_alt, "http://127.0.0.1:9/ms-alt");

        // blank values are ignored
        std::env::set_var("ARIA_HF_ENDPOINT", "   ");
        assert_eq!(HubEndpoints::from_env().hf, "https://huggingface.co");
        std::env::remove_var("ARIA_HF_ENDPOINT");
        std::env::remove_var("ARIA_MS_ENDPOINT");
        std::env::remove_var("ARIA_MS_ENDPOINT_ALT");
        assert_eq!(HubEndpoints::default().hf, "https://huggingface.co");
    }

    #[test]
    fn retry_policy_defaults() {
        let d = RetryPolicy::default();
        assert_eq!(d.attempts, FETCH_RETRIES);
        assert_eq!(d.backoff, Duration::from_secs(1));
        let n = RetryPolicy::none();
        assert_eq!(n.attempts, 1);
        assert!(n.backoff.is_zero());
    }

    #[test]
    fn afm_dd_skips_gguf_keeps_adapter() {
        assert!(keep_afm_dd_product_file("adapter_config.json"));
        assert!(keep_afm_dd_product_file("adapter_model.safetensors"));
        assert!(!keep_afm_dd_product_file("gguf/afm-dd-2b-Q8_0.gguf"));
        assert!(!keep_afm_dd_product_file("model.gguf"));
        assert!(!keep_afm_dd_product_file(".gitattributes"));
    }

    #[test]
    fn decoder_base_keeps_safetensors() {
        assert!(keep_decoder_base_file("model.safetensors"));
        assert!(keep_decoder_base_file("model-00001-of-00002.safetensors"));
        assert!(keep_decoder_base_file("model.safetensors.index.json"));
        assert!(keep_decoder_base_file("config.json"));
        assert!(keep_decoder_base_file("tokenizer.json"));
        assert!(keep_decoder_base_file("tokenizer/tokenizer_config.json"));
        assert!(keep_decoder_base_file("special_tokens_map.json"));
        assert!(keep_decoder_base_file("generation_config.json"));
        assert!(keep_decoder_base_file("chat_template.jinja"));
        assert!(!keep_decoder_base_file("README.md"));
        assert!(!keep_decoder_base_file("gguf/x.gguf"));
        assert!(!keep_decoder_base_file(".gitattributes"));
    }

    #[test]
    fn product_aliases_and_flags() {
        assert_eq!(short_product_name("afm_de"), "afm-de");
        assert_eq!(short_product_name("decoder"), "afm-dd");
        assert_eq!(short_product_name("other"), "other");
        assert!(is_afm_dd("afm-dd"));
        assert!(is_afm_dd("org/afm_dd"));
        assert!(!is_afm_dd("afm-de"));
    }

    #[test]
    fn parse_hf_keeps_nested_paths() {
        let json = json!([
            {"type": "file", "path": "model.safetensors"},
            {"type": "directory", "path": "tokenizer"},
            {"type": "file", "path": "tokenizer/vocab.json"},
            {"type": "file", "path": ".gitattributes"},
            {"type": "file", "path": ""},
            {"type": "symlink", "path": "link"},
            {"rfilename": "renamed.json"},
            {"type": "blob", "path": "dup.json"},
            {"type": "blob", "path": "dup.json"}
        ]);
        let files = parse_hf_tree(&json);
        assert_eq!(
            files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
            vec![
                "dup.json",
                "model.safetensors",
                "renamed.json",
                "tokenizer/vocab.json"
            ]
        );

        let wrapped = json!({"tree": [{"type": "file", "path": "a.json"}]});
        assert_eq!(parse_hf_tree(&wrapped)[0].path, "a.json");
        let wrapped = json!({"items": [{"path": "b.json"}]});
        assert_eq!(parse_hf_tree(&wrapped)[0].path, "b.json");
        assert!(parse_hf_tree(&json!("nope")).is_empty());
    }

    #[test]
    fn parse_ms_files_paths() {
        let json = json!({
            "Data": {
                "Files": [
                    {"Type": "blob", "Path": "rl_agent_config.json"},
                    {"Type": "tree", "Path": "encoder"},
                    {"Type": "blob", "Path": "encoder/config.json"},
                    {"Path": "by-path.json"},
                    {"Name": "by-name.json"},
                    {"name": "lower-name.json"},
                    {"Path": ""}
                ]
            }
        });
        let files = parse_ms_files(&json);
        assert_eq!(files.len(), 5);
        assert_eq!(files[0].path, "by-name.json");

        assert_eq!(
            parse_ms_files(&json!([{"Type": "blob", "Path": "direct.json"}]))[0].path,
            "direct.json"
        );
        assert_eq!(
            parse_ms_files(&json!({"data": [{"Path": "data-array.json"}]}))[0].path,
            "data-array.json"
        );
        assert_eq!(
            parse_ms_files(&json!({"data": {"files": [{"Path": "lower-files.json"}]}}))[0].path,
            "lower-files.json"
        );
        assert_eq!(
            parse_ms_files(&json!({"Files": [{"Path": "root-files.json"}]}))[0].path,
            "root-files.json"
        );
        assert!(parse_ms_files(&json!({"Data": {"Files": []}})).is_empty());
        assert!(parse_ms_files(&json!({"nope": 1})).is_empty());
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
        assert_eq!(short_model_name("ariacompute/a/b", "ariacompute"), "b");
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
    fn hub_org_names_per_hub() {
        assert_eq!(hub_org(PublicHub::HuggingFace), "ariacompute");
        assert_eq!(hub_org(PublicHub::ModelScope), "AriaCompute");
        assert_eq!(default_revision(PublicHub::HuggingFace), "main");
        assert_eq!(default_revision(PublicHub::ModelScope), "master");
        assert_eq!(hub_token_field(PublicHub::HuggingFace), "hf_token");
        assert_eq!(
            hub_token_field(PublicHub::ModelScope),
            "modelscope_api_token"
        );
    }

    #[test]
    fn next_link_parses_rel_next_only() {
        use reqwest::header::{HeaderMap, HeaderValue, LINK};

        assert_eq!(next_link(&HeaderMap::new()), None);

        let mut headers = HeaderMap::new();
        headers.insert(
            LINK,
            HeaderValue::from_static("<https://x/api?page=2>; rel=\"next\""),
        );
        assert_eq!(next_link(&headers).as_deref(), Some("https://x/api?page=2"));

        let mut headers = HeaderMap::new();
        headers.insert(
            LINK,
            HeaderValue::from_static("<https://x/api?page=2>; rel=next"),
        );
        assert_eq!(next_link(&headers).as_deref(), Some("https://x/api?page=2"));

        let mut headers = HeaderMap::new();
        headers.insert(
            LINK,
            HeaderValue::from_static(
                "<https://x/prev>; rel=\"prev\", <https://x/api?page=3>; rel=\"next\"",
            ),
        );
        assert_eq!(next_link(&headers).as_deref(), Some("https://x/api?page=3"));

        // next part without a usable target is ignored
        let mut headers = HeaderMap::new();
        headers.insert(LINK, HeaderValue::from_static("rel=\"next\""));
        assert_eq!(next_link(&headers), None);

        let mut headers = HeaderMap::new();
        headers.insert(LINK, HeaderValue::from_static("<>; rel=\"next\""));
        assert_eq!(next_link(&headers), None);

        // non-ASCII header values are rejected
        let mut headers = HeaderMap::new();
        headers.insert(
            LINK,
            HeaderValue::from_bytes(b"<\xff>; rel=\"next\"").unwrap(),
        );
        assert_eq!(next_link(&headers), None);
    }

    #[test]
    fn hf_download_builds_complete_encoder_checkpoint() {
        with_home(|home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/api/models/ariacompute/afm-de/tree/main"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                        {"type": "file", "path": "rl_agent_config.json"},
                        {"type": "file", "path": "model.safetensors"},
                        {"type": "directory", "path": "tokenizer"}
                    ])))
                    .mount(&server)
                    .await;
                Mock::given(method("GET"))
                    .and(mock_path(
                        "/ariacompute/afm-de/resolve/main/rl_agent_config.json",
                    ))
                    .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
                    .mount(&server)
                    .await;
                Mock::given(method("GET"))
                    .and(mock_path(
                        "/ariacompute/afm-de/resolve/main/model.safetensors",
                    ))
                    .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![1u8, 2, 3]))
                    .mount(&server)
                    .await;

                let ctx = ctx_for(&server);
                let dest = download_model_with_config_at("afm-de", &hf_cfg(), &ctx)
                    .await
                    .unwrap();
                assert_eq!(dest, home.join("models").join("afm-de"));
                assert!(dest.join("model.safetensors").is_file());
                assert!(!home.join("models").join("afm-de.partial").exists());

                // second call short-circuits on the existing checkpoint
                let again = download_model_with_config_at("afm-de", &hf_cfg(), &ctx)
                    .await
                    .unwrap();
                assert_eq!(again, dest);
            });
        });
    }

    #[test]
    fn auth_failure_mentions_the_setup_field() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/api/models/ariacompute/afm-de/tree/main"))
                    .respond_with(ResponseTemplate::new(401))
                    .mount(&server)
                    .await;
                let err = download_model_with_config_at("afm-de", &hf_cfg(), &ctx_for(&server))
                    .await
                    .unwrap_err();
                assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
                assert!(err.to_string().contains("hf_token"));

                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/api/v1/models/AriaCompute/afm-de/repo/files"))
                    .respond_with(ResponseTemplate::new(403))
                    .mount(&server)
                    .await;
                let err = download_model_with_config_at("afm-de", &ms_cfg(), &ctx_for(&server))
                    .await
                    .unwrap_err();
                assert!(err.to_string().contains("modelscope_api_token"));
            });
        });
    }

    #[test]
    fn list_failure_leaves_no_partial_cache() {
        with_home(|home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/api/models/ariacompute/afm-de/tree/main"))
                    .respond_with(ResponseTemplate::new(500))
                    .mount(&server)
                    .await;
                let err = download_model_with_config_at("afm-de", &hf_cfg(), &ctx_for(&server))
                    .await
                    .unwrap_err();
                assert!(err.to_string().contains("hub HTTP 500"));
                assert!(!home.join("models").join("afm-de").exists());
                assert!(!home.join("models").join("afm-de.partial").exists());
            });
        });
    }

    #[test]
    fn empty_file_list_is_an_error() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/api/models/ariacompute/afm-de/tree/main"))
                    .respond_with(
                        ResponseTemplate::new(200)
                            .set_body_json(json!([{"type": "directory", "path": "x"}])),
                    )
                    .mount(&server)
                    .await;
                let err = download_model_with_config_at("afm-de", &hf_cfg(), &ctx_for(&server))
                    .await
                    .unwrap_err();
                assert!(err.to_string().contains("empty file list"));
            });
        });
    }

    #[test]
    fn pagination_follows_the_link_header() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                let next = format!(
                    "{}/api/models/ariacompute/afm-de/tree/main?page=2",
                    server.uri()
                );
                Mock::given(method("GET"))
                    .and(mock_path("/api/models/ariacompute/afm-de/tree/main"))
                    .and(query_param("recursive", "true"))
                    .respond_with(
                        ResponseTemplate::new(200)
                            .insert_header("link", format!("<{next}>; rel=\"next\""))
                            .set_body_json(json!([{"type": "file", "path": "a.json"}])),
                    )
                    .mount(&server)
                    .await;
                Mock::given(method("GET"))
                    .and(mock_path("/api/models/ariacompute/afm-de/tree/main"))
                    .and(query_param("page", "2"))
                    .respond_with(
                        ResponseTemplate::new(200)
                            .set_body_json(json!([{"type": "file", "path": "b.json"}])),
                    )
                    .mount(&server)
                    .await;

                let client = http_client(LIST_TIMEOUT).unwrap();
                let url = format!(
                    "{}/api/models/ariacompute/afm-de/tree/main?recursive=true",
                    server.uri()
                );
                let json = get_json_paginated(&client, PublicHub::HuggingFace, &hf_cfg(), &url)
                    .await
                    .unwrap();
                let files = parse_hf_tree(&json);
                assert_eq!(
                    files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(),
                    vec!["a.json", "b.json"]
                );
            });
        });
    }

    #[test]
    fn paginated_non_array_payload_is_returned_directly() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/api/models"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!({"single": true})))
                    .mount(&server)
                    .await;
                let client = http_client(LIST_TIMEOUT).unwrap();
                let url = format!("{}/api/models", server.uri());
                let json = get_json_paginated(&client, PublicHub::HuggingFace, &hf_cfg(), &url)
                    .await
                    .unwrap();
                assert_eq!(json["single"], true);
            });
        });
    }

    #[test]
    fn retry_then_success_for_a_file() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/ariacompute/afm-de/resolve/main/a.json"))
                    .respond_with(ResponseTemplate::new(500))
                    .up_to_n_times(1)
                    .mount(&server)
                    .await;
                Mock::given(method("GET"))
                    .and(mock_path("/ariacompute/afm-de/resolve/main/a.json"))
                    .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
                    .mount(&server)
                    .await;

                let dir = tempdir().unwrap();
                let client = http_client(FETCH_TIMEOUT).unwrap();
                let ctx = DownloadCtx {
                    endpoints: HubEndpoints {
                        hf: server.uri(),
                        ms: server.uri(),
                        ms_alt: server.uri(),
                    },
                    retry: RetryPolicy {
                        attempts: 2,
                        backoff: Duration::ZERO,
                    },
                };
                fetch_one_file(
                    &ctx,
                    &client,
                    HubFile {
                        hub: PublicHub::HuggingFace,
                        repo: "ariacompute/afm-de",
                        path: "a.json",
                        revision: "main",
                    },
                    dir.path(),
                    &hf_cfg(),
                )
                .await
                .unwrap();
                assert_eq!(fs::read_to_string(dir.path().join("a.json")).unwrap(), "ok");
            });
        });
    }

    #[test]
    fn exhausted_retries_report_the_last_error() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/ariacompute/afm-de/resolve/main/a.json"))
                    .respond_with(ResponseTemplate::new(404))
                    .mount(&server)
                    .await;
                let dir = tempdir().unwrap();
                let client = http_client(FETCH_TIMEOUT).unwrap();
                let ctx = DownloadCtx {
                    endpoints: HubEndpoints {
                        hf: server.uri(),
                        ms: server.uri(),
                        ms_alt: server.uri(),
                    },
                    retry: RetryPolicy {
                        attempts: 3,
                        backoff: Duration::ZERO,
                    },
                };
                let err = fetch_one_file(
                    &ctx,
                    &client,
                    HubFile {
                        hub: PublicHub::HuggingFace,
                        repo: "ariacompute/afm-de",
                        path: "a.json",
                        revision: "main",
                    },
                    dir.path(),
                    &hf_cfg(),
                )
                .await
                .unwrap_err();
                assert!(err.to_string().contains("failed to fetch a.json"));
                assert!(err.to_string().contains("404"));
            });
        });
    }

    #[test]
    fn modelscope_falls_back_across_urls() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                // listing: the first variants 404, the last one answers
                Mock::given(method("GET"))
                    .and(mock_path("/api/v1/models/AriaCompute/afm-de/repo/files"))
                    .respond_with(ResponseTemplate::new(404))
                    .mount(&server)
                    .await;
                // resolve: only the legacy /resolve/ URL serves the bytes
                Mock::given(method("GET"))
                    .and(mock_path("/api/v1/models/AriaCompute/afm-de/repo"))
                    .respond_with(ResponseTemplate::new(404))
                    .mount(&server)
                    .await;
                Mock::given(method("GET"))
                    .and(mock_path(
                        "/models/AriaCompute/afm-de/resolve/master/a.json",
                    ))
                    .and(header("x-modelscope-authorization", "token ms-token"))
                    .respond_with(ResponseTemplate::new(200).set_body_string("ms"))
                    .mount(&server)
                    .await;

                let dir = tempdir().unwrap();
                let dest = dir.path().join("staging");
                fs::create_dir_all(&dest).unwrap();
                let client = http_client(FETCH_TIMEOUT).unwrap();
                fetch_one_file(
                    &ctx_for(&server),
                    &client,
                    HubFile {
                        hub: PublicHub::ModelScope,
                        repo: "AriaCompute/afm-de",
                        path: "a.json",
                        revision: "master",
                    },
                    &dest,
                    &ms_cfg(),
                )
                .await
                .unwrap();
                assert_eq!(fs::read_to_string(dest.join("a.json")).unwrap(), "ms");
            });
        });
    }

    #[test]
    fn modelscope_listing_uses_alt_endpoint_and_root_variant() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                let client = http_client(LIST_TIMEOUT).unwrap();
                let ctx = DownloadCtx {
                    endpoints: HubEndpoints {
                        hf: server.uri(),
                        ms: "http://127.0.0.1:1".into(),
                        ms_alt: server.uri(),
                    },
                    retry: RetryPolicy::none(),
                };
                // The plain `ms` endpoint is unreachable, the plain `ms_alt` query has no
                // matching mock (404), so only the `Root=` variant on `ms_alt` answers.
                Mock::given(method("GET"))
                    .and(mock_path("/api/v1/models/AriaCompute/afm-de/repo/files"))
                    // `Root=` (empty value) only matches through the raw query pairs.
                    .and(|req: &wiremock::Request| req.url.query_pairs().any(|(k, _)| k == "Root"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                        "Data": {"Files": [{"Type": "blob", "Path": "a.json"}]}
                    })))
                    .mount(&server)
                    .await;
                let files = list_hub_files(
                    &ctx,
                    &client,
                    PublicHub::ModelScope,
                    "AriaCompute/afm-de",
                    &ms_cfg(),
                    "master",
                )
                .await
                .unwrap();
                assert_eq!(files.len(), 1);
            });
        });
    }

    #[test]
    fn decoder_base_tries_the_next_revision() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                // pinned revision lists only config.json (no safetensors)
                Mock::given(method("GET"))
                    .and(mock_path(
                        "/api/models/openbmb/MiniCPM5-2B/tree/12a3808a956f869c767195e9266b59c4d21d92e2",
                    ))
                    .respond_with(
                        ResponseTemplate::new(200).set_body_json(json!([{"type": "file", "path": "config.json"}])),
                    )
                    .mount(&server)
                    .await;
                Mock::given(method("GET"))
                    .and(mock_path("/api/models/openbmb/MiniCPM5-2B/tree/main"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                        {"type": "file", "path": "config.json"},
                        {"type": "file", "path": "model.safetensors"}
                    ])))
                    .mount(&server)
                    .await;
                Mock::given(method("GET"))
                    .and(mock_path(
                        "/openbmb/MiniCPM5-2B/resolve/12a3808a956f869c767195e9266b59c4d21d92e2/config.json",
                    ))
                    .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
                    .mount(&server)
                    .await;
                Mock::given(method("GET"))
                    .and(mock_path(
                        "/openbmb/MiniCPM5-2B/resolve/main/config.json",
                    ))
                    .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
                    .mount(&server)
                    .await;
                Mock::given(method("GET"))
                    .and(mock_path(
                        "/openbmb/MiniCPM5-2B/resolve/main/model.safetensors",
                    ))
                    .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![0u8]))
                    .mount(&server)
                    .await;

                let dest = tempdir().unwrap();
                fetch_decoder_base(
                    &ctx_for(&server),
                    PublicHub::HuggingFace,
                    dest.path(),
                    &hf_cfg(),
                )
                .await
                .unwrap();
                assert!(dest.path().join("model.safetensors").is_file());
            });
        });
    }

    #[test]
    fn afm_dd_adapter_only_fills_the_base_directory() {
        with_home(|home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let dest = home.join("models").join("afm-dd");
                fs::create_dir_all(&dest).unwrap();
                fs::write(dest.join("adapter_config.json"), "{}").unwrap();

                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/api/models/openbmb/MiniCPM5-2B/tree/main"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                        {"type": "file", "path": "config.json"},
                        {"type": "file", "path": "model.safetensors"}
                    ])))
                    .mount(&server)
                    .await;
                Mock::given(method("GET"))
                    .and(mock_path("/openbmb/MiniCPM5-2B/resolve/main/config.json"))
                    .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
                    .mount(&server)
                    .await;
                Mock::given(method("GET"))
                    .and(mock_path(
                        "/openbmb/MiniCPM5-2B/resolve/main/model.safetensors",
                    ))
                    .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![0u8]))
                    .mount(&server)
                    .await;

                let out = download_model_with_config_at("afm-dd", &hf_cfg(), &ctx_for(&server))
                    .await
                    .unwrap();
                assert_eq!(out, dest);
                assert!(dest.join("base").join("model.safetensors").is_file());
                assert!(looks_like_checkpoint(&dest));
            });
        });
    }

    #[test]
    fn afm_dd_without_base_weights_fails_loudly() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                // product repo serves the adapter, base repo never has safetensors
                Mock::given(method("GET"))
                    .and(mock_path("/api/models/ariacompute/afm-dd/tree/main"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                        {"type": "file", "path": "adapter_config.json"}
                    ])))
                    .mount(&server)
                    .await;
                Mock::given(method("GET"))
                    .and(mock_path(
                        "/ariacompute/afm-dd/resolve/main/adapter_config.json",
                    ))
                    .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
                    .mount(&server)
                    .await;
                for rev in ["12a3808a956f869c767195e9266b59c4d21d92e2", "main"] {
                    Mock::given(method("GET"))
                        .and(mock_path(format!(
                            "/api/models/openbmb/MiniCPM5-2B/tree/{rev}"
                        )))
                        .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                            {"type": "file", "path": "config.json"}
                        ])))
                        .mount(&server)
                        .await;
                    Mock::given(method("GET"))
                        .and(mock_path(format!(
                            "/openbmb/MiniCPM5-2B/resolve/{rev}/config.json"
                        )))
                        .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
                        .mount(&server)
                        .await;
                }

                let err = download_model_with_config_at("afm-dd", &hf_cfg(), &ctx_for(&server))
                    .await
                    .unwrap_err();
                assert!(err.to_string().contains("no safetensors"));
            });
        });
    }

    #[test]
    fn list_merges_remote_and_local_rows() {
        with_home(|home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                write_checkpoint(&home.join("models").join("afm-de"));
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/api/models"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                        {"id": "ariacompute/afm-de"},
                        {"id": "ariacompute/afm-dd"},
                        {"id": "someone/else"},
                        {"modelId": "ariacompute/afm-xx"},
                        {"id": ""}
                    ])))
                    .mount(&server)
                    .await;

                let rows = list_models_with_hub_at(&hf_cfg(), &ctx_for(&server))
                    .await
                    .unwrap();
                let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
                assert_eq!(names, vec!["afm-dd", "afm-de", "afm-xx"]);
                assert!(!rows[0].downloaded);
                assert!(rows[1].downloaded);
            });
        });
    }

    #[test]
    fn list_degrades_to_local_when_hub_fails() {
        with_home(|home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                write_checkpoint(&home.join("models").join("afm-de"));
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/api/models"))
                    .respond_with(ResponseTemplate::new(503))
                    .mount(&server)
                    .await;
                let rows = list_models_with_hub_at(&hf_cfg(), &ctx_for(&server))
                    .await
                    .unwrap();
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].name, "afm-de");
                assert!(rows[0].downloaded);
                assert!(rows[0].hub_id.is_none());
            });
        });
    }

    #[test]
    fn list_modelscope_org_pages_until_exhausted() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/openapi/v1/models"))
                    .and(query_param("page_number", "1"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                        "data": {
                            "models": (0..50).map(|i| json!({"id": format!("m{i}")})).collect::<Vec<_>>(),
                            "total_count": 51
                        }
                    })))
                    .mount(&server)
                    .await;
                Mock::given(method("GET"))
                    .and(mock_path("/openapi/v1/models"))
                    .and(query_param("page_number", "2"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                        "data": {"models": [{"path": "AriaCompute/last"}], "total_count": 51}
                    })))
                    .mount(&server)
                    .await;

                let client = http_client(LIST_TIMEOUT).unwrap();
                let ids = list_ms_org_models(&ctx_for(&server), &client, "AriaCompute", &ms_cfg())
                    .await
                    .unwrap();
                assert_eq!(ids.len(), 51);
                assert!(ids.contains(&"AriaCompute/last".to_string()));
            });
        });
    }

    #[test]
    fn cmd_list_prints_rows_and_empty_marker() {
        with_home(|home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/api/models"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!([])))
                    .mount(&server)
                    .await;
                cmd_list_at(&ctx_for(&server)).await.unwrap();

                write_checkpoint(&home.join("models").join("afm-de"));
                cmd_list_at(&ctx_for(&server)).await.unwrap();
            });
        });
    }

    #[test]
    #[serial]
    fn local_cache_helpers_list_check_and_clean() {
        with_home(|home| {
            let models = home.join("models");
            fs::create_dir_all(&models).unwrap();
            assert!(list_models().unwrap().is_empty());

            write_checkpoint(&models.join("afm-de"));
            let junk = models.join("junk");
            fs::create_dir_all(&junk).unwrap();
            assert_eq!(list_models().unwrap(), vec!["afm-de".to_string()]);
            assert!(!junk.exists());

            check_model(Some("afm-de")).unwrap();
            check_model(Some("missing")).unwrap();
            check_model(None).unwrap();

            clean_model(Some("afm-de")).unwrap();
            assert!(!models.join("afm-de").exists());
            clean_model(Some("afm-de")).unwrap();
            write_checkpoint(&models.join("afm-de"));
            clean_model(None).unwrap();
            assert!(list_models().unwrap().is_empty());
        });
    }

    #[test]
    #[serial]
    fn check_model_without_hub_rows_reports_empty() {
        with_home(|home| {
            fs::create_dir_all(home.join("models")).unwrap();
            check_model(None).unwrap();
        });
    }

    #[test]
    fn progress_renderers_cover_bar_and_spinner() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // content-length known → progress bar branch
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(mock_path("/with-length"))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![7u8; 8]))
                .mount(&server)
                .await;
            let dir = tempdir().unwrap();
            let client = http_client(FETCH_TIMEOUT).unwrap();
            let resp = client
                .get(format!("{}/with-length", server.uri()))
                .send()
                .await
                .unwrap();
            let n = write_response_stream(resp, &dir.path().join("a.bin"), "label", true)
                .await
                .unwrap();
            assert_eq!(n, 8);

            // unknown length → spinner branch
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(mock_path("/chunked"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_raw(vec![5u8; 4], "application/octet-stream"),
                )
                .mount(&server)
                .await;
            let resp = client
                .get(format!("{}/chunked", server.uri()))
                .send()
                .await
                .unwrap();
            let n = write_response_stream(resp, &dir.path().join("b.bin"), "label", true)
                .await
                .unwrap();
            assert_eq!(n, 4);
        });
    }

    #[test]
    fn stream_without_label_hides_progress() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(mock_path("/quiet"))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![1u8, 2]))
                .mount(&server)
                .await;
            let dir = tempdir().unwrap();
            let client = http_client(FETCH_TIMEOUT).unwrap();
            let resp = client
                .get(format!("{}/quiet", server.uri()))
                .send()
                .await
                .unwrap();
            let out = dir.path().join("q.bin");
            let n = stream_response_to_file(resp, &out, "").await.unwrap();
            assert_eq!(n, 2);
            assert_eq!(fs::read(&out).unwrap(), vec![1, 2]);
        });
    }

    #[test]
    fn hub_token_from_config_not_env() {
        std::env::set_var("HF_TOKEN", "env-hf");
        std::env::set_var("MODELSCOPE_API_TOKEN", "env-ms");
        let empty = AriaConfig::default();
        assert_eq!(hub_token(PublicHub::HuggingFace, &empty), None);
        assert_eq!(hub_token(PublicHub::ModelScope, &empty), None);
        let cfg = AriaConfig {
            hf_token: "  hf_yml  ".into(),
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

        // adapter/ subdir marker and merged/ base layout also count
        let nested = tmp.path().join("nested");
        fs::create_dir_all(nested.join("adapter")).unwrap();
        fs::write(nested.join("adapter/adapter_config.json"), "{}").unwrap();
        assert!(has_decoder_adapter(&nested));
        let merged = nested.join("merged");
        fs::create_dir_all(&merged).unwrap();
        fs::write(merged.join("config.json"), "{}").unwrap();
        fs::write(merged.join("model.safetensors.index.json"), "{}").unwrap();
        assert!(has_decoder_base_weights(&nested));
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

        // a file (not a dir) is left alone
        let file = tmp.path().join("file");
        fs::write(&file, b"x").unwrap();
        remove_incomplete_cache(&file);
        assert!(file.exists());
    }

    #[test]
    fn atomic_replace_overwrites_destination() {
        let tmp = tempdir().unwrap();
        let staging = tmp.path().join("staging");
        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("new.txt"), b"n").unwrap();
        let dest = tmp.path().join("dest");
        fs::create_dir_all(&dest).unwrap();
        fs::write(dest.join("old.txt"), b"o").unwrap();
        atomic_replace(&staging, &dest).unwrap();
        assert!(dest.join("new.txt").is_file());
        assert!(!dest.join("old.txt").exists());
        assert!(!staging.exists());
    }

    #[test]
    fn llama_dir_detection_covers_scan_and_index() {
        let tmp = tempdir().unwrap();
        let p = tmp.path().join("d");
        fs::create_dir_all(&p).unwrap();
        assert!(!has_llama_safetensors(&p));
        fs::write(p.join("config.json"), "{}").unwrap();
        assert!(!has_llama_safetensors(&p));
        fs::write(p.join("shard-1.safetensors"), b"x").unwrap();
        assert!(has_llama_safetensors(&p));

        let indexed = tmp.path().join("i");
        fs::create_dir_all(&indexed).unwrap();
        fs::write(indexed.join("config.json"), "{}").unwrap();
        fs::write(indexed.join("model.safetensors.index.json"), "{}").unwrap();
        assert!(has_llama_safetensors(&indexed));

        // adapter/lora-only trees are not base weights
        let only_adapter = tmp.path().join("a");
        fs::create_dir_all(&only_adapter).unwrap();
        fs::write(only_adapter.join("config.json"), "{}").unwrap();
        fs::write(only_adapter.join("adapter_model.safetensors"), b"x").unwrap();
        fs::write(only_adapter.join("lora_x.safetensors"), b"x").unwrap();
        assert!(!has_llama_safetensors(&only_adapter));
    }

    #[test]
    fn hf_token_is_sent_as_bearer() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/ariacompute/afm-de/resolve/main/a.json"))
                    .and(header("authorization", "Bearer hf-secret"))
                    .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
                    .mount(&server)
                    .await;
                let cfg = AriaConfig {
                    site_url: "https://ariacompute.com".into(),
                    hf_token: "hf-secret".into(),
                    ..AriaConfig::default()
                };
                let dir = tempdir().unwrap();
                let client = http_client(FETCH_TIMEOUT).unwrap();
                fetch_one_file(
                    &ctx_for(&server),
                    &client,
                    HubFile {
                        hub: PublicHub::HuggingFace,
                        repo: "ariacompute/afm-de",
                        path: "a.json",
                        revision: "main",
                    },
                    dir.path(),
                    &cfg,
                )
                .await
                .unwrap();
                assert_eq!(fs::read_to_string(dir.path().join("a.json")).unwrap(), "ok");
            });
        });
    }

    #[test]
    fn incomplete_repo_layout_is_an_error_without_cache() {
        with_home(|home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/api/models/ariacompute/afm-de/tree/main"))
                    .respond_with(
                        ResponseTemplate::new(200)
                            .set_body_json(json!([{"type": "file", "path": "README.md"}])),
                    )
                    .mount(&server)
                    .await;
                Mock::given(method("GET"))
                    .and(mock_path("/ariacompute/afm-de/resolve/main/README.md"))
                    .respond_with(ResponseTemplate::new(200).set_body_string("hi"))
                    .mount(&server)
                    .await;

                let err = download_model_with_config_at("afm-de", &hf_cfg(), &ctx_for(&server))
                    .await
                    .unwrap_err();
                assert_eq!(err.kind(), io::ErrorKind::InvalidData);
                assert!(err.to_string().contains("incomplete checkpoint"));
                assert!(!home.join("models/afm-de").exists());
                assert!(!home.join("models/afm-de.partial").exists());
            });
        });
    }

    #[test]
    fn checkout_body_write_failure_is_reported() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/ariacompute/afm-de/resolve/main/a.json"))
                    .respond_with(ResponseTemplate::new(200).set_body_string("xyz"))
                    .mount(&server)
                    .await;
                let dir = tempdir().unwrap();
                // a directory in the destination slot makes File::create fail
                fs::create_dir_all(dir.path().join("a.json")).unwrap();

                let client = http_client(FETCH_TIMEOUT).unwrap();
                let err = fetch_one_file(
                    &ctx_for(&server),
                    &client,
                    HubFile {
                        hub: PublicHub::HuggingFace,
                        repo: "ariacompute/afm-de",
                        path: "a.json",
                        revision: "main",
                    },
                    dir.path(),
                    &hf_cfg(),
                )
                .await
                .unwrap_err();
                assert!(err.to_string().contains("body:"), "{err}");
            });
        });
    }

    #[test]
    fn all_urls_unreachable_is_reported() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let dir = tempdir().unwrap();
                let client = http_client(FETCH_TIMEOUT).unwrap();
                let ctx = DownloadCtx {
                    endpoints: HubEndpoints {
                        hf: "http://127.0.0.1:1".into(),
                        ms: "http://127.0.0.1:1".into(),
                        ms_alt: "http://127.0.0.1:1".into(),
                    },
                    retry: RetryPolicy::none(),
                };
                let err = fetch_one_file(
                    &ctx,
                    &client,
                    HubFile {
                        hub: PublicHub::HuggingFace,
                        repo: "ariacompute/afm-de",
                        path: "a.json",
                        revision: "main",
                    },
                    dir.path(),
                    &hf_cfg(),
                )
                .await
                .unwrap_err();
                assert!(err.to_string().contains("failed to fetch a.json"));
                assert!(err.to_string().contains("send:"), "{err}");
            });
        });
    }

    #[test]
    fn retry_backoff_is_honoured_before_giving_up() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/ariacompute/afm-de/resolve/main/a.json"))
                    .respond_with(ResponseTemplate::new(500))
                    .expect(2)
                    .mount(&server)
                    .await;
                let dir = tempdir().unwrap();
                let ctx = DownloadCtx {
                    endpoints: HubEndpoints {
                        hf: server.uri(),
                        ms: server.uri(),
                        ms_alt: server.uri(),
                    },
                    retry: RetryPolicy {
                        attempts: 2,
                        backoff: Duration::from_millis(1),
                    },
                };
                let client = http_client(FETCH_TIMEOUT).unwrap();
                let err = fetch_one_file(
                    &ctx,
                    &client,
                    HubFile {
                        hub: PublicHub::HuggingFace,
                        repo: "ariacompute/afm-de",
                        path: "a.json",
                        revision: "main",
                    },
                    dir.path(),
                    &hf_cfg(),
                )
                .await
                .unwrap_err();
                assert!(err.to_string().contains("HTTP 500"), "{err}");
            });
        });
    }

    #[test]
    fn empty_revision_falls_back_per_hub() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/api/models/ariacompute/afm-de/tree/main"))
                    .respond_with(
                        ResponseTemplate::new(200)
                            .set_body_json(json!([{"type": "file", "path": "a.json"}])),
                    )
                    .mount(&server)
                    .await;
                Mock::given(method("GET"))
                    .and(mock_path("/api/v1/models/AriaCompute/afm-de/repo/files"))
                    .respond_with(
                        ResponseTemplate::new(200)
                            .set_body_json(json!({"Data": {"Files": [{"Path": "a.json"}]}})),
                    )
                    .mount(&server)
                    .await;

                let client = http_client(LIST_TIMEOUT).unwrap();
                let hf = list_hub_files(
                    &ctx_for(&server),
                    &client,
                    PublicHub::HuggingFace,
                    "ariacompute/afm-de",
                    &hf_cfg(),
                    "",
                )
                .await
                .unwrap();
                assert_eq!(hf.len(), 1);

                let ms = list_hub_files(
                    &ctx_for(&server),
                    &client,
                    PublicHub::ModelScope,
                    "AriaCompute/afm-de",
                    &ms_cfg(),
                    "",
                )
                .await
                .unwrap();
                assert_eq!(ms.len(), 1);
            });
        });
    }

    #[test]
    fn pagination_breaks_when_a_page_is_not_an_array() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                let page2 = format!("{}/api/models?page=2", server.uri());
                Mock::given(method("GET"))
                    .and(mock_path("/api/models"))
                    .and(query_param("page", "1"))
                    .respond_with(
                        ResponseTemplate::new(200)
                            .insert_header("link", format!("<{page2}>; rel=\"next\""))
                            .set_body_json(json!([{"id": "ariacompute/afm-de"}])),
                    )
                    .mount(&server)
                    .await;
                Mock::given(method("GET"))
                    .and(mock_path("/api/models"))
                    .and(query_param("page", "2"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
                    .mount(&server)
                    .await;

                let ctx = ctx_for(&server);
                let client = http_client(LIST_TIMEOUT).unwrap();
                let json = get_json_paginated(
                    &client,
                    PublicHub::HuggingFace,
                    &hf_cfg(),
                    &format!("{}/api/models?page=1", server.uri()),
                )
                .await
                .unwrap();
                assert_eq!(json.as_array().unwrap().len(), 1);
                let _ = ctx;
            });
        });
    }

    #[test]
    fn model_scope_org_listing_stops_at_the_page_cap() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                let models: Vec<Value> =
                    (0..50).map(|i| json!({"id": format!("afm-{i}")})).collect();
                Mock::given(method("GET"))
                    .and(mock_path("/openapi/v1/models"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                        "data": {"models": models, "total_count": 10_000}
                    })))
                    .mount(&server)
                    .await;

                let ctx = ctx_for(&server);
                let client = http_client(LIST_TIMEOUT).unwrap();
                let ids = list_ms_org_models(&ctx, &client, "AriaCompute", &ms_cfg())
                    .await
                    .unwrap();
                // every page repeats the same 50 ids, deduped down to 50; the
                // `page > 50` cap stops the loop after 50 requests
                assert_eq!(ids.len(), 50);
                assert_eq!(server.received_requests().await.unwrap().len(), 50);
            });
        });
    }

    #[test]
    fn model_scope_decoder_base_uses_the_ms_repo() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/api/v1/models/OpenBMB/MiniCPM5-2B/repo/files"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                        "Data": {"Files": [{"Path": "config.json"}, {"Path": "model.safetensors"}]}
                    })))
                    .mount(&server)
                    .await;
                for file in ["config.json", "model.safetensors"] {
                    Mock::given(method("GET"))
                        .and(mock_path("/api/v1/models/OpenBMB/MiniCPM5-2B/repo"))
                        .and(query_param("FilePath", file))
                        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"{}".to_vec()))
                        .mount(&server)
                        .await;
                }
                let dir = tempdir().unwrap();
                fetch_decoder_base(
                    &ctx_for(&server),
                    PublicHub::ModelScope,
                    dir.path(),
                    &ms_cfg(),
                )
                .await
                .unwrap();
                assert!(dir.path().join("model.safetensors").is_file());
            });
        });
    }

    #[test]
    fn progress_bar_variants() {
        let hidden = progress_bar(None, "l", false);
        assert!(hidden.is_hidden());
        let sized = progress_bar(Some(10), "l", true);
        assert_eq!(sized.length(), Some(10));
        let spinner = progress_bar(None, "l", true);
        assert_eq!(spinner.length(), None);
    }

    #[test]
    fn downloaded_local_check_accepts_short_and_full_ids() {
        let dir = tempdir().unwrap();
        let models = dir.path().join("models");
        write_checkpoint(&models.join("afm-de"));
        assert!(is_downloaded_locally(&models, "afm-de", None));
        assert!(is_downloaded_locally(
            &models,
            "other",
            Some("ariacompute/afm-de")
        ));
        assert!(!is_downloaded_locally(&models, "nope", None));
    }

    #[test]
    fn list_models_without_a_models_dir_is_empty() {
        with_home(|_home| {
            assert!(list_models().unwrap().is_empty());
        });
    }

    #[test]
    fn auth_failure_names_the_setup_field() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/ariacompute/afm-de/resolve/main/a.json"))
                    .respond_with(ResponseTemplate::new(403))
                    .mount(&server)
                    .await;
                let dir = tempdir().unwrap();
                let client = http_client(FETCH_TIMEOUT).unwrap();
                let err = fetch_one_file(
                    &ctx_for(&server),
                    &client,
                    HubFile {
                        hub: PublicHub::HuggingFace,
                        repo: "ariacompute/afm-de",
                        path: "a.json",
                        revision: "main",
                    },
                    dir.path(),
                    &hf_cfg(),
                )
                .await
                .unwrap_err();
                assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
                assert!(err.to_string().contains("auth failed HTTP 403"));
                assert!(err.to_string().contains("hf_token"));
            });
        });
    }

    #[test]
    fn empty_modelscope_listing_across_all_urls_is_an_error() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/api/v1/models/AriaCompute/afm-de/repo/files"))
                    .respond_with(
                        ResponseTemplate::new(200).set_body_json(json!({"Data": {"Files": []}})),
                    )
                    .mount(&server)
                    .await;
                let client = http_client(LIST_TIMEOUT).unwrap();
                let err = list_hub_files(
                    &ctx_for(&server),
                    &client,
                    PublicHub::ModelScope,
                    "AriaCompute/afm-de",
                    &ms_cfg(),
                    "master",
                )
                .await
                .unwrap_err();
                assert!(
                    err.to_string().contains("empty ModelScope listing"),
                    "{err}"
                );
            });
        });
    }

    #[test]
    #[serial_test::serial]
    fn list_models_with_hub_uses_env_endpoints() {
        with_home(|home| {
            write_checkpoint(&home.join("models/afm-de"));
            std::env::set_var("ARIA_HF_ENDPOINT", "http://127.0.0.1:1");
            let rt = tokio::runtime::Runtime::new().unwrap();
            let rows = rt.block_on(list_models_with_hub(&hf_cfg())).unwrap();
            std::env::remove_var("ARIA_HF_ENDPOINT");
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].name, "afm-de");
            assert!(rows[0].downloaded);
            assert!(rows[0].hub_id.is_none());
        });
    }

    #[test]
    fn list_modelscope_hub_branch_is_used_for_cn_sites() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/openapi/v1/models"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                        "data": {"models": [{"id": "afm-de"}], "total_count": 1}
                    })))
                    .mount(&server)
                    .await;
                let cfg = AriaConfig {
                    site_url: "https://ariacompute.cn".into(),
                    ..AriaConfig::default()
                };
                let rows = list_models_with_hub_at(&cfg, &ctx_for(&server))
                    .await
                    .unwrap();
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].name, "afm-de");
                assert!(!rows[0].downloaded);
            });
        });
    }

    #[test]
    fn hf_org_listing_ignores_non_arrays_and_blank_ids() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/api/models"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!({"nope": 1})))
                    .mount(&server)
                    .await;
                let client = http_client(LIST_TIMEOUT).unwrap();
                let ids = list_hf_org_models(&ctx_for(&server), &client, "ariacompute", &hf_cfg())
                    .await
                    .unwrap();
                assert!(ids.is_empty());
            });
        });
    }

    #[test]
    fn ms_org_listing_skips_blank_ids_and_empty_pages() {
        with_home(|_home| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/openapi/v1/models"))
                    .and(query_param("page_number", "1"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                        "data": {"models": [{"id": ""}, {"id": "AriaCompute/afm-de"}], "total_count": 2}
                    })))
                    .mount(&server)
                    .await;
                let client = http_client(LIST_TIMEOUT).unwrap();
                let ids = list_ms_org_models(&ctx_for(&server), &client, "AriaCompute", &ms_cfg())
                    .await
                    .unwrap();
                assert_eq!(ids, vec!["AriaCompute/afm-de"]);

                // an empty page terminates the loop
                let empty = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/openapi/v1/models"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                        "data": {"models": [], "total_count": 9}
                    })))
                    .mount(&empty)
                    .await;
                let ctx = ctx_for(&empty);
                let ids = list_ms_org_models(&ctx, &client, "AriaCompute", &ms_cfg())
                    .await
                    .unwrap();
                assert!(ids.is_empty());
            });
        });
    }

    #[test]
    fn clean_cache_removes_dirs_and_keeps_loose_files() {
        with_home(|home| {
            write_checkpoint(&home.join("models/afm-de"));
            fs::write(home.join("models/loose.txt"), b"x").unwrap();
            // loose files are ignored by the listing but survive `clean`
            assert_eq!(list_models().unwrap(), vec!["afm-de".to_string()]);
            clean_model(None).unwrap();
            assert!(!home.join("models/afm-de").exists());
            assert!(home.join("models/loose.txt").is_file());
        });
    }

    #[test]
    fn clean_cache_tolerates_a_missing_models_dir() {
        with_home(|home| {
            // no `models/` at all: `clean` reports success instead of erroring
            clean_model(None).unwrap();
            assert!(!home.join("models").exists());
        });
    }

    #[test]
    fn ensure_parent_creates_dirs_and_tolerates_roots() {
        let dir = tempdir().unwrap();
        let nested = dir.path().join("a/b/c.json");
        ensure_parent(&nested).unwrap();
        assert!(dir.path().join("a/b").is_dir());
        // a filesystem root has no parent
        ensure_parent(Path::new("/")).unwrap();
    }
}
