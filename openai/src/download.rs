//! Probe Dashboard / Hugging Face / ModelScope and download Aria bundles.

use crate::config::{self, AriaConfig};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use zip::ZipArchive;

const DEFAULT_SDK: &str = "v1.0";
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);
const PROBE_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadSource {
    Dashboard,
    HuggingFace,
    ModelScope,
}

impl DownloadSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dashboard => "dashboard",
            Self::HuggingFace => "huggingface",
            Self::ModelScope => "modelscope",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleRef {
    pub model: String,
    pub slug: String,
    pub quant: String,
    pub sdk: String,
}

/// Parse `<model>` → slug + quant. `*_q4`→int4, `*_q8`→int8, `*_q326`/`*_q3.26`→int326.
pub fn parse_bundle_name(model: &str) -> BundleRef {
    let model = model.trim().trim_end_matches('/').to_string();
    let lower = model.to_ascii_lowercase();
    let (slug, quant) = if let Some(base) = lower.strip_suffix("_q3.26") {
        (model[..base.len()].to_string(), "int326".into())
    } else if let Some(base) = lower.strip_suffix("_q326") {
        (model[..base.len()].to_string(), "int326".into())
    } else if let Some(base) = lower.strip_suffix("_q8") {
        (model[..base.len()].to_string(), "int8".into())
    } else if let Some(base) = lower.strip_suffix("_q4") {
        (model[..base.len()].to_string(), "int4".into())
    } else {
        (model.clone(), "int4".into())
    };
    BundleRef {
        model,
        slug,
        quant,
        sdk: DEFAULT_SDK.into(),
    }
}

#[derive(Debug, Clone)]
pub struct ProbeResult {
    pub source: DownloadSource,
    pub reachable: bool,
    pub bytes_per_sec: f64,
    pub detail: String,
}

#[derive(Debug, Deserialize)]
struct DashboardJsonMeta {
    mode: String,
    url: String,
    #[serde(default)]
    filename: String,
}

pub async fn download_model(model: &str, cfg: &AriaConfig) -> io::Result<PathBuf> {
    let bundle = parse_bundle_name(model);
    config::ensure_aria_home()?;
    let dest = config::model_dir(&bundle.model)?;
    if is_valid_bundle(&dest) {
        eprintln!("download: already present at {}", dest.display());
        return Ok(dest);
    }

    let ranked = probe_and_rank(&bundle, cfg).await;
    let usable: Vec<_> = ranked.into_iter().filter(|p| p.reachable).collect();
    if usable.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotConnected,
            "no download source reachable (dashboard / huggingface / modelscope)",
        ));
    }

    let mut last_err = None;
    for probe in &usable {
        eprintln!(
            "download: using {} ({})",
            probe.source.as_str(),
            probe.detail
        );
        match fetch_into(&bundle, cfg, probe.source, &dest).await {
            Ok(()) => {
                if is_valid_bundle(&dest) {
                    return Ok(dest);
                }
                let _ = fs::remove_dir_all(&dest);
                last_err = Some(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "{} fetch completed but bundle invalid (need weight.bin + aria-quant-bundle config.json)",
                        probe.source.as_str()
                    ),
                ));
            }
            Err(e) => {
                let _ = fs::remove_dir_all(&dest);
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| io::Error::other("download failed for all sources")))
}

pub async fn probe_and_rank(bundle: &BundleRef, cfg: &AriaConfig) -> Vec<ProbeResult> {
    let mut results = Vec::new();
    if !cfg.cloud_api_key.is_empty() && !cfg.site_url.is_empty() {
        results.push(probe_dashboard(bundle, cfg).await);
    }
    results.push(probe_hub(DownloadSource::HuggingFace, bundle).await);
    results.push(probe_hub(DownloadSource::ModelScope, bundle).await);

    results.sort_by(|a, b| {
        b.reachable.cmp(&a.reachable).then(
            b.bytes_per_sec
                .partial_cmp(&a.bytes_per_sec)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });
    results
}

async fn probe_dashboard(bundle: &BundleRef, cfg: &AriaConfig) -> ProbeResult {
    let url = dashboard_meta_url(cfg, bundle);
    let client = match http_client(PROBE_TIMEOUT) {
        Ok(c) => c,
        Err(e) => {
            return ProbeResult {
                source: DownloadSource::Dashboard,
                reachable: false,
                bytes_per_sec: 0.0,
                detail: e.to_string(),
            }
        }
    };
    let start = Instant::now();
    let resp = client
        .get(&url)
        .bearer_auth(&cfg.cloud_api_key)
        .header("Accept", "application/json")
        .send()
        .await;
    match resp {
        Ok(r) if r.status().is_success() => {
            let elapsed = start.elapsed().as_secs_f64().max(1e-3);
            let bytes = r.content_length().unwrap_or(256) as f64;
            let bps = bytes / elapsed;
            ProbeResult {
                source: DownloadSource::Dashboard,
                reachable: true,
                bytes_per_sec: bps,
                detail: format!("{:.2} MB/s", bps / (1024.0 * 1024.0)),
            }
        }
        Ok(r) => ProbeResult {
            source: DownloadSource::Dashboard,
            reachable: false,
            bytes_per_sec: 0.0,
            detail: format!("HTTP {}", r.status()),
        },
        Err(e) => ProbeResult {
            source: DownloadSource::Dashboard,
            reachable: false,
            bytes_per_sec: 0.0,
            detail: e.to_string(),
        },
    }
}

async fn probe_hub(source: DownloadSource, bundle: &BundleRef) -> ProbeResult {
    let urls = hub_config_urls(source, bundle);
    let client = match http_client(PROBE_TIMEOUT) {
        Ok(c) => c,
        Err(e) => {
            return ProbeResult {
                source,
                reachable: false,
                bytes_per_sec: 0.0,
                detail: e.to_string(),
            }
        }
    };
    for url in urls {
        let start = Instant::now();
        let mut req = client.get(&url);
        if let Some(token) = hub_token(source) {
            req = req.bearer_auth(token);
        }
        // Prefer Range to cap probe size.
        req = req.header("Range", format!("bytes=0-{}", PROBE_BYTES.saturating_sub(1)));
        match req.send().await {
            Ok(r) if r.status().is_success() || r.status().as_u16() == 206 => {
                let bytes = match r.bytes().await {
                    Ok(b) => b.len(),
                    Err(_) => 0,
                };
                if bytes == 0 {
                    continue;
                }
                let elapsed = start.elapsed().as_secs_f64().max(1e-3);
                let bps = bytes as f64 / elapsed;
                return ProbeResult {
                    source,
                    reachable: true,
                    bytes_per_sec: bps,
                    detail: format!("{:.2} MB/s", bps / (1024.0 * 1024.0)),
                };
            }
            Ok(r) if r.status().as_u16() == 401 || r.status().as_u16() == 403 => {
                return ProbeResult {
                    source,
                    reachable: false,
                    bytes_per_sec: 0.0,
                    detail: format!("auth failed HTTP {}", r.status()),
                };
            }
            _ => continue,
        }
    }
    ProbeResult {
        source,
        reachable: false,
        bytes_per_sec: 0.0,
        detail: "unreachable".into(),
    }
}

async fn fetch_into(
    bundle: &BundleRef,
    cfg: &AriaConfig,
    source: DownloadSource,
    dest: &Path,
) -> io::Result<()> {
    match source {
        DownloadSource::Dashboard => fetch_dashboard(bundle, cfg, dest).await,
        DownloadSource::HuggingFace | DownloadSource::ModelScope => {
            fetch_hub(source, bundle, dest).await
        }
    }
}

async fn fetch_dashboard(bundle: &BundleRef, cfg: &AriaConfig, dest: &Path) -> io::Result<()> {
    let meta_url = dashboard_meta_url(cfg, bundle);
    let client = http_client(Duration::from_secs(600)).map_err(io_err)?;
    let meta_resp = client
        .get(&meta_url)
        .bearer_auth(&cfg.cloud_api_key)
        .header("Accept", "application/json")
        .send()
        .await
        .map_err(io_err)?;
    if !meta_resp.status().is_success() {
        return Err(io::Error::other(format!(
            "dashboard metadata HTTP {}",
            meta_resp.status()
        )));
    }
    let meta: DashboardJsonMeta = meta_resp.json().await.map_err(io_err)?;

    let staging = dest.with_extension("partial");
    let _ = fs::remove_dir_all(&staging);
    fs::create_dir_all(&staging)?;
    let zip_path = staging.join("bundle.zip");

    let resp = client
        .get(&meta.url)
        .bearer_auth(&cfg.cloud_api_key)
        .send()
        .await
        .map_err(io_err)?
        .error_for_status()
        .map_err(io_err)?;
    stream_response_to_file(resp, &zip_path, &format!("download {}", bundle.model)).await?;

    // Confirm zip magic after stream (redirect/zip modes both deliver a zip body).
    let mut magic = [0u8; 4];
    {
        let mut f = fs::File::open(&zip_path)?;
        let n = f.read(&mut magic)?;
        if n < 4 || !looks_like_zip(&magic) {
            let _ = fs::remove_dir_all(&staging);
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "dashboard returned mode={} (expected zip for Aria bundle); filename={}",
                    meta.mode, meta.filename
                ),
            ));
        }
    }
    extract_zip_path(&zip_path, &staging)?;
    let _ = fs::remove_file(&zip_path);
    atomic_replace(&staging, dest)?;
    Ok(())
}

async fn fetch_hub(source: DownloadSource, bundle: &BundleRef, dest: &Path) -> io::Result<()> {
    let files = ["config.json", "weight.bin"];
    let staging = dest.with_extension("partial");
    let _ = fs::remove_dir_all(&staging);
    fs::create_dir_all(&staging)?;
    let client = http_client(Duration::from_secs(600)).map_err(io_err)?;

    for file in files {
        let mut fetched = false;
        for url in hub_file_urls(source, bundle, file) {
            let mut req = client.get(&url);
            if let Some(token) = hub_token(source) {
                req = req.bearer_auth(token);
            }
            match req.send().await {
                Ok(r) if r.status().is_success() => {
                    let out = staging.join(file);
                    let label = format!("download {} ({file})", bundle.model);
                    stream_response_to_file(r, &out, &label).await?;
                    fetched = true;
                    break;
                }
                _ => continue,
            }
        }
        if !fetched {
            let _ = fs::remove_dir_all(&staging);
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("{}: missing {file}", source.as_str()),
            ));
        }
    }
    // Best-effort tokenizer sidecars (optional; quiet, no progress).
    for extra in [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ] {
        for url in hub_file_urls(source, bundle, extra) {
            let mut req = client.get(&url);
            if let Some(token) = hub_token(source) {
                req = req.bearer_auth(token);
            }
            if let Ok(r) = req.send().await {
                if r.status().is_success() {
                    let out = staging.join(extra);
                    let _ = stream_response_to_file(r, &out, "").await;
                    break;
                }
            }
        }
    }
    atomic_replace(&staging, dest)?;
    Ok(())
}

fn dashboard_meta_url(cfg: &AriaConfig, bundle: &BundleRef) -> String {
    let base = cfg.site_url.trim_end_matches('/');
    format!(
        "{base}/api/dashboard/models/{}/download?quant={}&sdk={}&format=json",
        urlencoding::encode(&bundle.slug),
        urlencoding::encode(&bundle.quant),
        urlencoding::encode(&bundle.sdk),
    )
}

fn hub_config_urls(source: DownloadSource, bundle: &BundleRef) -> Vec<String> {
    hub_file_urls(source, bundle, "config.json")
}

fn hub_file_urls(source: DownloadSource, bundle: &BundleRef, file: &str) -> Vec<String> {
    let repos = match source {
        DownloadSource::HuggingFace => vec![
            format!("ariacompute/{}", bundle.model),
            "ariacompute/model".into(),
        ],
        DownloadSource::ModelScope => vec![
            format!("AriaCompute/{}", bundle.model),
            "AriaCompute/model".into(),
        ],
        DownloadSource::Dashboard => return vec![],
    };
    let mut urls = Vec::new();
    for repo in repos {
        match source {
            DownloadSource::HuggingFace => {
                urls.push(format!(
                    "https://huggingface.co/{repo}/resolve/main/{}/{}/{}",
                    bundle.sdk, bundle.model, file
                ));
            }
            DownloadSource::ModelScope => {
                urls.push(format!(
                    "https://www.modelscope.cn/models/{repo}/resolve/master/{}/{}/{}",
                    bundle.sdk, bundle.model, file
                ));
                urls.push(format!(
                    "https://modelscope.cn/models/{repo}/resolve/master/{}/{}/{}",
                    bundle.sdk, bundle.model, file
                ));
            }
            DownloadSource::Dashboard => {}
        }
    }
    urls
}

fn hub_token(source: DownloadSource) -> Option<String> {
    match source {
        DownloadSource::HuggingFace => std::env::var("HF_TOKEN")
            .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
            .ok()
            .filter(|s| !s.is_empty()),
        DownloadSource::ModelScope => std::env::var("MODELSCOPE_API_TOKEN")
            .or_else(|_| std::env::var("MODELSCOPE_TOKEN"))
            .ok()
            .filter(|s| !s.is_empty()),
        DownloadSource::Dashboard => None,
    }
}

fn http_client(timeout: Duration) -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
}

fn io_err<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::other(e.to_string())
}

fn looks_like_zip(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && bytes[0] == 0x50 && bytes[1] == 0x4b
}

/// Stream an HTTP body to `path`, showing a green progress bar when stderr is a TTY
/// and `label` is non-empty.
pub(crate) async fn stream_response_to_file(
    resp: reqwest::Response,
    path: &Path,
    label: &str,
) -> io::Result<u64> {
    let total = resp.content_length();
    let show = !label.is_empty() && io::IsTerminal::is_terminal(&io::stderr());
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

fn extract_zip_path(zip_path: &Path, dest: &Path) -> io::Result<()> {
    let file = fs::File::open(zip_path)?;
    let mut archive =
        ZipArchive::new(file).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let name = file
            .enclosed_name()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "unsafe zip path"))?
            .to_owned();
        let out_path = dest.join(&name);
        if file.is_dir() {
            fs::create_dir_all(&out_path)?;
            continue;
        }
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut out = fs::File::create(&out_path)?;
        io::copy(&mut file, &mut out)?;
    }
    flatten_single_subdir(dest)?;
    Ok(())
}

fn flatten_single_subdir(dest: &Path) -> io::Result<()> {
    let mut entries: Vec<_> = fs::read_dir(dest)?
        .filter_map(|e| e.ok())
        .collect();
    if entries.len() != 1 {
        return Ok(());
    }
    let only = entries.pop().unwrap();
    if !only.file_type()?.is_dir() {
        return Ok(());
    }
    let sub = only.path();
    // Only flatten if config.json lives inside.
    if !sub.join("config.json").exists() {
        return Ok(());
    }
    for entry in fs::read_dir(&sub)? {
        let entry = entry?;
        let target = dest.join(entry.file_name());
        fs::rename(entry.path(), target)?;
    }
    fs::remove_dir_all(sub)?;
    Ok(())
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

pub fn is_valid_bundle(dir: &Path) -> bool {
    let weight = dir.join("weight.bin");
    let config = dir.join("config.json");
    if !weight.is_file() || !config.is_file() {
        return false;
    }
    let Ok(raw) = fs::read_to_string(&config) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return false;
    };
    v.get("format").and_then(|x| x.as_str()) == Some("aria-quant-bundle")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListedModel {
    pub name: String,
    pub status: String,
}

#[derive(Debug, Deserialize)]
struct CatalogModel {
    slug: String,
    #[serde(default)]
    available: bool,
    #[serde(default, rename = "int4DownloadUrl")]
    int4_download_url: String,
    #[serde(default, rename = "int8DownloadUrl")]
    int8_download_url: String,
    #[serde(default, rename = "int326DownloadUrl")]
    int326_download_url: String,
}

const CATALOG_TIMEOUT: Duration = Duration::from_secs(15);

/// Fetch Dashboard catalog and merge with local `~/.ariacompute/models` status.
pub async fn list_models_with_catalog(cfg: &AriaConfig) -> io::Result<Vec<ListedModel>> {
    if cfg.site_url.trim().is_empty() || cfg.cloud_api_key.trim().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "list requires site_url and cloud_api_key; run `aria-engine auth`",
        ));
    }
    let catalog = fetch_dashboard_catalog(cfg).await?;
    let local = local_model_status()?;
    let mut seen = std::collections::HashSet::new();
    let mut rows = Vec::new();

    for bundle in expand_catalog_bundles(&catalog) {
        seen.insert(bundle.clone());
        let status = match local.get(&bundle) {
            Some(LocalStatus::Valid) => "downloaded",
            Some(LocalStatus::Incomplete) => "incomplete",
            None => "not downloaded",
        };
        rows.push(ListedModel {
            name: bundle,
            status: status.into(),
        });
    }

    let mut orphans: Vec<_> = local
        .into_iter()
        .filter(|(name, _)| !seen.contains(name))
        .collect();
    orphans.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, st) in orphans {
        rows.push(ListedModel {
            name,
            status: match st {
                LocalStatus::Valid => "downloaded".into(),
                LocalStatus::Incomplete => "incomplete".into(),
            },
        });
    }
    Ok(rows)
}

fn expand_catalog_bundles(catalog: &[CatalogModel]) -> Vec<String> {
    let mut names = Vec::new();
    for item in catalog {
        if !item.available {
            continue;
        }
        let slug = item.slug.trim();
        if slug.is_empty() {
            continue;
        }
        if !item.int4_download_url.trim().is_empty() {
            names.push(format!("{slug}_q4"));
        }
        if !item.int8_download_url.trim().is_empty() {
            names.push(format!("{slug}_q8"));
        }
        if !item.int326_download_url.trim().is_empty() {
            names.push(format!("{slug}_q326"));
        }
    }
    names.sort();
    names.dedup();
    names
}

async fn fetch_dashboard_catalog(cfg: &AriaConfig) -> io::Result<Vec<CatalogModel>> {
    let base = cfg.site_url.trim_end_matches('/');
    let url = format!("{base}/api/dashboard/models");
    let client = http_client(CATALOG_TIMEOUT).map_err(io_err)?;
    let resp = client
        .get(&url)
        .bearer_auth(&cfg.cloud_api_key)
        .header("Accept", "application/json")
        .send()
        .await
        .map_err(|e| {
            io::Error::new(
                io::ErrorKind::NotConnected,
                format!("catalog fetch failed ({e}); check site_url / network or re-run `aria-engine auth`"),
            )
        })?;
    if !resp.status().is_success() {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "catalog HTTP {}; check cloud_api_key or re-run `aria-engine auth`",
                resp.status()
            ),
        ));
    }
    resp.json::<Vec<CatalogModel>>().await.map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("catalog JSON parse failed: {e}"),
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalStatus {
    Valid,
    Incomplete,
}

fn local_model_status() -> io::Result<std::collections::HashMap<String, LocalStatus>> {
    let dir = config::models_dir()?;
    let mut map = std::collections::HashMap::new();
    if !dir.exists() {
        return Ok(map);
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let status = if is_valid_bundle(&entry.path()) {
            LocalStatus::Valid
        } else {
            LocalStatus::Incomplete
        };
        map.insert(name, status);
    }
    Ok(map)
}

/// Local-only listing (no catalog). Kept for tests / callers that do not need remote.
pub fn list_models() -> io::Result<Vec<String>> {
    let local = local_model_status()?;
    let mut names: Vec<String> = local
        .into_iter()
        .map(|(name, st)| match st {
            LocalStatus::Valid => name,
            LocalStatus::Incomplete => format!("{name} (incomplete)"),
        })
        .collect();
    names.sort();
    Ok(names)
}

pub fn clean_models(model: Option<&str>) -> io::Result<()> {
    match model {
        Some(m) => {
            let path = config::model_dir(m)?;
            if path.exists() {
                fs::remove_dir_all(path)?;
            }
        }
        None => {
            let dir = config::models_dir()?;
            if dir.exists() {
                for entry in fs::read_dir(dir)? {
                    let entry = entry?;
                    if entry.file_type()?.is_dir() {
                        fs::remove_dir_all(entry.path())?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_quant_suffixes() {
        let a = parse_bundle_name("qwen3-0.6b_q4");
        assert_eq!(a.slug, "qwen3-0.6b");
        assert_eq!(a.quant, "int4");
        let b = parse_bundle_name("gemma-4-e2b-it_q8");
        assert_eq!(b.quant, "int8");
        let c = parse_bundle_name("model_q326");
        assert_eq!(c.quant, "int326");
        let d = parse_bundle_name("model_q3.26");
        assert_eq!(d.quant, "int326");
        let e = parse_bundle_name("plain-slug");
        assert_eq!(e.slug, "plain-slug");
        assert_eq!(e.quant, "int4");
    }

    #[test]
    fn hub_urls_follow_upload_layout() {
        let b = parse_bundle_name("gemma-4-e2b-it_q4");
        let hf = hub_file_urls(DownloadSource::HuggingFace, &b, "config.json");
        assert!(hf[0].contains("/ariacompute/gemma-4-e2b-it_q4/resolve/main/v1.0/gemma-4-e2b-it_q4/config.json"));
        assert!(hf[1].contains("/ariacompute/model/resolve/main/v1.0/gemma-4-e2b-it_q4/config.json"));
        let ms = hub_file_urls(DownloadSource::ModelScope, &b, "weight.bin");
        assert!(ms[0].contains("AriaCompute/gemma-4-e2b-it_q4"));
        assert!(ms[0].contains("/v1.0/gemma-4-e2b-it_q4/weight.bin"));
    }

    #[test]
    fn expand_available_quants_to_bundles() {
        let catalog = vec![
            CatalogModel {
                slug: "gemma-4-e2b-it".into(),
                available: true,
                int4_download_url: "https://x/int4".into(),
                int8_download_url: "https://x/int8".into(),
                int326_download_url: String::new(),
            },
            CatalogModel {
                slug: "hidden".into(),
                available: false,
                int4_download_url: "https://x/int4".into(),
                int8_download_url: String::new(),
                int326_download_url: String::new(),
            },
            CatalogModel {
                slug: "only326".into(),
                available: true,
                int4_download_url: String::new(),
                int8_download_url: String::new(),
                int326_download_url: "https://x/int326".into(),
            },
        ];
        assert_eq!(
            expand_catalog_bundles(&catalog),
            vec![
                "gemma-4-e2b-it_q4".to_string(),
                "gemma-4-e2b-it_q8".to_string(),
                "only326_q326".to_string(),
            ]
        );
    }

    #[test]
    fn select_highest_score() {
        let ranked = vec![
            ProbeResult {
                source: DownloadSource::HuggingFace,
                reachable: true,
                bytes_per_sec: 1e6,
                detail: "1".into(),
            },
            ProbeResult {
                source: DownloadSource::ModelScope,
                reachable: true,
                bytes_per_sec: 5e6,
                detail: "5".into(),
            },
            ProbeResult {
                source: DownloadSource::Dashboard,
                reachable: false,
                bytes_per_sec: 0.0,
                detail: "x".into(),
            },
        ];
        let mut sorted = ranked;
        sorted.sort_by(|a, b| {
            b.reachable.cmp(&a.reachable).then(
                b.bytes_per_sec
                    .partial_cmp(&a.bytes_per_sec)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
        assert_eq!(sorted[0].source, DownloadSource::ModelScope);
        assert!(sorted[2].source == DownloadSource::Dashboard);
    }
}
