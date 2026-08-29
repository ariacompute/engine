//! Probe the regional public hub (Hugging Face or ModelScope) and download Aria bundles.

use crate::config::{self, AriaConfig};
use crate::gateway_detect::GatewayPair;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
#[cfg(test)]
use serde::Deserialize;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

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

/// Parse `<model>` → slug + quant.
/// `*_q4`→int4, `*_q8`→int8, `*_q326`/`*_q3.26`→int326.
/// Optional codebook-share suffix `_channel` / `_group` (e.g. `*_q326_channel`).
pub fn parse_bundle_name(model: &str) -> BundleRef {
    let model = model.trim().trim_end_matches('/').to_string();
    let lower = model.to_ascii_lowercase();
    let suffixes = [
        ("_q3.26_channel", "int326"),
        ("_q326_channel", "int326"),
        ("_q8_channel", "int8"),
        ("_q4_channel", "int4"),
        ("_q3.26_group", "int326"),
        ("_q326_group", "int326"),
        ("_q8_group", "int8"),
        ("_q4_group", "int4"),
        ("_q3.26", "int326"),
        ("_q326", "int326"),
        ("_q8", "int8"),
        ("_q4", "int4"),
    ];
    let (slug, quant) = suffixes
        .iter()
        .find_map(|(suf, quant)| {
            lower
                .strip_suffix(suf)
                .map(|base| (model[..base.len()].to_string(), (*quant).to_string()))
        })
        .unwrap_or_else(|| (model.clone(), "int4".into()));
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

pub async fn download_model(model: &str, cfg: &AriaConfig) -> io::Result<PathBuf> {
    let bundle = parse_bundle_name(model);
    config::ensure_aria_home()?;
    let dest = config::model_dir(&bundle.model)?;
    if is_valid_bundle(&dest) {
        eprintln!("download: already present at {}", dest.display());
        return Ok(dest);
    }
    for alias in config::bundle_cache_aliases(&bundle.model) {
        let cached = config::model_dir(&alias)?;
        if is_valid_bundle(&cached) {
            eprintln!("download: already present at {}", cached.display());
            return Ok(cached);
        }
    }

    let source = preferred_public_hub(&cfg.site_url);
    let probe = probe_hub(source, &bundle, cfg).await;
    if !probe.reachable {
        return Err(io::Error::new(
            io::ErrorKind::NotConnected,
            format!(
                "no download source reachable ({}: {})",
                source.as_str(),
                probe.detail
            ),
        ));
    }

    eprintln!(
        "download: using {} ({})",
        probe.source.as_str(),
        probe.detail
    );
    match fetch_hub(source, &bundle, &dest, cfg).await {
        Ok(()) => {
            if is_valid_bundle(&dest) {
                return Ok(dest);
            }
            let _ = fs::remove_dir_all(&dest);
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{} fetch completed but bundle invalid (need weight.bin + aria-quant-bundle config.json)",
                    source.as_str()
                ),
            ))
        }
        Err(e) => {
            let _ = fs::remove_dir_all(&dest);
            Err(e)
        }
    }
}

/// Public hub paired with `site_url` (`.com` → Hugging Face, `.cn` → ModelScope).
pub(crate) fn preferred_public_hub(site_url: &str) -> DownloadSource {
    match GatewayPair::from_url(site_url) {
        Some(GatewayPair::CN) => DownloadSource::ModelScope,
        _ => DownloadSource::HuggingFace,
    }
}

async fn probe_hub(source: DownloadSource, bundle: &BundleRef, cfg: &AriaConfig) -> ProbeResult {
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
        if let Some(token) = hub_token(source, cfg) {
            req = req.bearer_auth(token);
        }
        // Prefer Range to cap probe size.
        req = req.header(
            "Range",
            format!("bytes=0-{}", PROBE_BYTES.saturating_sub(1)),
        );
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
                    detail: format!(
                        "auth failed HTTP {}; run `aria-engine setup` to set {}",
                        r.status(),
                        hub_token_field(source)
                    ),
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

async fn fetch_hub(
    source: DownloadSource,
    bundle: &BundleRef,
    dest: &Path,
    cfg: &AriaConfig,
) -> io::Result<()> {
    let files = ["config.json", "weight.bin"];
    let staging = dest.with_extension("partial");
    let _ = fs::remove_dir_all(&staging);
    fs::create_dir_all(&staging)?;
    let client = http_client(Duration::from_secs(600)).map_err(io_err)?;

    for file in files {
        let mut fetched = false;
        for url in hub_file_urls(source, bundle, file) {
            let mut req = client.get(&url);
            if let Some(token) = hub_token(source, cfg) {
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
            if let Some(token) = hub_token(source, cfg) {
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

fn hub_config_urls(source: DownloadSource, bundle: &BundleRef) -> Vec<String> {
    hub_file_urls(source, bundle, "config.json")
}

/// `(repo_id, path_dirname)` candidates, same order as `hub_file_urls`.
pub(crate) fn hub_repo_candidates(
    source: DownloadSource,
    bundle: &BundleRef,
) -> Vec<(String, String)> {
    let mut names = vec![bundle.model.clone()];
    names.extend(config::bundle_cache_aliases(&bundle.model));
    let mut out = Vec::new();
    for name in names {
        match source {
            DownloadSource::HuggingFace => {
                out.push((format!("ariacompute/{name}"), name.clone()));
                out.push(("ariacompute/model".into(), name));
            }
            DownloadSource::ModelScope => {
                out.push((format!("AriaCompute/{name}"), name.clone()));
                out.push(("AriaCompute/model".into(), name));
            }
            DownloadSource::Dashboard => {}
        }
    }
    out
}

pub(crate) fn hub_file_urls(source: DownloadSource, bundle: &BundleRef, file: &str) -> Vec<String> {
    let mut urls = Vec::new();
    for (repo, name) in hub_repo_candidates(source, bundle) {
        match source {
            DownloadSource::HuggingFace => {
                urls.push(format!(
                    "https://huggingface.co/{repo}/resolve/main/{}/{}/{}",
                    bundle.sdk, name, file
                ));
            }
            DownloadSource::ModelScope => {
                urls.push(format!(
                    "https://www.modelscope.cn/models/{repo}/resolve/master/{}/{}/{}",
                    bundle.sdk, name, file
                ));
                urls.push(format!(
                    "https://modelscope.cn/models/{repo}/resolve/master/{}/{}/{}",
                    bundle.sdk, name, file
                ));
            }
            DownloadSource::Dashboard => {}
        }
    }
    urls
}

fn hub_token_field(source: DownloadSource) -> &'static str {
    match source {
        DownloadSource::HuggingFace => "hf_token",
        DownloadSource::ModelScope => "modelscope_api_token",
        DownloadSource::Dashboard => "hf_token",
    }
}

pub(crate) fn hub_token(source: DownloadSource, cfg: &AriaConfig) -> Option<String> {
    let raw = match source {
        DownloadSource::HuggingFace => cfg.hf_token.as_str(),
        DownloadSource::ModelScope => cfg.modelscope_api_token.as_str(),
        DownloadSource::Dashboard => "",
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

pub(crate) fn http_client(timeout: Duration) -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
}

pub(crate) fn io_err<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::other(e.to_string())
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
#[cfg(test)]
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

/// Local-only listing (catalog / Dashboard removed).
pub async fn list_models_with_catalog(_cfg: &AriaConfig) -> io::Result<Vec<ListedModel>> {
    let local = local_model_status()?;
    let mut rows: Vec<ListedModel> = local
        .into_iter()
        .map(|(name, st)| ListedModel {
            name,
            status: match st {
                LocalStatus::Valid => "downloaded".into(),
                LocalStatus::Incomplete => "incomplete".into(),
            },
        })
        .collect();
    rows.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(rows)
}

#[cfg(test)]
fn merge_catalog_and_local(
    catalog_bundles: Vec<String>,
    local: std::collections::HashMap<String, LocalStatus>,
) -> Vec<ListedModel> {
    let mut seen = std::collections::HashSet::new();
    let mut rows = Vec::new();

    for bundle in catalog_bundles {
        seen.insert(bundle.clone());
        let status = match lookup_local_status(&local, &bundle) {
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
        .filter(|(name, _)| !covered_by_catalog(name, &seen))
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
    rows
}

#[cfg(test)]
fn lookup_local_status(
    local: &std::collections::HashMap<String, LocalStatus>,
    catalog_name: &str,
) -> Option<LocalStatus> {
    if let Some(&st) = local.get(catalog_name) {
        return Some(st);
    }
    let mut incomplete = None;
    for alias in config::bundle_cache_aliases(catalog_name) {
        match local.get(&alias) {
            Some(LocalStatus::Valid) => return Some(LocalStatus::Valid),
            Some(LocalStatus::Incomplete) => incomplete = Some(LocalStatus::Incomplete),
            None => {}
        }
    }
    incomplete
}

#[cfg(test)]
fn covered_by_catalog(name: &str, seen: &std::collections::HashSet<String>) -> bool {
    seen.contains(name)
        || config::bundle_cache_aliases(name)
            .iter()
            .any(|alias| seen.contains(alias))
}

#[cfg(test)]
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
    fn hub_token_reads_config_not_env() {
        std::env::set_var("HF_TOKEN", "env-hf");
        std::env::set_var("MODELSCOPE_API_TOKEN", "env-ms");
        let empty = AriaConfig::default();
        assert_eq!(hub_token(DownloadSource::HuggingFace, &empty), None);
        assert_eq!(hub_token(DownloadSource::ModelScope, &empty), None);
        let cfg = AriaConfig {
            hf_token: "hf_from_yml".into(),
            modelscope_api_token: "ms_from_yml".into(),
            ..AriaConfig::default()
        };
        assert_eq!(
            hub_token(DownloadSource::HuggingFace, &cfg).as_deref(),
            Some("hf_from_yml")
        );
        assert_eq!(
            hub_token(DownloadSource::ModelScope, &cfg).as_deref(),
            Some("ms_from_yml")
        );
        std::env::remove_var("HF_TOKEN");
        std::env::remove_var("MODELSCOPE_API_TOKEN");
    }

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
        let f = parse_bundle_name("gemma-3-1b-it_q326_channel");
        assert_eq!(f.slug, "gemma-3-1b-it");
        assert_eq!(f.quant, "int326");
        assert_eq!(f.model, "gemma-3-1b-it_q326_channel");
        let g = parse_bundle_name("model_q4_channel");
        assert_eq!(g.slug, "model");
        assert_eq!(g.quant, "int4");
    }

    #[test]
    fn hub_urls_follow_upload_layout() {
        let b = parse_bundle_name("gemma-4-e2b-it_q4");
        let hf = hub_file_urls(DownloadSource::HuggingFace, &b, "config.json");
        assert!(hf[0].contains(
            "/ariacompute/gemma-4-e2b-it_q4/resolve/main/v1.0/gemma-4-e2b-it_q4/config.json"
        ));
        assert!(
            hf[1].contains("/ariacompute/model/resolve/main/v1.0/gemma-4-e2b-it_q4/config.json")
        );
        let q326 = parse_bundle_name("gemma-3-1b-it_q326");
        let q326_hf = hub_file_urls(DownloadSource::HuggingFace, &q326, "config.json");
        assert!(q326_hf
            .iter()
            .any(|u| u.contains("/v1.0/gemma-3-1b-it_q326_channel/config.json")));
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
    fn preferred_hub_follows_site_tld() {
        assert_eq!(
            preferred_public_hub("https://ariacompute.com"),
            DownloadSource::HuggingFace
        );
        assert_eq!(
            preferred_public_hub("https://ariacompute.cn"),
            DownloadSource::ModelScope
        );
        assert_eq!(preferred_public_hub(""), DownloadSource::HuggingFace);
    }

    #[test]
    fn download_never_selects_dashboard() {
        assert_ne!(
            preferred_public_hub("https://ariacompute.com"),
            DownloadSource::Dashboard
        );
        assert_ne!(
            preferred_public_hub("https://ariacompute.cn"),
            DownloadSource::Dashboard
        );
    }

    #[test]
    fn list_maps_q326_channel_to_catalog_q326() {
        let local = std::collections::HashMap::from([(
            "gemma-3-1b-it_q326_channel".to_string(),
            LocalStatus::Valid,
        )]);
        let rows = merge_catalog_and_local(
            vec!["gemma-3-1b-it_q4".into(), "gemma-3-1b-it_q326".into()],
            local,
        );
        assert_eq!(
            rows,
            vec![
                ListedModel {
                    name: "gemma-3-1b-it_q4".into(),
                    status: "not downloaded".into(),
                },
                ListedModel {
                    name: "gemma-3-1b-it_q326".into(),
                    status: "downloaded".into(),
                },
            ]
        );
    }

    #[test]
    fn list_does_not_orphan_q326_channel_when_catalog_has_q326() {
        let local = std::collections::HashMap::from([
            ("gemma-3-1b-it_q326_channel".to_string(), LocalStatus::Valid),
            ("local-only_q4".to_string(), LocalStatus::Valid),
        ]);
        let rows = merge_catalog_and_local(vec!["gemma-3-1b-it_q326".into()], local);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].name, "gemma-3-1b-it_q326");
        assert_eq!(rows[0].status, "downloaded");
        assert_eq!(rows[1].name, "local-only_q4");
        assert_eq!(rows[1].status, "downloaded");
    }

    #[tokio::test]
    async fn list_models_with_catalog_is_local_only() {
        let rows = list_models_with_catalog(&AriaConfig::default())
            .await
            .unwrap();
        let names = list_models().unwrap();
        assert_eq!(rows.len(), names.len());
    }
}
