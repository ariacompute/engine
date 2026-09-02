//! Regional public-hub model auto-download for the Rust SDK.
//!
//! Matches `ariaengine download`: `.com` → Hugging Face, `.cn` → ModelScope.
//! Dashboard zip meta is not used. A Dashboard `sk-` / `bfvk-` token is ignored
//! for hub auth.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DownloadError {
    #[error("invalid model name: {0}")]
    InvalidModelName(String),
    #[error("{0}")]
    Request(String),
    #[error("download stream failed: {0}")]
    Stream(String),
    #[error("invalid bundle after download: {0}")]
    InvalidBundle(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

const DEFAULT_SITE: &str = "https://ariacompute.com";
const DEFAULT_SDK: &str = "v1.0";
const HUB_REQUIRED: &[&str] = &["config.json", "weight.bin"];
const HUB_OPTIONAL: &[&str] = &[
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
];

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

fn preferred_public_hub(site: Option<&str>) -> &'static str {
    if site
        .unwrap_or("")
        .to_ascii_lowercase()
        .contains("ariacompute.cn")
    {
        "modelscope"
    } else {
        "huggingface"
    }
}

fn hub_bearer(token: &str) -> Option<&str> {
    let t = token.trim();
    if t.is_empty() {
        return None;
    }
    let low = t.to_ascii_lowercase();
    if low.starts_with("sk-") || low.starts_with("bfvk-") {
        None
    } else {
        Some(t)
    }
}

fn unquote_yaml(v: &str) -> String {
    let t = v.trim();
    let b = t.as_bytes();
    if b.len() >= 2
        && ((b[0] == b'"' && *b.last().unwrap() == b'"')
            || (b[0] == b'\'' && *b.last().unwrap() == b'\''))
    {
        t[1..t.len() - 1].to_string()
    } else {
        t.to_string()
    }
}

fn scalar_from_yml_path(path: &std::path::Path, key: &str) -> Option<String> {
    let raw = std::fs::read_to_string(path).ok()?;
    for line in raw.lines() {
        if line.starts_with(' ') || line.starts_with('\t') {
            continue;
        }
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        let Some((k, v)) = s.split_once(':') else {
            continue;
        };
        if k.trim() != key {
            continue;
        }
        let val = unquote_yaml(v);
        if val.is_empty() {
            return None;
        }
        return Some(val);
    }
    None
}

fn config_yml_scalar(key: &str) -> Option<String> {
    let home = aria_home().ok()?;
    for name in ["engine.yml", "config.yml"] {
        if let Some(v) = scalar_from_yml_path(&home.join(name), key) {
            return Some(v);
        }
    }
    None
}

fn hub_token_field(source: &str) -> &'static str {
    if source == "modelscope" {
        "modelscope_api_token"
    } else {
        "hf_token"
    }
}

fn resolve_hub_token(
    source: &str,
    token: &str,
    hf_token: Option<&str>,
    modelscope_api_token: Option<&str>,
) -> Option<String> {
    let named = if source == "modelscope" {
        modelscope_api_token.unwrap_or("")
    } else {
        hf_token.unwrap_or("")
    };
    let from_cfg = config_yml_scalar(hub_token_field(source)).unwrap_or_default();
    for cand in [named, token, from_cfg.as_str()] {
        if let Some(b) = hub_bearer(cand) {
            return Some(b.to_string());
        }
    }
    None
}

fn hub_path_names(model: &str) -> Vec<String> {
    let mut names = vec![model.to_string()];
    let mut lower = model.to_ascii_lowercase();
    let mut core = model.to_string();
    for suf in ["_channel", "_group"] {
        if lower.ends_with(suf) {
            core = model[..model.len() - suf.len()].to_string();
            lower = core.to_ascii_lowercase();
            break;
        }
    }
    let mut stems = vec![core.clone()];
    if lower.ends_with("_q326") {
        stems.push(format!("{}q3.26", &core[..core.len() - 5]));
    } else if lower.ends_with("_q3.26") {
        stems.push(format!("{}q326", &core[..core.len() - 6]));
    }
    for stem in stems {
        for share in ["", "_channel", "_group"] {
            let cand = format!("{stem}{share}");
            if !names.iter().any(|n| n == &cand) {
                names.push(cand);
            }
        }
    }
    names
}

fn hub_file_urls(source: &str, model: &str, file: &str) -> Vec<String> {
    let mut urls = Vec::new();
    for name in hub_path_names(model) {
        if source == "modelscope" {
            for repo in [format!("AriaCompute/{name}"), "AriaCompute/model".into()] {
                urls.push(format!(
                    "https://www.modelscope.cn/models/{repo}/resolve/master/{DEFAULT_SDK}/{name}/{file}"
                ));
                urls.push(format!(
                    "https://modelscope.cn/models/{repo}/resolve/master/{DEFAULT_SDK}/{name}/{file}"
                ));
            }
        } else {
            for repo in [format!("ariacompute/{name}"), "ariacompute/model".into()] {
                urls.push(format!(
                    "https://huggingface.co/{repo}/resolve/main/{DEFAULT_SDK}/{name}/{file}"
                ));
            }
        }
    }
    urls
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

fn atomic_replace(src: &Path, dst: &Path) -> std::io::Result<()> {
    if dst.exists() {
        std::fs::remove_dir_all(dst)?;
    }
    std::fs::rename(src, dst)
}

fn auth_error(source: &str, code: u16) -> DownloadError {
    let field = if source == "modelscope" {
        "modelscope_api_token"
    } else {
        "hf_token"
    };
    DownloadError::Request(format!(
        "auth failed HTTP {code}; set {field} via ariaengine setup (do not pass a Dashboard sk-/bfvk- key as the hub token)"
    ))
}

fn fetch_url_to_file(url: &str, dest: &Path, token: Option<&str>) -> Result<(), DownloadError> {
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(600))
        .build();
    let mut req = agent.get(url);
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let resp = match req.call() {
        Ok(r) => r,
        Err(ureq::Error::Status(code, _)) if code == 401 || code == 403 => {
            return Err(DownloadError::Request(format!("HTTP {code}")));
        }
        Err(ureq::Error::Status(code, _)) => {
            return Err(DownloadError::Request(format!("HTTP {code}")));
        }
        Err(e) => return Err(DownloadError::Request(e.to_string())),
    };
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut reader = resp.into_reader();
    let mut out = std::fs::File::create(dest)?;
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| DownloadError::Stream(e.to_string()))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
    }
    Ok(())
}

fn fetch_hub_file(
    source: &str,
    model: &str,
    file: &str,
    dest: &Path,
    token: Option<&str>,
    required: bool,
) -> Result<bool, DownloadError> {
    let mut last: Option<DownloadError> = None;
    for url in hub_file_urls(source, model, file) {
        match fetch_url_to_file(&url, dest, token) {
            Ok(()) => return Ok(true),
            Err(DownloadError::Request(msg))
                if msg.contains("HTTP 401") || msg.contains("HTTP 403") =>
            {
                let code = if msg.contains("401") { 401 } else { 403 };
                return Err(auth_error(source, code));
            }
            Err(e) => last = Some(e),
        }
    }
    if required {
        Err(DownloadError::Request(format!(
            "{source}: missing {file}{}",
            last.map(|e| format!(": {e}")).unwrap_or_default()
        )))
    } else {
        Ok(false)
    }
}

/// Download `model` from the regional public hub into
/// `~/.ariacompute/models/{model}`, then return that directory.
///
/// If a valid bundle already exists at the cache path, the download is skipped.
/// Hub auth: explicit `hf_token` / `modelscope_api_token`, then generic `token`,
/// then `~/.ariacompute/engine.yml` (same keys as `ariaengine setup`).
/// Dashboard `sk-` / `bfvk-` keys are not sent to the hub.
pub fn download_model(
    model: &str,
    token: &str,
    site: Option<&str>,
) -> Result<PathBuf, DownloadError> {
    download_model_setup(model, token, site, None, None)
}

/// Like [`download_model`], with named hub tokens matching `ariaengine setup`.
pub fn download_model_setup(
    model: &str,
    token: &str,
    site: Option<&str>,
    hf_token: Option<&str>,
    modelscope_api_token: Option<&str>,
) -> Result<PathBuf, DownloadError> {
    parse_bundle_name(model)?;
    let site = site.unwrap_or(DEFAULT_SITE);
    let source = preferred_public_hub(Some(site));
    let hub_owned = resolve_hub_token(source, token, hf_token, modelscope_api_token);
    let hub_token = hub_owned.as_deref();
    let cache = models_dir()?.join(model);

    if cache.exists() && is_valid_bundle(&cache) {
        return Ok(cache);
    }

    let staging = models_dir()?.join(format!(".{}.partial", model));
    if staging.exists() {
        std::fs::remove_dir_all(&staging)?;
    }
    std::fs::create_dir_all(&staging)?;
    let result = (|| {
        for file in HUB_REQUIRED {
            fetch_hub_file(source, model, file, &staging.join(file), hub_token, true)?;
        }
        for extra in HUB_OPTIONAL {
            let _ = fetch_hub_file(source, model, extra, &staging.join(extra), hub_token, false);
        }
        if !is_valid_bundle(&staging) {
            return Err(DownloadError::InvalidBundle(
                "need weight.bin + aria-quant-bundle config.json".into(),
            ));
        }
        atomic_replace(&staging, &cache)?;
        Ok(cache.clone())
    })();
    if result.is_err() && staging.exists() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    result
}

const SDK_UA: &str = "ariaengine-sdk/0.1.0";

fn ffi_lib_name() -> &'static str {
    if cfg!(windows) {
        "ariaengine_ffi.dll"
    } else if cfg!(target_os = "macos") {
        "libariaengine_ffi.dylib"
    } else {
        "libariaengine_ffi.so"
    }
}

fn lib_dir() -> Result<PathBuf, DownloadError> {
    Ok(aria_home()?.join("lib"))
}

fn cached_ffi_path() -> Result<Option<PathBuf>, DownloadError> {
    let p = lib_dir()?.join(ffi_lib_name());
    Ok(if p.is_file() { Some(p) } else { None })
}

pub(crate) fn ffi_asset_os(os: &str, arch: &str) -> Result<&'static str, DownloadError> {
    match (os, arch) {
        ("linux", "x86_64") => Ok("linux_x86_64"),
        ("linux", "aarch64") => Ok("linux_arm64"),
        ("macos", _) => Ok("macos"),
        ("windows", "x86_64") => Ok("windows_x86_64"),
        _ => Err(DownloadError::Request(format!(
            "unsupported platform {os}/{arch} for libariaengine_ffi"
        ))),
    }
}

fn strip_v(tag: &str) -> &str {
    let t = tag.trim();
    t.strip_prefix('v').or_else(|| t.strip_prefix('V')).unwrap_or(t)
}

fn parse_semver(tag: &str) -> Option<(u64, u64, u64)> {
    let core = strip_v(tag).split(['-', '+']).next().unwrap_or("");
    let mut parts = core.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    let patch = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor, patch))
}

pub(crate) fn select_latest_stable(releases: &[serde_json::Value]) -> Result<String, DownloadError> {
    let mut best_tag: Option<&str> = None;
    let mut best_key = (0u64, 0u64, 0u64);
    let mut found = false;
    for rel in releases {
        if rel.get("draft").and_then(|v| v.as_bool()).unwrap_or(false)
            || rel.get("prerelease").and_then(|v| v.as_bool()).unwrap_or(false)
        {
            continue;
        }
        let tag = rel
            .get("tag_name")
            .or_else(|| rel.get("tag"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if let Some(parsed) = parse_semver(tag) {
            if !found || parsed > best_key {
                best_key = parsed;
                best_tag = Some(tag);
                found = true;
            }
        }
    }
    best_tag
        .map(|t| strip_v(t).to_string())
        .ok_or_else(|| DownloadError::Request("no stable release found for libariaengine_ffi".into()))
}

fn upgrade_org(site: Option<&str>) -> String {
    if let Some(cfg) = config_yml_scalar("upgrade_url") {
        return cfg.trim_end_matches('/').to_string();
    }
    let from_cfg = config_yml_scalar("site_url");
    let hint = site
        .or(from_cfg.as_deref())
        .unwrap_or(DEFAULT_SITE)
        .to_ascii_lowercase();
    if hint.contains("ariacompute.cn") || hint.contains("gitee.com") {
        "https://gitee.com/ariacompute".into()
    } else {
        "https://github.com/ariacompute".into()
    }
}

fn releases_api_url(org: &str) -> String {
    let owner = org.trim_end_matches('/').rsplit('/').next().unwrap_or("ariacompute");
    if org.to_ascii_lowercase().contains("gitee.com") {
        format!("https://gitee.com/api/v5/repos/{owner}/engine/releases?per_page=30")
    } else {
        format!("https://api.github.com/repos/{owner}/engine/releases?per_page=30")
    }
}

fn http_get_bytes(url: &str) -> Result<Vec<u8>, DownloadError> {
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(600))
        .build();
    let resp = agent
        .get(url)
        .set("User-Agent", SDK_UA)
        .call()
        .map_err(|e| DownloadError::Request(e.to_string()))?;
    let mut reader = resp.into_reader();
    let mut buf = Vec::new();
    reader
        .read_to_end(&mut buf)
        .map_err(|e| DownloadError::Stream(e.to_string()))?;
    Ok(buf)
}

pub(crate) fn extract_ffi_archive(
    archive: &Path,
    dest_dir: &Path,
    want: &str,
) -> Result<PathBuf, DownloadError> {
    let file = std::fs::File::open(archive)?;
    let dec = flate2::read::GzDecoder::new(file);
    let mut ar = tar::Archive::new(dec);
    for entry in ar.entries()? {
        let mut entry = entry?;
        let name = entry.path()?;
        let base = name
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if base != want {
            continue;
        }
        std::fs::create_dir_all(dest_dir)?;
        let dest = dest_dir.join(want);
        {
            let mut out = std::fs::File::create(&dest)?;
            std::io::copy(&mut entry, &mut out)?;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&dest)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&dest, perms)?;
        }
        return Ok(dest);
    }
    Err(DownloadError::Request(format!(
        "{} not found in {}",
        want,
        archive.display()
    )))
}

/// Return a path to libariaengine_ffi, downloading the latest stable Release if needed.
pub fn ensure_ffi_lib(site: Option<&str>) -> Result<PathBuf, DownloadError> {
    if let Ok(env) = std::env::var("ARIAENGINE_FFI_LIB") {
        let p = PathBuf::from(&env);
        if p.is_file() {
            return Ok(p);
        }
    }
    if let Some(cached) = cached_ffi_path()? {
        return Ok(cached);
    }

    let org = upgrade_org(site);
    let raw = http_get_bytes(&releases_api_url(&org))?;
    let releases: Vec<serde_json::Value> = serde_json::from_slice(&raw)
        .map_err(|e| DownloadError::Request(format!("invalid releases JSON from {org}: {e}")))?;
    let ver = select_latest_stable(&releases)?;
    let asset_os = ffi_asset_os(std::env::consts::OS, std::env::consts::ARCH)?;
    let asset_name = format!("libariaengine_ffi_{ver}_{asset_os}.tar.gz");
    let mut url = None;
    for rel in &releases {
        let tag = rel
            .get("tag_name")
            .or_else(|| rel.get("tag"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if strip_v(tag) != ver {
            continue;
        }
        if let Some(assets) = rel.get("assets").and_then(|v| v.as_array()) {
            for asset in assets {
                if asset.get("name").and_then(|v| v.as_str()) == Some(asset_name.as_str()) {
                    url = asset
                        .get("browser_download_url")
                        .or_else(|| asset.get("direct_asset_url"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    break;
                }
            }
        }
        if url.is_some() {
            break;
        }
    }
    let url = url.ok_or_else(|| DownloadError::Request(format!("release asset not found: {asset_name}")))?;

    let staging = aria_home()?.join("tmp").join(format!("ffi-{ver}"));
    if staging.exists() {
        std::fs::remove_dir_all(&staging)?;
    }
    std::fs::create_dir_all(&staging)?;
    let archive = staging.join(&asset_name);
    let result = (|| {
        let bytes = http_get_bytes(&url)?;
        if let Some(parent) = archive.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&archive, bytes)?;
        let dir = lib_dir()?;
        extract_ffi_archive(&archive, &dir, ffi_lib_name())
    })();
    let _ = std::fs::remove_dir_all(&staging);
    result
}

#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use super::ENV_LOCK;

    #[test]
    fn preferred_hub_follows_site_tld() {
        assert_eq!(
            preferred_public_hub(Some("https://ariacompute.com")),
            "huggingface"
        );
        assert_eq!(
            preferred_public_hub(Some("https://ariacompute.cn")),
            "modelscope"
        );
        assert_eq!(preferred_public_hub(None), "huggingface");
    }

    #[test]
    fn dashboard_token_not_sent_to_hub() {
        assert!(hub_bearer("sk-bf-95076ed1-8c1a-4efa-b33c-f52c1d7f9f24").is_none());
        assert!(hub_bearer("bfvk-test").is_none());
        assert_eq!(hub_bearer("hf_abc"), Some("hf_abc"));
    }

    #[test]
    fn hub_urls_follow_upload_layout() {
        let hf = hub_file_urls("huggingface", "gemma-4-e2b-it_q4", "config.json");
        assert!(hf.iter().any(|u| u.contains(
            "/ariacompute/gemma-4-e2b-it_q4/resolve/main/v1.0/gemma-4-e2b-it_q4/config.json"
        )));
        let ms = hub_file_urls("modelscope", "gemma-4-e2b-it_q4", "weight.bin");
        assert!(ms
            .iter()
            .any(|u| u.contains("/v1.0/gemma-4-e2b-it_q4/weight.bin")));
        assert!(hf.iter().chain(ms.iter()).all(|u| !u.contains("/api/dashboard/")));
    }

    #[test]
    fn parse_channel_suffix() {
        let (slug, quant) = parse_bundle_name("foo_q326_channel").unwrap();
        assert_eq!(slug, "foo");
        assert_eq!(quant, "int326");
    }

    #[test]
    fn cached_bundle_skips_download() {
        let _guard = ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("ARIA_COMPUTE_HOME", tmp.path());
        let cache = tmp.path().join("models").join("foo_q4");
        std::fs::create_dir_all(&cache).unwrap();
        std::fs::write(cache.join("weight.bin"), b"x").unwrap();
        std::fs::write(
            cache.join("config.json"),
            br#"{"format":"aria-quant-bundle"}"#,
        )
        .unwrap();
        let got = download_model("foo_q4", "", None).unwrap();
        assert_eq!(got, cache);
        std::env::remove_var("ARIA_COMPUTE_HOME");
    }

    #[test]
    fn resolve_named_and_config_yml() {
        let _guard = ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("ARIA_COMPUTE_HOME", tmp.path());
        std::fs::write(
            tmp.path().join("config.yml"),
            "hf_token: hf_from_yml\nmodelscope_api_token: \"ms_from_yml\"\n",
        )
        .unwrap();
        assert_eq!(
            resolve_hub_token("huggingface", "hf_generic", Some("hf_named"), None).as_deref(),
            Some("hf_named")
        );
        assert_eq!(
            resolve_hub_token("modelscope", "", None, Some("ms_named")).as_deref(),
            Some("ms_named")
        );
        assert_eq!(
            resolve_hub_token("huggingface", "", None, None).as_deref(),
            Some("hf_from_yml")
        );
        assert_eq!(
            resolve_hub_token("modelscope", "", None, None).as_deref(),
            Some("ms_from_yml")
        );
        assert_eq!(
            resolve_hub_token("huggingface", "sk-bf-not-hub", None, None).as_deref(),
            Some("hf_from_yml")
        );
        std::env::remove_var("ARIA_COMPUTE_HOME");
    }

    #[test]
    fn ffi_asset_os_matches_upgrade() {
        assert_eq!(ffi_asset_os("linux", "x86_64").unwrap(), "linux_x86_64");
        assert_eq!(ffi_asset_os("linux", "aarch64").unwrap(), "linux_arm64");
        assert_eq!(ffi_asset_os("macos", "aarch64").unwrap(), "macos");
        assert_eq!(ffi_asset_os("windows", "x86_64").unwrap(), "windows_x86_64");
        assert!(ffi_asset_os("linux", "powerpc64").is_err());
    }

    #[test]
    fn select_latest_stable_skips_draft_and_prerelease() {
        let releases = serde_json::json!([
            {"tag_name": "v0.7.1", "draft": false, "prerelease": false},
            {"tag_name": "v0.8.0-rc1", "draft": false, "prerelease": true},
            {"tag_name": "v0.7.2", "draft": false, "prerelease": false},
            {"tag_name": "v0.9.0", "draft": true, "prerelease": false}
        ]);
        let arr = releases.as_array().unwrap();
        assert_eq!(select_latest_stable(arr).unwrap(), "0.7.2");
    }

    #[test]
    fn extract_ffi_and_cached_skip() {
        let _guard = ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("ARIA_COMPUTE_HOME", tmp.path());
        let prev_lib = std::env::var("ARIAENGINE_FFI_LIB").ok();
        std::env::remove_var("ARIAENGINE_FFI_LIB");
        let src_dir = tmp.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let want = if cfg!(windows) {
            "ariaengine_ffi.dll"
        } else if cfg!(target_os = "macos") {
            "libariaengine_ffi.dylib"
        } else {
            "libariaengine_ffi.so"
        };
        std::fs::write(src_dir.join(want), b"dummy-ffi").unwrap();
        let archive = tmp.path().join("libariaengine_ffi.tar.gz");
        {
            let f = std::fs::File::create(&archive).unwrap();
            let enc = flate2::write::GzEncoder::new(f, flate2::Compression::default());
            let mut builder = tar::Builder::new(enc);
            builder.append_path_with_name(src_dir.join(want), want).unwrap();
            builder.finish().unwrap();
        }
        let dest_dir = tmp.path().join("lib");
        let got = extract_ffi_archive(&archive, &dest_dir, want).unwrap();
        assert_eq!(got.file_name().unwrap(), want);
        assert_eq!(std::fs::read(&got).unwrap(), b"dummy-ffi");
        let cached = ensure_ffi_lib(None).unwrap();
        assert_eq!(cached, got);
        std::env::remove_var("ARIA_COMPUTE_HOME");
        match prev_lib {
            Some(v) => std::env::set_var("ARIAENGINE_FFI_LIB", v),
            None => std::env::remove_var("ARIAENGINE_FFI_LIB"),
        }
    }
}
