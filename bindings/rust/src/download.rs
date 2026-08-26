//! Regional public-hub model auto-download for the Rust SDK.
//!
//! Matches `aria-engine download`: `.com` → Hugging Face, `.cn` → ModelScope.
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

fn config_yml_scalar(key: &str) -> Option<String> {
    let path = aria_home().ok()?.join("config.yml");
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
        "auth failed HTTP {code}; set {field} via aria-engine auth (do not pass a Dashboard sk-/bfvk- key as the hub token)"
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
/// then `~/.ariacompute/config.yml` (same keys as `aria-engine auth`).
/// Dashboard `sk-` / `bfvk-` keys are not sent to the hub.
pub fn download_model(
    model: &str,
    token: &str,
    site: Option<&str>,
) -> Result<PathBuf, DownloadError> {
    download_model_auth(model, token, site, None, None)
}

/// Like [`download_model`], with named hub tokens matching `aria-engine auth`.
pub fn download_model_auth(
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

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
}
