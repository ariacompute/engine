//! Compare a local Aria bundle against the regional hub (count, names, SHA-256).

use crate::config::{self, AriaConfig};
use crate::download::{
    self, hub_file_urls, hub_repo_candidates, hub_token, io_err, parse_bundle_name,
    preferred_public_hub, DownloadSource,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

const LIST_TIMEOUT: Duration = Duration::from_secs(30);
const SMALL_GET_MAX: u64 = 8 * 1024 * 1024;
const HASH_BUF: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteFile {
    pub name: String,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileStatus {
    Ok,
    Missing,
    Extra,
    Mismatch { local: String, remote: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffRow {
    pub name: String,
    pub status: FileStatus,
}

pub fn skip_hub_name(name: &str) -> bool {
    let base = name.rsplit('/').next().unwrap_or(name);
    base.starts_with('.') || base == ".gitattributes" || base == ".gitignore"
}

pub fn basename(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_string()
}

pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

pub fn parse_sha256_etag(etag: &str) -> Option<String> {
    let mut s = etag.trim();
    if let Some(rest) = s.strip_prefix("W/") {
        s = rest.trim();
    }
    s = s.trim_matches('"').trim();
    if let Some(rest) = s.strip_prefix("sha256:") {
        s = rest.trim();
    }
    s = s.trim_matches('"').trim();
    if is_sha256_hex(s) {
        Some(s.to_ascii_lowercase())
    } else {
        None
    }
}

pub fn parse_hf_tree(json: &Value) -> Vec<RemoteFile> {
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
        if path.is_empty() {
            continue;
        }
        let name = basename(path);
        if skip_hub_name(&name) {
            continue;
        }
        let sha = entry
            .get("lfs")
            .and_then(|lfs| lfs.get("oid").or_else(|| lfs.get("sha256")))
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| is_sha256_hex(s));
        out.push(RemoteFile { name, sha256: sha });
    }
    dedupe_remote(out)
}

pub fn parse_ms_files(json: &Value) -> Vec<RemoteFile> {
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
        if path.is_empty() {
            continue;
        }
        let name = basename(path);
        if skip_hub_name(&name) {
            continue;
        }
        let sha = entry
            .get("Sha256")
            .or_else(|| entry.get("sha256"))
            .or_else(|| entry.get("Sha256Sum"))
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| is_sha256_hex(s));
        out.push(RemoteFile { name, sha256: sha });
    }
    dedupe_remote(out)
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

fn dedupe_remote(files: Vec<RemoteFile>) -> Vec<RemoteFile> {
    let mut map: BTreeMap<String, RemoteFile> = BTreeMap::new();
    for f in files {
        map.entry(f.name.clone()).or_insert(f);
    }
    map.into_values().collect()
}

pub fn list_local_files(dir: &Path) -> io::Result<BTreeMap<String, PathBuf>> {
    let mut map = BTreeMap::new();
    if !dir.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("not a bundle directory: {}", dir.display()),
        ));
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if skip_hub_name(&name) {
            continue;
        }
        map.insert(name, entry.path());
    }
    Ok(map)
}

pub fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; HASH_BUF];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Compare local SHA-256 map to remote SHA-256 map (keys = basenames).
pub fn diff_inventory(
    local: &BTreeMap<String, String>,
    remote: &BTreeMap<String, String>,
) -> Vec<DiffRow> {
    let mut names: Vec<_> = local.keys().chain(remote.keys()).cloned().collect();
    names.sort();
    names.dedup();
    let mut rows = Vec::new();
    for name in names {
        match (local.get(&name), remote.get(&name)) {
            (Some(l), Some(r)) if l == r => rows.push(DiffRow {
                name,
                status: FileStatus::Ok,
            }),
            (Some(l), Some(r)) => rows.push(DiffRow {
                name,
                status: FileStatus::Mismatch {
                    local: l.clone(),
                    remote: r.clone(),
                },
            }),
            (None, Some(_)) => rows.push(DiffRow {
                name,
                status: FileStatus::Missing,
            }),
            (Some(_), None) => rows.push(DiffRow {
                name,
                status: FileStatus::Extra,
            }),
            (None, None) => {}
        }
    }
    rows
}

pub async fn check_model(model: &str, cfg: &AriaConfig) -> io::Result<bool> {
    let dir = config::resolve_model_path(model)?;
    let bundle_name = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(model)
        .to_string();
    let bundle = parse_bundle_name(&bundle_name);
    let source = preferred_public_hub(&cfg.site_url);
    let (repo, mut remote) = fetch_remote_inventory(source, &bundle, cfg).await?;
    fill_missing_sha256(source, &bundle, cfg, &mut remote).await?;

    let local_paths = list_local_files(&dir)?;
    let mut local_hashes = BTreeMap::new();
    for (name, path) in &local_paths {
        local_hashes.insert(name.clone(), sha256_file(path)?);
    }
    let mut remote_hashes = BTreeMap::new();
    let mut missing_sha = Vec::new();
    for f in &remote {
        match &f.sha256 {
            Some(h) => {
                remote_hashes.insert(f.name.clone(), h.clone());
            }
            None => missing_sha.push(f.name.clone()),
        }
    }
    if !missing_sha.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{}: no SHA-256 from hub for {}",
                source.as_str(),
                missing_sha.join(", ")
            ),
        ));
    }

    let rows = diff_inventory(&local_hashes, &remote_hashes);
    let ok = rows.iter().all(|r| r.status == FileStatus::Ok);
    let count_ok = local_hashes.len() == remote_hashes.len();
    println!(
        "checking {bundle_name} against {} ({repo})",
        source.as_str()
    );
    println!(
        "  files: {} local / {} remote  {}",
        local_hashes.len(),
        remote_hashes.len(),
        if count_ok { "OK" } else { "FAIL" }
    );
    let width = rows.iter().map(|r| r.name.len()).max().unwrap_or(0);
    for row in &rows {
        match &row.status {
            FileStatus::Ok => {
                let hash = local_hashes
                    .get(&row.name)
                    .map(|s| s.as_str())
                    .unwrap_or("");
                println!(
                    "  {:<width$}  OK         {}",
                    row.name,
                    short_hash(hash),
                    width = width
                );
            }
            FileStatus::Missing => {
                println!("  {:<width$}  MISSING", row.name, width = width);
            }
            FileStatus::Extra => {
                println!("  {:<width$}  EXTRA", row.name, width = width);
            }
            FileStatus::Mismatch { local, remote } => {
                println!(
                    "  {:<width$}  MISMATCH   local={} remote={}",
                    row.name,
                    short_hash(local),
                    short_hash(remote),
                    width = width
                );
            }
        }
    }
    println!("{} {bundle_name}", if ok { "OK" } else { "FAIL" });
    Ok(ok)
}

pub async fn check_all(cfg: &AriaConfig) -> io::Result<bool> {
    let dir = config::models_dir()?;
    if !dir.exists() {
        println!("(no local models)");
        return Ok(true);
    }
    let mut names: Vec<String> = Vec::new();
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
    names.sort();
    if names.is_empty() {
        println!("(no local models)");
        return Ok(true);
    }
    let mut all_ok = true;
    for (i, name) in names.iter().enumerate() {
        if i > 0 {
            println!();
        }
        if !check_model(name, cfg).await? {
            all_ok = false;
        }
    }
    Ok(all_ok)
}

fn short_hash(h: &str) -> &str {
    if h.len() > 16 {
        &h[..16]
    } else {
        h
    }
}

async fn fetch_remote_inventory(
    source: DownloadSource,
    bundle: &download::BundleRef,
    cfg: &AriaConfig,
) -> io::Result<(String, Vec<RemoteFile>)> {
    let client = download::http_client(LIST_TIMEOUT).map_err(io_err)?;
    let mut last_err = None;
    for (repo, name) in hub_repo_candidates(source, bundle) {
        match list_hub_files(&client, source, &repo, &bundle.sdk, &name, cfg).await {
            Ok(files) if !files.is_empty() => return Ok((repo, files)),
            Ok(_) => {
                last_err = Some(format!("{repo}: empty listing"));
            }
            Err(e) => last_err = Some(format!("{repo}: {e}")),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "hub listing failed ({}): {}",
            source.as_str(),
            last_err.unwrap_or_else(|| "no repo candidates".into())
        ),
    ))
}

async fn list_hub_files(
    client: &reqwest::Client,
    source: DownloadSource,
    repo: &str,
    sdk: &str,
    name: &str,
    cfg: &AriaConfig,
) -> io::Result<Vec<RemoteFile>> {
    match source {
        DownloadSource::HuggingFace => {
            let url = format!(
                "https://huggingface.co/api/models/{repo}/tree/main/{sdk}/{name}?recursive=true"
            );
            let json = get_json_paginated(client, source, cfg, &url).await?;
            Ok(parse_hf_tree(&json))
        }
        DownloadSource::ModelScope => {
            let roots = [
                format!(
                    "https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true&Root={sdk}/{name}"
                ),
                format!(
                    "https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true&Root={sdk}/{name}"
                ),
            ];
            let mut last = None;
            for url in roots {
                match get_json(client, source, cfg, &url).await {
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
        DownloadSource::Dashboard => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "check does not use dashboard",
        )),
    }
}

async fn get_json(
    client: &reqwest::Client,
    source: DownloadSource,
    cfg: &AriaConfig,
    url: &str,
) -> io::Result<Value> {
    let mut req = client.get(url).header("Accept", "application/json");
    if let Some(token) = hub_token(source, cfg) {
        req = req.bearer_auth(token);
    }
    let resp = req.send().await.map_err(io_err)?;
    let status = resp.status();
    if status.as_u16() == 401 || status.as_u16() == 403 {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "auth failed HTTP {}; run `aria-engine setup` to set {}",
                status,
                match source {
                    DownloadSource::HuggingFace => "hf_token",
                    DownloadSource::ModelScope => "modelscope_api_token",
                    DownloadSource::Dashboard => "hf_token",
                }
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
    source: DownloadSource,
    cfg: &AriaConfig,
    start_url: &str,
) -> io::Result<Value> {
    let mut url = start_url.to_string();
    let mut all = Vec::new();
    for _ in 0..32 {
        let mut req = client.get(&url).header("Accept", "application/json");
        if let Some(token) = hub_token(source, cfg) {
            req = req.bearer_auth(token);
        }
        let resp = req.send().await.map_err(io_err)?;
        let status = resp.status();
        if status.as_u16() == 401 || status.as_u16() == 403 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "auth failed HTTP {}; run `aria-engine setup` to set hf_token",
                    status
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

async fn fill_missing_sha256(
    source: DownloadSource,
    bundle: &download::BundleRef,
    cfg: &AriaConfig,
    files: &mut [RemoteFile],
) -> io::Result<()> {
    let client = download::http_client(LIST_TIMEOUT).map_err(io_err)?;
    for f in files.iter_mut() {
        if f.sha256.is_some() {
            continue;
        }
        f.sha256 = Some(fetch_file_sha256(source, bundle, cfg, &client, &f.name).await?);
    }
    Ok(())
}

async fn fetch_file_sha256(
    source: DownloadSource,
    bundle: &download::BundleRef,
    cfg: &AriaConfig,
    client: &reqwest::Client,
    file: &str,
) -> io::Result<String> {
    let mut last = None;
    for url in hub_file_urls(source, bundle, file) {
        match try_remote_sha256(client, source, cfg, &url).await {
            Ok(h) => return Ok(h),
            Err(e) => last = Some(e),
        }
    }
    Err(last.unwrap_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("no SHA-256 for {file} on {}", source.as_str()),
        )
    }))
}

async fn try_remote_sha256(
    client: &reqwest::Client,
    source: DownloadSource,
    cfg: &AriaConfig,
    url: &str,
) -> io::Result<String> {
    let mut head = client.head(url);
    if let Some(token) = hub_token(source, cfg) {
        head = head.bearer_auth(token);
    }
    if let Ok(resp) = head.send().await {
        if resp.status().is_success() || resp.status().as_u16() == 302 {
            if let Some(h) = sha_from_headers(resp.headers()) {
                return Ok(h);
            }
            if let Some(len) = resp.content_length() {
                if len > SMALL_GET_MAX {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("no SHA-256 metadata and file too large to fetch ({len} bytes)"),
                    ));
                }
            }
        }
    }

    let mut get = client.get(url);
    if let Some(token) = hub_token(source, cfg) {
        get = get.bearer_auth(token);
    }
    let resp = get.send().await.map_err(io_err)?;
    if !resp.status().is_success() {
        return Err(io::Error::other(format!(
            "hub HTTP {} for {url}",
            resp.status()
        )));
    }
    if let Some(h) = sha_from_headers(resp.headers()) {
        return Ok(h);
    }
    if let Some(len) = resp.content_length() {
        if len > SMALL_GET_MAX {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("no SHA-256 metadata and file too large to fetch ({len} bytes)"),
            ));
        }
    }
    let bytes = resp.bytes().await.map_err(io_err)?;
    if bytes.len() as u64 > SMALL_GET_MAX {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "no SHA-256 metadata and file too large to fetch",
        ));
    }
    Ok(sha256_bytes(&bytes))
}

fn sha_from_headers(headers: &reqwest::header::HeaderMap) -> Option<String> {
    for key in ["x-linked-etag", "etag", "x-amz-meta-sha256"] {
        if let Some(v) = headers.get(key).and_then(|v| v.to_str().ok()) {
            if let Some(h) = parse_sha256_etag(v) {
                return Some(h);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn skip_dotfiles_and_git_attrs() {
        assert!(skip_hub_name(".gitattributes"));
        assert!(skip_hub_name(".gitignore"));
        assert!(skip_hub_name(".hidden"));
        assert!(skip_hub_name("v1.0/bundle/.gitattributes"));
        assert!(!skip_hub_name("config.json"));
        assert!(!skip_hub_name("weight.bin"));
        assert!(!skip_hub_name("README.md"));
    }

    #[test]
    fn parse_hf_tree_uses_lfs_oid() {
        let json = serde_json::json!([
            {
                "type": "directory",
                "path": "v1.0/demo_q4"
            },
            {
                "type": "file",
                "path": "v1.0/demo_q4/config.json",
                "oid": "aabbcc"
            },
            {
                "type": "file",
                "path": "v1.0/demo_q4/weight.bin",
                "lfs": {
                    "oid": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "size": 12
                }
            },
            {
                "type": "file",
                "path": "v1.0/demo_q4/.gitattributes"
            }
        ]);
        let files = parse_hf_tree(&json);
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].name, "config.json");
        assert!(files[0].sha256.is_none());
        assert_eq!(files[1].name, "weight.bin");
        assert_eq!(
            files[1].sha256.as_deref(),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        );
    }

    #[test]
    fn parse_ms_files_uses_sha256() {
        let json = serde_json::json!({
            "Code": 200,
            "Data": {
                "Files": [
                    {"Name": ".", "Path": "v1.0/demo_q4", "Type": "tree"},
                    {
                        "Name": "config.json",
                        "Path": "v1.0/demo_q4/config.json",
                        "Type": "blob",
                        "Sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                    },
                    {
                        "Name": ".gitattributes",
                        "Path": "v1.0/demo_q4/.gitattributes",
                        "Type": "blob",
                        "Sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
                    }
                ]
            }
        });
        let files = parse_ms_files(&json);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].name, "config.json");
        assert_eq!(
            files[0].sha256.as_deref(),
            Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        );
    }

    #[test]
    fn diff_inventory_equal_missing_extra_mismatch() {
        let mut local = BTreeMap::new();
        local.insert("a".into(), "11".into());
        local.insert("b".into(), "22".into());
        local.insert("extra".into(), "99".into());
        let mut remote = BTreeMap::new();
        remote.insert("a".into(), "11".into());
        remote.insert("b".into(), "33".into());
        remote.insert("miss".into(), "44".into());
        let rows = diff_inventory(&local, &remote);
        let by_name: BTreeMap<_, _> = rows.into_iter().map(|r| (r.name, r.status)).collect();
        assert_eq!(by_name.get("a"), Some(&FileStatus::Ok));
        assert_eq!(
            by_name.get("b"),
            Some(&FileStatus::Mismatch {
                local: "22".into(),
                remote: "33".into()
            })
        );
        assert_eq!(by_name.get("miss"), Some(&FileStatus::Missing));
        assert_eq!(by_name.get("extra"), Some(&FileStatus::Extra));
    }

    #[test]
    fn sha256_file_matches_known_digest() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sample.txt");
        {
            let mut f = fs::File::create(&path).unwrap();
            f.write_all(b"hello").unwrap();
        }
        let got = sha256_file(&path).unwrap();
        assert_eq!(
            got,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        assert_eq!(got, sha256_bytes(b"hello"));
    }

    #[test]
    fn parse_sha256_etag_accepts_quoted_and_sha_prefix() {
        let h = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        assert_eq!(parse_sha256_etag(&format!("\"{h}\"")).as_deref(), Some(h));
        assert_eq!(parse_sha256_etag(&format!("W/\"{h}\"")).as_deref(), Some(h));
        assert_eq!(
            parse_sha256_etag(&format!("sha256:{h}")).as_deref(),
            Some(h)
        );
        assert!(parse_sha256_etag("\"abc\"").is_none());
        assert!(parse_sha256_etag("\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"").is_none());
    }

    #[test]
    fn list_local_files_skips_dirs_and_dotfiles() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("config.json"), b"{}").unwrap();
        fs::write(dir.path().join(".gitattributes"), b"x").unwrap();
        fs::create_dir(dir.path().join("nested")).unwrap();
        let files = list_local_files(dir.path()).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files.contains_key("config.json"));
    }
}
