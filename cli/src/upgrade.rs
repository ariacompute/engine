//! Upgrade CLI binary + libaria-engine_ffi from GitHub/Gitee Releases.

use ariacompute_core::config::{self, ensure_aria_home};
use serde::Deserialize;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseAsset {
    pub name: String,
    pub download_url: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseInfo {
    pub tag: String,
    pub prerelease: bool,
    pub draft: bool,
    pub assets: Vec<ReleaseAsset>,
}

#[derive(Debug, Deserialize)]
struct GhRelease {
    tag_name: String,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    #[serde(default)]
    assets: Vec<GhAsset>,
}

#[derive(Debug, Deserialize)]
struct GhAsset {
    name: String,
    browser_download_url: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReleaseHost {
    GitHub,
    Gitee,
}

pub fn strip_v(v: &str) -> String {
    v.strip_prefix('v').unwrap_or(v).to_string()
}

pub fn detect_asset_os() -> io::Result<&'static str> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    match (os, arch) {
        ("linux", "x86_64") => Ok("linux_x86_64"),
        ("linux", "aarch64") => Ok("linux_arm64"),
        ("macos", _) => Ok("macos"),
        ("windows", "x86_64") => Ok("windows_x86_64"),
        _ => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("unsupported platform {os}/{arch} for upgrade"),
        )),
    }
}

pub fn engine_asset_name(version: &str, asset_os: &str) -> String {
    if asset_os.starts_with("windows") {
        format!("aria-engine_{version}_{asset_os}.zip")
    } else {
        format!("aria-engine_{version}_{asset_os}.tar.gz")
    }
}

fn detect_host(org_url: &str) -> io::Result<ReleaseHost> {
    let lower = org_url.to_ascii_lowercase();
    if lower.contains("gitee.com") {
        Ok(ReleaseHost::Gitee)
    } else if lower.contains("github.com") {
        Ok(ReleaseHost::GitHub)
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unsupported upgrade_url host: {org_url}"),
        ))
    }
}

fn owner_from_org(org: &str) -> io::Result<String> {
    let trimmed = org.trim_end_matches('/');
    let owner = trimmed
        .rsplit('/')
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "bad upgrade_url"))?;
    Ok(owner.to_string())
}

fn releases_api_url(host: ReleaseHost, org: &str) -> io::Result<String> {
    let owner = owner_from_org(org)?;
    Ok(match host {
        ReleaseHost::GitHub => {
            format!("https://api.github.com/repos/{owner}/engine/releases?per_page=30")
        }
        ReleaseHost::Gitee => {
            format!("https://gitee.com/api/v5/repos/{owner}/engine/releases?per_page=30")
        }
    })
}

async fn fetch_releases(host: ReleaseHost, org: &str) -> io::Result<Vec<ReleaseInfo>> {
    let url = releases_api_url(host, org)?;
    let client = reqwest::Client::builder()
        .user_agent(format!("aria-engine-upgrade/{}", env!("CARGO_PKG_VERSION")))
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(io::Error::other)?;
    let resp = client.get(&url).send().await.map_err(io::Error::other)?;
    if !resp.status().is_success() {
        return Err(io::Error::other(format!(
            "releases API {}: {}",
            resp.status(),
            url
        )));
    }
    let body: Vec<GhRelease> = resp
        .json()
        .await
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(body
        .into_iter()
        .map(|r| ReleaseInfo {
            tag: r.tag_name,
            draft: r.draft,
            prerelease: r.prerelease,
            assets: r
                .assets
                .into_iter()
                .map(|a| ReleaseAsset {
                    name: a.name,
                    download_url: a.browser_download_url,
                })
                .collect(),
        })
        .collect())
}

pub fn select_release<'a>(
    releases: &'a [ReleaseInfo],
    version: Option<&str>,
) -> io::Result<&'a ReleaseInfo> {
    let candidates: Vec<_> = releases.iter().filter(|r| !r.draft).collect();
    if candidates.is_empty() {
        return Err(io::Error::new(io::ErrorKind::NotFound, "no releases found"));
    }
    if let Some(v) = version {
        let want = strip_v(v);
        return candidates
            .into_iter()
            .find(|r| strip_v(&r.tag) == want)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, format!("release {want} not found"))
            });
    }
    if let Some(r) = candidates.iter().copied().find(|r| !r.prerelease) {
        return Ok(r);
    }
    candidates
        .first()
        .copied()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no stable release"))
}

fn find_asset<'a>(assets: &'a [ReleaseAsset], name: &str) -> io::Result<&'a ReleaseAsset> {
    assets
        .iter()
        .find(|a| a.name == name)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("asset {name} missing")))
}

async fn download_file(url: &str, dest: &Path) -> io::Result<()> {
    let client = reqwest::Client::builder()
        .user_agent(format!("aria-engine-upgrade/{}", env!("CARGO_PKG_VERSION")))
        .timeout(Duration::from_secs(300))
        .build()
        .map_err(io::Error::other)?;
    let bytes = client
        .get(url)
        .send()
        .await
        .map_err(io::Error::other)?
        .error_for_status()
        .map_err(io::Error::other)?
        .bytes()
        .await
        .map_err(io::Error::other)?;
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(dest, &bytes)?;
    Ok(())
}

fn extract_tar_gz(archive: &Path, dest_dir: &Path) -> io::Result<()> {
    let file = fs::File::open(archive)?;
    let dec = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(dec);
    archive.unpack(dest_dir)?;
    Ok(())
}

fn extract_zip(archive: &Path, dest_dir: &Path) -> io::Result<()> {
    let file = fs::File::open(archive)?;
    let mut zip =
        zip::ZipArchive::new(file).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    for i in 0..zip.len() {
        let mut entry = zip
            .by_index(i)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let out = dest_dir.join(entry.name());
        if entry.is_dir() {
            fs::create_dir_all(&out)?;
        } else {
            if let Some(p) = out.parent() {
                fs::create_dir_all(p)?;
            }
            let mut outfile = fs::File::create(&out)?;
            io::copy(&mut entry, &mut outfile)?;
        }
    }
    Ok(())
}

fn replace_exe(src: &Path) -> io::Result<()> {
    let current = std::env::current_exe()?;
    let tmp = current.with_extension(format!("upgrade-{}", std::process::id()));
    fs::copy(src, &tmp)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&tmp)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&tmp, perms)?;
    }
    fs::rename(&tmp, &current).or_else(|_| {
        // Windows may need replace via move of old
        let bak = current.with_extension("old");
        let _ = fs::remove_file(&bak);
        fs::rename(&current, &bak)?;
        fs::rename(&tmp, &current)
    })?;
    Ok(())
}

fn install_ffi(staging: &Path, lib_dir: &Path) -> io::Result<()> {
    fs::create_dir_all(lib_dir)?;
    let candidates = [
        "libaria-engine_ffi.so",
        "libaria-engine_ffi.dylib",
        "libaria-engine_ffi.a",
        "aria-engine_ffi.dll",
        "libaria_ffi.so",
        "libaria_ffi.dylib",
        "aria_ffi.dll",
    ];
    for name in candidates {
        let src = staging.join(name);
        if src.is_file() {
            let dest_name = match name {
                "libaria_ffi.so" => "libaria-engine_ffi.so",
                "libaria_ffi.dylib" => "libaria-engine_ffi.dylib",
                "aria_ffi.dll" => "aria-engine_ffi.dll",
                other => other,
            };
            fs::copy(&src, lib_dir.join(dest_name))?;
            return Ok(());
        }
    }
    // search one level deep
    for entry in fs::read_dir(staging)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() && install_ffi(&entry.path(), lib_dir).is_ok() {
            return Ok(());
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "libaria-engine_ffi not found in release archive",
    ))
}

/// Run `aria-engine upgrade [version]`.
pub async fn run(version: Option<&str>, current_version: &str) -> io::Result<()> {
    let cfg = config::load_config()?;
    let upgrade_url = cfg.upgrade_url.trim();
    if upgrade_url.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "upgrade_url not set; run `aria-engine setup` first",
        ));
    }
    let org = upgrade_url.trim_end_matches('/');
    let host = detect_host(org)?;
    let releases = fetch_releases(host, org).await?;
    let target = select_release(&releases, version)?;
    let ver = strip_v(&target.tag);
    let current = strip_v(current_version);
    if ver == current {
        println!("already at {ver}");
        return Ok(());
    }

    let asset_os = detect_asset_os()?;
    let engine_name = engine_asset_name(&ver, asset_os);
    let ffi_name = format!("libaria-engine_ffi_{ver}_{asset_os}.tar.gz");
    let engine_asset = find_asset(&target.assets, &engine_name)?;
    let ffi_asset = find_asset(&target.assets, &ffi_name)?;

    ensure_aria_home()?;
    let staging = config::aria_home()?.join(format!("tmp/upgrade-{ver}"));
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(&staging)?;

    let engine_archive = staging.join(&engine_name);
    let ffi_archive = staging.join(&ffi_name);
    println!("downloading {engine_name}…");
    download_file(&engine_asset.download_url, &engine_archive).await?;
    println!("downloading {ffi_name}…");
    download_file(&ffi_asset.download_url, &ffi_archive).await?;

    let bin_dir = staging.join("bin");
    let ffi_dir = staging.join("ffi");
    fs::create_dir_all(&bin_dir)?;
    fs::create_dir_all(&ffi_dir)?;
    if engine_name.ends_with(".zip") {
        extract_zip(&engine_archive, &bin_dir)?;
    } else {
        extract_tar_gz(&engine_archive, &bin_dir)?;
    }
    extract_tar_gz(&ffi_archive, &ffi_dir)?;

    let bin_name = if cfg!(windows) {
        "aria-engine.exe"
    } else {
        "aria-engine"
    };
    let bin_src = bin_dir.join(bin_name);
    if !bin_src.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("{bin_name} missing in archive"),
        ));
    }
    replace_exe(&bin_src)?;
    install_ffi(&ffi_dir, &config::lib_dir()?)?;
    let _ = fs::remove_dir_all(&staging);
    println!("upgraded to {ver}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_names() {
        assert_eq!(
            engine_asset_name("0.1.0", "linux_x86_64"),
            "aria-engine_0.1.0_linux_x86_64.tar.gz"
        );
        assert_eq!(
            engine_asset_name("0.1.0", "windows_x86_64"),
            "aria-engine_0.1.0_windows_x86_64.zip"
        );
    }
}
