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

/// Release API roots (overridable so tests can point at a local mock server).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseApi {
    pub github: String,
    pub gitee: String,
}

impl Default for ReleaseApi {
    fn default() -> Self {
        Self {
            github: "https://api.github.com".into(),
            gitee: "https://gitee.com/api/v5".into(),
        }
    }
}

impl ReleaseApi {
    /// Production roots, optionally replaced by `ARIA_UPGRADE_API_GITHUB` /
    /// `ARIA_UPGRADE_API_GITEE` (test seam; empty values are ignored).
    pub fn from_env() -> Self {
        let default = Self::default();
        Self {
            github: non_empty_env("ARIA_UPGRADE_API_GITHUB").unwrap_or(default.github),
            gitee: non_empty_env("ARIA_UPGRADE_API_GITEE").unwrap_or(default.gitee),
        }
    }

    fn releases_url(&self, host: ReleaseHost, org: &str) -> io::Result<String> {
        let owner = owner_from_org(org)?;
        Ok(match host {
            ReleaseHost::GitHub => format!(
                "{}/repos/{owner}/engine/releases?per_page=30",
                self.github.trim_end_matches('/')
            ),
            ReleaseHost::Gitee => format!(
                "{}/repos/{owner}/engine/releases?per_page=30",
                self.gitee.trim_end_matches('/')
            ),
        })
    }
}

fn non_empty_env(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

pub fn strip_v(v: &str) -> String {
    v.strip_prefix('v').unwrap_or(v).to_string()
}

/// Map a `(std::env::consts::OS, ARCH)` pair onto a release asset triple.
pub fn asset_os_for(os: &str, arch: &str) -> io::Result<&'static str> {
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

pub fn detect_asset_os() -> io::Result<&'static str> {
    asset_os_for(std::env::consts::OS, std::env::consts::ARCH)
}

pub fn engine_asset_name(version: &str, asset_os: &str) -> String {
    if asset_os.starts_with("windows") {
        format!("aria-engine_{version}_{asset_os}.zip")
    } else {
        format!("aria-engine_{version}_{asset_os}.tar.gz")
    }
}

/// Binary name inside the engine archive for a given asset triple.
pub fn engine_bin_name(asset_os: &str) -> &'static str {
    if asset_os.starts_with("windows") {
        "aria-engine.exe"
    } else {
        "aria-engine"
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

async fn fetch_releases(
    api: &ReleaseApi,
    host: ReleaseHost,
    org: &str,
) -> io::Result<Vec<ReleaseInfo>> {
    let url = api.releases_url(host, org)?;
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
    ensure_parent(dest)?;
    fs::write(dest, &bytes)?;
    Ok(())
}

/// Create the parent directory of `path` (no-op at a filesystem root).
fn ensure_parent(path: &Path) -> io::Result<()> {
    match path.parent() {
        Some(parent) => fs::create_dir_all(parent),
        None => Ok(()),
    }
}

fn extract_tar_gz(archive: &Path, dest_dir: &Path) -> io::Result<()> {
    let file = fs::File::open(archive)?;
    let dec = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(dec);
    archive.unpack(dest_dir)?;
    Ok(())
}

/// Unpack the engine archive, picking the extractor from the asset name.
fn extract_engine_archive(archive_name: &str, archive: &Path, dest_dir: &Path) -> io::Result<()> {
    if archive_name.ends_with(".zip") {
        extract_zip(archive, dest_dir)
    } else {
        extract_tar_gz(archive, dest_dir)
    }
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
            ensure_parent(&out)?;
            let mut outfile = fs::File::create(&out)?;
            io::copy(&mut entry, &mut outfile)?;
        }
    }
    Ok(())
}

/// Atomically put `src` in place of `current` (never touches the running binary
/// unless `current` is it — tests pass a throwaway path).
fn replace_exe(current: &Path, src: &Path) -> io::Result<()> {
    let tmp = current.with_extension(format!("upgrade-{}", std::process::id()));
    ensure_parent(current)?;
    fs::copy(src, &tmp)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&tmp)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&tmp, perms)?;
    }
    fs::rename(&tmp, current).or_else(|_| {
        // Windows may need replace via move of old
        let bak = current.with_extension("old");
        let _ = fs::remove_file(&bak);
        fs::rename(current, &bak)?;
        fs::rename(&tmp, current)
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
    let found = fs::read_dir(staging)?
        .flatten()
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .any(|e| install_ffi(&e.path(), lib_dir).is_ok());
    if found {
        return Ok(());
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "libaria-engine_ffi not found in release archive",
    ))
}

/// Run `aria-engine upgrade [version]` against the configured release host.
pub async fn run(version: Option<&str>, current_version: &str) -> io::Result<()> {
    let api = ReleaseApi::from_env();
    let exe = std::env::current_exe()?;
    run_with(version, current_version, &api, &exe).await
}

/// `run` with the release API roots and the binary to replace made explicit.
pub async fn run_with(
    version: Option<&str>,
    current_version: &str,
    api: &ReleaseApi,
    exe: &Path,
) -> io::Result<()> {
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
    let releases = fetch_releases(api, host, org).await?;
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
    extract_engine_archive(&engine_name, &engine_archive, &bin_dir)?;
    extract_tar_gz(&ffi_archive, &ffi_dir)?;

    let bin_name = engine_bin_name(asset_os);
    let bin_src = bin_dir.join(bin_name);
    if !bin_src.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("{bin_name} missing in archive"),
        ));
    }
    replace_exe(exe, &bin_src)?;
    install_ffi(&ffi_dir, &config::lib_dir()?)?;
    let _ = fs::remove_dir_all(&staging);
    println!("upgraded to {ver}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_env::with_home;
    use ariacompute_core::config::AriaConfig;
    use serde_json::json;
    use std::io::Write;
    use tempfile::tempdir;
    use wiremock::matchers::{method, path as mock_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn release(tag: &str, prerelease: bool, draft: bool, assets: &[&str]) -> ReleaseInfo {
        ReleaseInfo {
            tag: tag.into(),
            prerelease,
            draft,
            assets: assets
                .iter()
                .map(|n| ReleaseAsset {
                    name: (*n).into(),
                    download_url: format!("https://example.invalid/{n}"),
                })
                .collect(),
        }
    }

    fn tar_gz(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        {
            let mut builder = tar::Builder::new(&mut enc);
            for (name, data) in entries {
                let mut header = tar::Header::new_gnu();
                header.set_size(data.len() as u64);
                header.set_mode(0o644);
                header.set_path(name).unwrap();
                header.set_cksum();
                builder.append(&header, *data).unwrap();
            }
            builder.finish().unwrap();
        }
        enc.finish().unwrap()
    }

    fn zip_archive(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let cursor = std::io::Cursor::new(Vec::new());
        let mut writer = zip::ZipWriter::new(cursor);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        for (name, data) in entries {
            writer.start_file(*name, options).unwrap();
            writer.write_all(data).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }

    /// Engine archive bytes for the asset name (format follows the asset suffix).
    fn engine_archive_bytes(asset_name: &str, bin_name: &str) -> Vec<u8> {
        if asset_name.ends_with(".zip") {
            zip_archive(&[(bin_name, b"new-binary")])
        } else {
            tar_gz(&[(bin_name, b"new-binary")])
        }
    }

    fn write_config(upgrade_url: &str) {
        config::save_config(&AriaConfig {
            site_url: "https://ariacompute.com".into(),
            upgrade_url: upgrade_url.into(),
            compute: "auto".into(),
            hf_token: String::new(),
            modelscope_api_token: String::new(),
        })
        .unwrap();
    }

    #[test]
    fn strip_v_and_asset_names() {
        assert_eq!(strip_v("v1.2.3"), "1.2.3");
        assert_eq!(strip_v("1.2.3"), "1.2.3");
        assert_eq!(
            engine_asset_name("0.1.0", "linux_x86_64"),
            "aria-engine_0.1.0_linux_x86_64.tar.gz"
        );
        assert_eq!(
            engine_asset_name("0.1.0", "windows_x86_64"),
            "aria-engine_0.1.0_windows_x86_64.zip"
        );
        // the host platform always resolves to a known asset triple
        let os = detect_asset_os().unwrap();
        assert!(engine_asset_name("9.9.9", os).contains("aria-engine_9.9.9_"));
    }

    #[test]
    fn asset_os_covers_every_supported_platform() {
        assert_eq!(asset_os_for("linux", "x86_64").unwrap(), "linux_x86_64");
        assert_eq!(asset_os_for("linux", "aarch64").unwrap(), "linux_arm64");
        assert_eq!(asset_os_for("macos", "aarch64").unwrap(), "macos");
        assert_eq!(asset_os_for("windows", "x86_64").unwrap(), "windows_x86_64");
        // unsupported host / arch pair
        let err = asset_os_for("windows", "aarch64").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
        assert!(err.to_string().contains("unsupported platform"));

        assert_eq!(engine_bin_name("windows_x86_64"), "aria-engine.exe");
        assert_eq!(engine_bin_name("linux_x86_64"), "aria-engine");
    }

    #[test]
    fn extract_engine_archive_picks_the_extractor_by_suffix() {
        let dir = tempdir().unwrap();
        let tgz = dir.path().join("engine.tar.gz");
        fs::write(&tgz, tar_gz(&[("aria-engine", b"tar")])).unwrap();
        let tar_out = dir.path().join("tar");
        extract_engine_archive("aria-engine_1_linux_x86_64.tar.gz", &tgz, &tar_out).unwrap();
        assert_eq!(fs::read(tar_out.join("aria-engine")).unwrap(), b"tar");

        let zip_path = dir.path().join("engine.zip");
        fs::write(&zip_path, zip_archive(&[("aria-engine.exe", b"zip")])).unwrap();
        let zip_out = dir.path().join("zip");
        extract_engine_archive("aria-engine_1_windows_x86_64.zip", &zip_path, &zip_out).unwrap();
        assert_eq!(fs::read(zip_out.join("aria-engine.exe")).unwrap(), b"zip");

        // the archive builder helper honours the same suffix convention
        assert!(!engine_archive_bytes("x.zip", "aria-engine.exe").is_empty());
        assert!(!engine_archive_bytes("x.tar.gz", "aria-engine").is_empty());
    }

    #[test]
    fn detect_host_and_owner_from_org() {
        assert_eq!(
            detect_host("https://github.com/ariacompute").unwrap(),
            ReleaseHost::GitHub
        );
        assert_eq!(
            detect_host("https://GITEE.com/ariacompute").unwrap(),
            ReleaseHost::Gitee
        );
        let err = detect_host("https://example.com/ariacompute").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("unsupported upgrade_url host"));

        assert_eq!(owner_from_org("https://github.com/acme/").unwrap(), "acme");
        assert_eq!(owner_from_org("https://github.com/acme").unwrap(), "acme");
        assert!(owner_from_org("").is_err());
        assert!(owner_from_org("/").is_err());
        // a bare host degrades to the host name; `run` never reaches it because
        // `detect_host` only accepts known hosts and `upgrade_url` is validated first
        assert_eq!(owner_from_org("https://github.com/").unwrap(), "github.com");
    }

    #[test]
    fn releases_url_per_host_and_env_override() {
        let api = ReleaseApi::default();
        assert_eq!(
            api.releases_url(ReleaseHost::GitHub, "https://github.com/acme")
                .unwrap(),
            "https://api.github.com/repos/acme/engine/releases?per_page=30"
        );
        assert_eq!(
            api.releases_url(ReleaseHost::Gitee, "https://gitee.com/acme")
                .unwrap(),
            "https://gitee.com/api/v5/repos/acme/engine/releases?per_page=30"
        );

        let custom = ReleaseApi {
            github: "http://127.0.0.1:1/github/".into(),
            gitee: "http://127.0.0.1:1/gitee/".into(),
        };
        assert_eq!(
            custom
                .releases_url(ReleaseHost::GitHub, "https://github.com/acme")
                .unwrap(),
            "http://127.0.0.1:1/github/repos/acme/engine/releases?per_page=30"
        );
        assert_eq!(
            custom
                .releases_url(ReleaseHost::Gitee, "https://gitee.com/acme")
                .unwrap(),
            "http://127.0.0.1:1/gitee/repos/acme/engine/releases?per_page=30"
        );
    }

    #[test]
    #[serial_test::serial]
    fn release_api_from_env_reads_overrides() {
        let _guard = crate::test_env::lock_env();
        std::env::set_var("ARIA_UPGRADE_API_GITHUB", "http://127.0.0.1:9/gh");
        std::env::set_var("ARIA_UPGRADE_API_GITEE", "   ");
        let api = ReleaseApi::from_env();
        assert_eq!(api.github, "http://127.0.0.1:9/gh");
        // blank values keep the production default
        assert_eq!(api.gitee, ReleaseApi::default().gitee);
        std::env::remove_var("ARIA_UPGRADE_API_GITHUB");
        std::env::remove_var("ARIA_UPGRADE_API_GITEE");
    }

    #[test]
    fn select_release_honours_drafts_prereleases_and_version() {
        assert_eq!(
            select_release(&[], None).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );

        let draft_only = vec![release("v0.1.0", false, true, &[])];
        assert!(select_release(&draft_only, None).is_err());

        let releases = vec![
            release(
                "v0.3.0",
                true,
                false,
                &["aria-engine_0.3.0_linux_x86_64.tar.gz"],
            ),
            release(
                "v0.2.0",
                false,
                false,
                &["aria-engine_0.2.0_linux_x86_64.tar.gz"],
            ),
            release("v0.1.0", false, true, &[]),
        ];
        assert_eq!(
            releases[1].assets[0].name,
            "aria-engine_0.2.0_linux_x86_64.tar.gz"
        );
        assert_eq!(select_release(&releases, None).unwrap().tag, "v0.2.0");
        assert_eq!(
            select_release(&releases, Some("0.3.0")).unwrap().tag,
            "v0.3.0"
        );
        assert_eq!(
            select_release(&releases, Some("v0.2.0")).unwrap().tag,
            "v0.2.0"
        );
        // drafts are never selectable, even by explicit version
        assert!(select_release(&releases, Some("0.1.0")).is_err());
        assert!(select_release(&releases, Some("9.9.9")).is_err());

        // only prereleases available → newest prerelease wins
        let only_pre = vec![release("v0.4.0", true, false, &[])];
        assert_eq!(select_release(&only_pre, None).unwrap().tag, "v0.4.0");
    }

    #[test]
    fn find_asset_reports_missing_names() {
        let assets = vec![ReleaseAsset {
            name: "a.tar.gz".into(),
            download_url: "http://x/a".into(),
        }];
        assert_eq!(find_asset(&assets, "a.tar.gz").unwrap().name, "a.tar.gz");
        let err = find_asset(&assets, "b.tar.gz").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        assert!(err.to_string().contains("asset b.tar.gz missing"));
    }

    #[test]
    fn extract_tar_gz_and_zip_unpack_nested_entries() {
        let dir = tempdir().unwrap();
        let tgz = dir.path().join("a.tar.gz");
        fs::write(&tgz, tar_gz(&[("nested/bin.txt", b"hi")])).unwrap();
        extract_tar_gz(&tgz, dir.path()).unwrap();
        assert_eq!(
            fs::read_to_string(dir.path().join("nested/bin.txt")).unwrap(),
            "hi"
        );

        let zip_path = dir.path().join("a.zip");
        fs::write(
            &zip_path,
            zip_archive(&[
                ("aria-engine", b"bin"),
                ("lib/x.so", b"so"),
                ("empty-dir/", b""),
            ]),
        )
        .unwrap();
        let out = dir.path().join("unzipped");
        extract_zip(&zip_path, &out).unwrap();
        assert_eq!(fs::read_to_string(out.join("aria-engine")).unwrap(), "bin");
        assert_eq!(fs::read_to_string(out.join("lib/x.so")).unwrap(), "so");
        assert!(out.join("empty-dir").is_dir());

        // a corrupt archive surfaces as InvalidData
        let bad = dir.path().join("bad.zip");
        fs::write(&bad, b"not a zip").unwrap();
        let err = extract_zip(&bad, &out).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn install_ffi_finds_renamed_and_nested_libraries() {
        let dir = tempdir().unwrap();
        let lib_dir = dir.path().join("lib");

        // legacy name is published under the release name
        let staging = dir.path().join("staging-legacy");
        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("libaria_ffi.so"), b"legacy").unwrap();
        install_ffi(&staging, &lib_dir).unwrap();
        assert_eq!(
            fs::read_to_string(lib_dir.join("libaria-engine_ffi.so")).unwrap(),
            "legacy"
        );

        // canonical name inside a nested directory is found one level deep
        let nested = dir.path().join("staging-nested/lib");
        fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join("aria_ffi.dll"), b"dll").unwrap();
        install_ffi(&dir.path().join("staging-nested"), &lib_dir).unwrap();
        assert_eq!(
            fs::read_to_string(lib_dir.join("aria-engine_ffi.dll")).unwrap(),
            "dll"
        );

        let empty = dir.path().join("staging-empty");
        fs::create_dir_all(&empty).unwrap();
        let err = install_ffi(&empty, &lib_dir).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        assert!(err.to_string().contains("libaria-engine_ffi not found"));
    }

    #[test]
    fn ensure_parent_creates_dirs_and_tolerates_roots() {
        let dir = tempdir().unwrap();
        let nested = dir.path().join("bin/aria-engine");
        ensure_parent(&nested).unwrap();
        assert!(dir.path().join("bin").is_dir());
        ensure_parent(Path::new("/")).unwrap();
    }

    #[test]
    fn replace_exe_swaps_the_target_file() {
        let dir = tempdir().unwrap();
        let target = dir.path().join("bin/aria-engine");
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(&target, b"old").unwrap();
        let src = dir.path().join("new");
        fs::write(&src, b"new").unwrap();

        replace_exe(&target, &src).unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"new");
        assert!(!target
            .with_extension(format!("upgrade-{}", std::process::id()))
            .exists());

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&target).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o755);
        }

        // a target that cannot be renamed over (a directory) exercises the
        // move-old-aside fallback instead of failing outright
        let blocked = dir.path().join("blocked");
        fs::create_dir_all(blocked.join("child")).unwrap();
        replace_exe(&blocked, &src).unwrap();
        assert_eq!(fs::read(&blocked).unwrap(), b"new");
        assert!(blocked.with_extension("old").join("child").is_dir());
    }

    #[test]
    fn run_rejects_missing_upgrade_url_and_bad_host() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        with_home(|home| {
            rt.block_on(async {
                let exe = home.join("bin/aria-engine");
                let api = ReleaseApi::default();

                let err = run_with(None, "0.1.0", &api, &exe).await.unwrap_err();
                assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
                assert!(err.to_string().contains("upgrade_url not set"));

                write_config("https://example.com/acme");
                let err = run_with(None, "0.1.0", &api, &exe).await.unwrap_err();
                assert!(err.to_string().contains("unsupported upgrade_url host"));
            });
        });
    }

    #[test]
    fn run_installs_binary_and_ffi_from_a_release() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        with_home(|home| {
            rt.block_on(async {
                let server = MockServer::start().await;
                let os = detect_asset_os().unwrap();
                let engine_name = engine_asset_name("0.2.0", os);
                let ffi_name = format!("libaria-engine_ffi_0.2.0_{os}.tar.gz");
                let bin_name = engine_bin_name(os);
                let engine_bytes = engine_archive_bytes(&engine_name, bin_name);
                let ffi_bytes = tar_gz(&[("libaria-engine_ffi.so", b"ffi")]);
                Mock::given(method("GET"))
                    .and(mock_path("/dl/engine"))
                    .respond_with(ResponseTemplate::new(200).set_body_bytes(engine_bytes))
                    .mount(&server)
                    .await;
                Mock::given(method("GET"))
                    .and(mock_path("/dl/ffi"))
                    .respond_with(ResponseTemplate::new(200).set_body_bytes(ffi_bytes))
                    .mount(&server)
                    .await;

                Mock::given(method("GET"))
                    .and(mock_path("/repos/acme/engine/releases"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                        {
                            "tag_name": "v0.2.0",
                            "draft": false,
                            "prerelease": false,
                            "assets": [
                                {"name": engine_name, "browser_download_url": format!("{}/dl/engine", server.uri())},
                                {"name": ffi_name, "browser_download_url": format!("{}/dl/ffi", server.uri())}
                            ]
                        }
                    ])))
                    .mount(&server)
                    .await;

                write_config("https://github.com/acme/");
                let api = ReleaseApi {
                    github: server.uri(),
                    gitee: server.uri(),
                };
                let exe = home.join("bin/aria-engine");
                fs::create_dir_all(exe.parent().unwrap()).unwrap();
                fs::write(&exe, b"old-binary").unwrap();
                // stale staging from an earlier attempt is cleared first
                let stale = home.join("tmp/upgrade-0.2.0");
                fs::create_dir_all(&stale).unwrap();
                fs::write(stale.join("leftover"), b"x").unwrap();

                run_with(None, "0.1.0", &api, &exe).await.unwrap();
                assert_eq!(fs::read(&exe).unwrap(), b"new-binary");
                assert_eq!(
                    fs::read_to_string(home.join("lib/libaria-engine_ffi.so")).unwrap(),
                    "ffi"
                );
                // staging is cleaned up after a successful upgrade
                assert!(!home.join("tmp/upgrade-0.2.0").exists());

                // a second run against the same version is a no-op
                run_with(Some("0.2.0"), "v0.2.0", &api, &exe).await.unwrap();
            });
        });
    }

    #[test]
    fn run_reports_missing_assets_and_broken_archives() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        with_home(|home| {
            rt.block_on(async {
                let server = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/repos/acme/engine/releases"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                        {"tag_name": "v0.3.0", "draft": false, "prerelease": false, "assets": []}
                    ])))
                    .mount(&server)
                    .await;
                write_config("https://github.com/acme");
                let api = ReleaseApi {
                    github: server.uri(),
                    gitee: server.uri(),
                };
                let exe = home.join("bin/aria-engine");
                let err = run_with(None, "0.1.0", &api, &exe).await.unwrap_err();
                assert!(err.to_string().contains("missing"));

                // releases API failure is reported with the status
                let broken = MockServer::start().await;
                Mock::given(method("GET"))
                    .and(mock_path("/repos/acme/engine/releases"))
                    .respond_with(ResponseTemplate::new(503))
                    .mount(&broken)
                    .await;
                let api = ReleaseApi {
                    github: broken.uri(),
                    gitee: broken.uri(),
                };
                let err = run_with(None, "0.1.0", &api, &exe).await.unwrap_err();
                assert!(err.to_string().contains("releases API"));
            });
        });
    }

    #[test]
    fn run_rejects_archives_without_the_binary() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        with_home(|home| {
            rt.block_on(async {
                let server = MockServer::start().await;
                let os = detect_asset_os().unwrap();
                let engine_name = engine_asset_name("0.4.0", os);
                let ffi_name = format!("libaria-engine_ffi_0.4.0_{os}.tar.gz");
                Mock::given(method("GET"))
                    .and(mock_path("/dl/engine"))
                    .respond_with(
                        ResponseTemplate::new(200).set_body_bytes(tar_gz(&[("readme.txt", b"x")])),
                    )
                    .mount(&server)
                    .await;
                Mock::given(method("GET"))
                    .and(mock_path("/dl/ffi"))
                    .respond_with(
                        ResponseTemplate::new(200)
                            .set_body_bytes(tar_gz(&[("libaria-engine_ffi.so", b"ffi")])),
                    )
                    .mount(&server)
                    .await;
                Mock::given(method("GET"))
                    .and(mock_path("/repos/acme/engine/releases"))
                    .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                        {
                            "tag_name": "v0.4.0",
                            "draft": false,
                            "prerelease": false,
                            "assets": [
                                {"name": engine_name, "browser_download_url": format!("{}/dl/engine", server.uri())},
                                {"name": ffi_name, "browser_download_url": format!("{}/dl/ffi", server.uri())}
                            ]
                        }
                    ])))
                    .mount(&server)
                    .await;

                write_config("https://github.com/acme");
                let api = ReleaseApi {
                    github: server.uri(),
                    gitee: server.uri(),
                };
                let err = run_with(None, "0.1.0", &api, &home.join("bin/aria-engine"))
                    .await
                    .unwrap_err();
                assert_eq!(err.kind(), io::ErrorKind::NotFound);
                assert!(err.to_string().contains("missing in archive"));
            });
        });
    }
}
