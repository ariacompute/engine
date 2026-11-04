//! End-to-end CLI tests: drive the real `aria-engine` binary.
//!
//! Every case is offline — hub / release traffic is either pointed at an
//! unreachable loopback port (fast, deterministic failure) or never reached.

use assert_cmd::prelude::*;
use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};

fn home() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

/// `aria-engine` binary pinned to a throwaway `ARIA_COMPUTE_HOME`.
fn engine(home: impl AsRef<Path>) -> Command {
    let mut cmd = Command::cargo_bin("aria-engine").unwrap();
    cmd.env("ARIA_COMPUTE_HOME", home.as_ref());
    // never touch real hubs / release hosts
    cmd.env("ARIA_HF_ENDPOINT", "http://127.0.0.1:1");
    cmd.env("ARIA_MS_ENDPOINT", "http://127.0.0.1:1");
    cmd.env("ARIA_MS_ENDPOINT_ALT", "http://127.0.0.1:1");
    cmd.env("ARIA_UPGRADE_API_GITHUB", "http://127.0.0.1:1");
    cmd.env("ARIA_UPGRADE_API_GITEE", "http://127.0.0.1:1");
    cmd
}

fn write_encoder_checkpoint(home: &Path) -> std::path::PathBuf {
    let dir = home.join("models/afm-de");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("rl_agent_config.json"), "{}").unwrap();
    dir
}

fn write_decoder_checkpoint(home: &Path) -> std::path::PathBuf {
    let dir = home.join("models/afm-dd");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("dd_config.json"), "{}").unwrap();
    fs::write(dir.join("adapter_config.json"), "{}").unwrap();
    dir
}

const CHOICE_RECORD: &str = r#"{
  "id": "1",
  "task": "choice",
  "instructions": "pick",
  "options": [{"name": "a", "description": "alpha"}, {"name": "b", "description": "beta"}],
  "state": "hello"
}"#;

#[test]
fn version_flag_and_subcommand() {
    let home = home();
    engine(&home)
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("aria-engine "));

    engine(&home)
        .arg("-v")
        .assert()
        .success()
        .stdout(predicate::str::contains("aria-engine "));

    engine(&home)
        .arg("version")
        .assert()
        .success()
        .stdout(predicate::str::contains("aria-engine "));
}

#[test]
fn no_subcommand_prints_help_and_fails() {
    let home = home();
    engine(&home)
        .assert()
        .failure()
        .stderr(predicate::str::contains("Usage"));
}

#[test]
fn check_and_clean_report_cache_state() {
    let home = home();
    let ckpt = write_encoder_checkpoint(home.path());

    engine(&home)
        .arg("check")
        .assert()
        .success()
        .stdout(predicate::str::contains("afm-de"))
        .stdout(predicate::str::contains("encoder_like=true"));

    engine(&home)
        .args(["check", "afm-de"])
        .assert()
        .success()
        .stdout(predicate::str::contains("complete=true"));

    engine(&home)
        .args(["check", "absent"])
        .assert()
        .success()
        .stdout(predicate::str::contains("not found"));

    engine(&home)
        .args(["clean", "afm-de"])
        .assert()
        .success()
        .stdout(predicate::str::contains("removed afm-de"));
    assert!(!ckpt.exists());

    engine(&home)
        .args(["clean", "afm-de"])
        .assert()
        .success()
        .stdout(predicate::str::contains("not found: afm-de"));

    // `clean` without a name wipes whatever is cached
    write_encoder_checkpoint(home.path());
    engine(&home)
        .arg("clean")
        .assert()
        .success()
        .stdout(predicate::str::contains("cleared models cache"));
    assert!(!home.path().join("models/afm-de").exists());

    // an empty cache prints the empty marker
    engine(&home)
        .arg("check")
        .assert()
        .success()
        .stdout(predicate::str::contains("(no models)"));
}

#[test]
fn list_warns_and_falls_back_to_local_models() {
    let home = home();
    write_encoder_checkpoint(home.path());

    engine(&home)
        .arg("list")
        .assert()
        .success()
        .stdout(predicate::str::contains("afm-de"))
        .stdout(predicate::str::contains("downloaded"))
        .stderr(predicate::str::contains("hub catalog unavailable"));
}

#[test]
fn setup_status_clear_and_non_interactive_flags() {
    let home = home();

    engine(&home)
        .arg("setup")
        .args(["--site-url", "https://ariacompute.com"])
        .args(["--upgrade-url", "https://github.com/ariacompute"])
        .args(["--compute", "cpu"])
        .write_stdin("\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("wrote"));
    assert!(home.path().join("engine.yml").is_file());

    engine(&home)
        .args(["setup", "--status"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "site_url: https://ariacompute.com",
        ))
        .stdout(predicate::str::contains(
            "upgrade_url: https://github.com/ariacompute",
        ))
        .stdout(predicate::str::contains("compute: cpu"))
        .stdout(predicate::str::contains("hf_token: (not set)"));

    engine(&home)
        .args(["setup", "--clear"])
        .assert()
        .success()
        .stdout(predicate::str::contains("cleared"));
    assert!(!home.path().join("engine.yml").is_file());
}

#[test]
fn setup_rejects_unknown_compute() {
    let home = home();
    engine(&home)
        .args(["setup", "--compute", "gpu"])
        .write_stdin("\n")
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid compute"));
}

#[test]
fn decide_encoder_uses_stdin_logits_and_files() {
    let home = home();
    write_encoder_checkpoint(home.path());

    engine(&home)
        .args(["decide", "--track", "encoder", "--logits", "[0.1, 3.0]"])
        .write_stdin(CHOICE_RECORD)
        .assert()
        .success()
        .stdout(predicate::str::contains("\"choice\": \"b\""));

    let input = home.path().join("record.json");
    fs::write(&input, CHOICE_RECORD).unwrap();
    engine(&home)
        .arg("decide")
        .args([
            "--checkpoint",
            home.path().join("models/afm-de").to_str().unwrap(),
        ])
        .args(["--logits", "[3.0, 0.1]"])
        .arg("--file")
        .arg(&input)
        .assert()
        .success()
        .stdout(predicate::str::contains("\"choice\": \"a\""));
}

#[test]
fn decide_accepts_systemone_request_shape() {
    let home = home();
    write_encoder_checkpoint(home.path());

    engine(&home)
        .args(["decide", "--logits", "[0.1, 3.0]"])
        .write_stdin(
            r#"{"state": "hello", "questions": {"q1": {"type": "choice", "criteria": {"a": "alpha", "b": "beta"}}}}"#,
        )
        .assert()
        .success()
        .stdout(predicate::str::contains("\"choice\": \"b\""));
}

#[test]
fn decide_without_golden_input_requires_real_weights() {
    let home = home();
    write_encoder_checkpoint(home.path());
    engine(&home)
        .arg("decide")
        .write_stdin(CHOICE_RECORD)
        .assert()
        .failure()
        .stderr(predicate::str::contains("encoder weights missing"));

    write_decoder_checkpoint(home.path());
    engine(&home)
        .args(["decide", "--track", "decoder"])
        .write_stdin(CHOICE_RECORD)
        .assert()
        .failure()
        .stderr(predicate::str::contains("error:"));
}

#[test]
fn decide_decoder_uses_semif_out() {
    let home = home();
    let ckpt = write_decoder_checkpoint(home.path());

    engine(&home)
        .args(["decide", "--track", "decoder"])
        .args(["--checkpoint", ckpt.to_str().unwrap()])
        .args([
            "--semif-out",
            r#"{"option_ids": ["a", "b"], "probabilities": [0.1, 0.9]}"#,
        ])
        .write_stdin(CHOICE_RECORD)
        .assert()
        .success()
        .stdout(predicate::str::contains("\"choice\": \"b\""));
}

#[test]
fn decide_reports_usage_errors() {
    let home = home();
    write_encoder_checkpoint(home.path());

    engine(&home)
        .args(["decide", "--track", "bogus"])
        .write_stdin(CHOICE_RECORD)
        .assert()
        .failure()
        .stderr(predicate::str::contains("error:"))
        .stderr(predicate::str::contains("unknown track"));

    // nothing cached under this name
    engine(&home)
        .args(["decide", "--model-name", "afm-missing"])
        .write_stdin(CHOICE_RECORD)
        .assert()
        .failure()
        .stderr(predicate::str::contains("checkpoint not found"));

    // an encoder layout requested as decoder (and vice versa)
    let dec = write_decoder_checkpoint(home.path());
    engine(&home)
        .args(["decide", "--track", "encoder"])
        .args(["--checkpoint", dec.to_str().unwrap()])
        .write_stdin(CHOICE_RECORD)
        .assert()
        .failure()
        .stderr(predicate::str::contains("looks like a decoder checkpoint"));

    let enc = write_encoder_checkpoint(home.path());
    engine(&home)
        .args(["decide", "--track", "decoder"])
        .args(["--checkpoint", enc.to_str().unwrap()])
        .write_stdin(CHOICE_RECORD)
        .assert()
        .failure()
        .stderr(predicate::str::contains("looks like an encoder checkpoint"));

    // malformed payloads surface as errors instead of panics
    engine(&home)
        .args(["decide", "--logits", "[0.1, 3.0]"])
        .write_stdin("{not json")
        .assert()
        .failure()
        .stderr(predicate::str::contains("error:"));

    engine(&home)
        .args(["decide", "--logits", "[0.1, 3.0]"])
        .write_stdin(r#"{"state": "s", "questions": []}"#)
        .assert()
        .failure()
        .stderr(predicate::str::contains("questions must be object"));

    // the raw record path rejects an unknown task
    engine(&home)
        .args(["decide", "--logits", "[1.0]"])
        .write_stdin(
            r#"{"id":"1","task":"bogus","instructions":"p","options":[{"name":"a","description":""}],"state":"s"}"#,
        )
        .assert()
        .failure()
        .stderr(predicate::str::contains("error:"));
}

#[test]
fn download_and_upgrade_fail_fast_when_offline() {
    let home = home();

    engine(&home)
        .args(["download", "afm-de"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("error:"));
    // a failed download never leaves a cache entry behind
    assert!(!home.path().join("models/afm-de").exists());

    fs::write(
        home.path().join("engine.yml"),
        "site_url: https://ariacompute.com\nupgrade_url: https://github.com/ariacompute\ncompute: cpu\n",
    )
    .unwrap();

    engine(&home)
        .arg("upgrade")
        .assert()
        .failure()
        .stderr(predicate::str::contains("error:"));
}

/// Mock the Hugging Face tree + resolve endpoints for `repo` with `files`.
async fn mock_hf_repo(server: &wiremock::MockServer, repo: &str, files: &[&str]) {
    use wiremock::matchers::{method, path as mock_path};
    use wiremock::{Mock, ResponseTemplate};
    let listing: Vec<serde_json::Value> = files
        .iter()
        .map(|f| serde_json::json!({"type": "file", "path": f}))
        .collect();
    Mock::given(method("GET"))
        .and(mock_path(format!("/api/models/{repo}/tree/main")))
        .respond_with(ResponseTemplate::new(200).set_body_json(listing))
        .mount(server)
        .await;
    for f in files {
        Mock::given(method("GET"))
            .and(mock_path(format!("/{repo}/resolve/main/{f}")))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(f.as_bytes().to_vec()))
            .mount(server)
            .await;
    }
}

#[test]
fn download_writes_a_complete_encoder_checkpoint() {
    let home = home();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let server = rt.block_on(wiremock::MockServer::start());
    rt.block_on(mock_hf_repo(
        &server,
        "ariacompute/afm-de",
        &["rl_agent_config.json", "tokenizer.json"],
    ));

    engine(&home)
        .env("ARIA_HF_ENDPOINT", server.uri())
        .args(["download", "afm-de"])
        .assert()
        .success()
        .stderr(predicate::str::contains("2 files"));
    assert!(home
        .path()
        .join("models/afm-de/rl_agent_config.json")
        .is_file());

    engine(&home)
        .args(["check", "afm-de"])
        .assert()
        .success()
        .stdout(predicate::str::contains("complete=true"));
}

#[test]
fn download_pulls_the_decoder_adapter_and_base_weights() {
    let home = home();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let server = rt.block_on(wiremock::MockServer::start());
    rt.block_on(mock_hf_repo(
        &server,
        "ariacompute/afm-dd",
        &[
            "adapter_config.json",
            "adapter_model.safetensors",
            "afm-dd.gguf",
        ],
    ));
    rt.block_on(mock_hf_repo(
        &server,
        "openbmb/MiniCPM5-2B",
        &["config.json", "model.safetensors"],
    ));

    engine(&home)
        .env("ARIA_HF_ENDPOINT", server.uri())
        .args(["download", "afm-dd"])
        .assert()
        .success();

    let ckpt = home.path().join("models/afm-dd");
    assert!(ckpt.join("adapter_config.json").is_file());
    // GGUF is dropped, the candle base weights land under base/
    assert!(!ckpt.join("afm-dd.gguf").exists());
    assert!(ckpt.join("base/model.safetensors").is_file());

    engine(&home)
        .args(["check", "afm-dd"])
        .assert()
        .success()
        .stdout(predicate::str::contains("complete=true"));
}

#[test]
fn list_reads_the_modelscope_catalog_for_cn_sites() {
    use wiremock::matchers::{method, path as mock_path};
    use wiremock::{Mock, ResponseTemplate};

    let home = home();
    fs::write(
        home.path().join("engine.yml"),
        "site_url: https://ariacompute.cn\nupgrade_url: https://gitee.com/ariacompute\ncompute: cpu\n",
    )
    .unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let server = rt.block_on(wiremock::MockServer::start());
    rt.block_on(async {
        Mock::given(method("GET"))
            .and(mock_path("/openapi/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {"models": [{"id": "afm-de"}], "total_count": 1}
            })))
            .mount(&server)
            .await;
    });

    engine(&home)
        .env("ARIA_MS_ENDPOINT", server.uri())
        .env("ARIA_MS_ENDPOINT_ALT", server.uri())
        .arg("list")
        .assert()
        .success()
        .stdout(predicate::str::contains("afm-de"))
        .stdout(predicate::str::contains("not downloaded"));
}

fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

/// Raw `GET http://127.0.0.1:{port}{path}` → status code (None when not ready).
fn http_status(port: u16, path: &str) -> Option<u16> {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
    )
    .ok()?;
    let mut buf = String::new();
    stream.read_to_string(&mut buf).ok()?;
    let line = buf.lines().next()?;
    line.split_whitespace().nth(1)?.parse().ok()
}

// `kill -INT` is the only portable way to exercise the Ctrl-C path from a test;
// Windows has no console-signal plumbing here, and `Child::kill` terminates hard.
#[cfg_attr(
    windows,
    ignore = "no portable Ctrl-C delivery for a spawned console process"
)]
#[test]
fn serve_binds_answers_health_and_exits_on_sigint() {
    let home = home();
    write_encoder_checkpoint(home.path());
    let port = free_port();

    let mut child = std::process::Command::cargo_bin("aria-engine")
        .unwrap()
        .env("ARIA_COMPUTE_HOME", home.path())
        .args(["serve", "--track", "encoder"])
        .args(["--bind", &format!("127.0.0.1:{port}")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();

    // A successful HTTP round-trip proves the server loop (and therefore the
    // Ctrl-C handler) is live — a bare TCP connect can win the bind race.
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut health = None;
    while Instant::now() < deadline {
        health = http_status(port, "/health");
        if health == Some(200) {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(health, Some(200), "serve never answered /health on :{port}");

    let status = std::process::Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .unwrap();
    assert!(status.success(), "kill -INT failed");

    let deadline = Instant::now() + Duration::from_secs(20);
    let exit = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        assert!(Instant::now() < deadline, "serve ignored SIGINT");
        std::thread::sleep(Duration::from_millis(20));
    };
    assert!(exit.success(), "serve did not exit cleanly: {exit}");
}

#[test]
fn serve_rejects_bad_track_and_compute() {
    let home = home();
    write_encoder_checkpoint(home.path());

    engine(&home)
        .args(["serve", "--track", "bogus"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown track"));

    engine(&home)
        .args(["serve", "--compute", "gpu"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid compute"));
}

#[test]
fn help_lists_every_subcommand() {
    let home = home();
    engine(&home)
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("AFM typed-decision engine CLI"))
        .stdout(predicate::str::contains("setup"))
        .stdout(predicate::str::contains("download"))
        .stdout(predicate::str::contains("list"))
        .stdout(predicate::str::contains("check"))
        .stdout(predicate::str::contains("clean"))
        .stdout(predicate::str::contains("upgrade"))
        .stdout(predicate::str::contains("serve"))
        .stdout(predicate::str::contains("decide"))
        .stdout(predicate::str::contains("version"));
}

#[test]
fn stdin_is_consumed_for_interactive_setup() {
    let home = home();
    let mut child = std::process::Command::cargo_bin("aria-engine")
        .unwrap()
        .env("ARIA_COMPUTE_HOME", home.path())
        .arg("setup")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    {
        let stdin = child.stdin.as_mut().unwrap();
        // site_url, upgrade_url, compute, token
        stdin
            .write_all(b"https://ariacompute.cn\n\n\nms-token\n")
            .unwrap();
    }
    let status = child.wait().unwrap();
    assert!(status.success());
    let cfg = fs::read_to_string(home.path().join("engine.yml")).unwrap();
    assert!(cfg.contains("ariacompute.cn"), "{cfg}");
    assert!(cfg.contains("gitee.com"), "{cfg}");
    assert!(cfg.contains("ms-token"), "{cfg}");
}
