//! `aria-engine setup` — five-field engine.yml (no router).

use ariacompute_core::config::{self, apply_hub_token_input, parse_compute, AriaConfig};
use ariacompute_core::gateway::GatewayPair;
use std::io::{self, BufRead, Write};

/// Input seam for the interactive flow (stdin in production, scripted in tests).
pub trait Prompter {
    fn ask(&mut self, label: &str) -> io::Result<String>;
    fn ask_choice(&mut self, label: &str, allowed: &[&str], default: &str) -> io::Result<String>;
    fn ask_secret(&mut self, label: &str) -> io::Result<String>;
}

/// stdin-backed prompter used by the CLI.
pub struct StdinPrompter;

impl StdinPrompter {
    fn read_line(&self, label: &str) -> io::Result<String> {
        eprint!("{label}");
        io::stderr().flush()?;
        let mut line = String::new();
        io::stdin().lock().read_line(&mut line)?;
        Ok(line.trim().to_string())
    }
}

impl Prompter for StdinPrompter {
    fn ask(&mut self, label: &str) -> io::Result<String> {
        self.read_line(label)
    }

    fn ask_choice(&mut self, label: &str, allowed: &[&str], default: &str) -> io::Result<String> {
        ask_choice(self, label, allowed, default)
    }

    fn ask_secret(&mut self, label: &str) -> io::Result<String> {
        // Masked input on a terminal; piped stdin (CI, tests) has no TTY and
        // falls back to a plain line read.
        let prompt = label.trim_end_matches([' ', ':']);
        dialoguer::Password::new()
            .with_prompt(prompt)
            .allow_empty_password(true)
            .interact()
            .or_else(|_| self.ask(label))
    }
}

fn ask_choice<P: Prompter + ?Sized>(
    prompter: &mut P,
    label: &str,
    allowed: &[&str],
    default: &str,
) -> io::Result<String> {
    let joined = allowed.join("|");
    loop {
        let raw = prompter.ask(&format!("{label} [{joined}] (default: {default}): "))?;
        if raw.is_empty() {
            return Ok(default.to_string());
        }
        let lower = raw.to_ascii_lowercase();
        if allowed.iter().any(|a| *a == lower) {
            return Ok(lower);
        }
        eprintln!("invalid choice: {raw}");
    }
}

fn redact_secret(value: &str) -> String {
    if value.is_empty() {
        "(not set)".into()
    } else if value.len() <= 8 {
        "********".into()
    } else {
        format!("{}…{}", &value[..4], &value[value.len() - 4..])
    }
}

fn prompt_regional_hub_token<P: Prompter + ?Sized>(
    prompter: &mut P,
    pair: GatewayPair,
    existing: &AriaConfig,
) -> io::Result<(String, String)> {
    let cn = pair.is_cn();
    let entered = if cn {
        prompter.ask_secret("modelscope_api_token (ModelScope, optional): ")?
    } else {
        prompter.ask_secret("hf_token (Hugging Face, optional): ")?
    };
    Ok(apply_hub_token_input(existing, cn, &entered))
}

/// Run `aria-engine setup` (status / clear / interactive or flags).
pub fn cmd_setup(
    status: bool,
    clear: bool,
    site_url_flag: Option<String>,
    upgrade_url_flag: Option<String>,
    compute_flag: Option<String>,
) -> anyhow::Result<()> {
    cmd_setup_with(
        &mut StdinPrompter,
        status,
        clear,
        site_url_flag,
        upgrade_url_flag,
        compute_flag,
    )
}

/// `cmd_setup` with an injectable prompter.
pub fn cmd_setup_with<P: Prompter + ?Sized>(
    prompter: &mut P,
    status: bool,
    clear: bool,
    site_url_flag: Option<String>,
    upgrade_url_flag: Option<String>,
    compute_flag: Option<String>,
) -> anyhow::Result<()> {
    if clear {
        config::clear_config()?;
        println!("cleared {}", config::engine_yml_path()?.display());
        return Ok(());
    }
    if status {
        let cfg = config::load_config()?;
        println!(
            "site_url: {}",
            if cfg.site_url.is_empty() {
                "(empty)"
            } else {
                &cfg.site_url
            }
        );
        println!(
            "upgrade_url: {}",
            if cfg.upgrade_url.is_empty() {
                "(empty)"
            } else {
                &cfg.upgrade_url
            }
        );
        println!("compute: {}", cfg.compute);
        println!("hf_token: {}", redact_secret(&cfg.hf_token));
        println!(
            "modelscope_api_token: {}",
            redact_secret(&cfg.modelscope_api_token)
        );
        println!("config: {}", config::engine_yml_path()?.display());
        println!("lib: {}", config::lib_dir()?.display());
        return Ok(());
    }

    let existing = config::load_config().unwrap_or_default();
    let detect = GatewayPair::detect_default();

    let site_url = match site_url_flag {
        Some(s) => s,
        None => {
            let default_shown = if existing.site_url.is_empty() {
                detect.site_url()
            } else {
                existing.site_url.as_str()
            };
            let s = prompter.ask(&format!("site_url (default: {default_shown}): "))?;
            if s.is_empty() {
                if existing.site_url.is_empty() {
                    detect.site_url().to_string()
                } else {
                    existing.site_url.clone()
                }
            } else {
                s
            }
        }
    };

    let pair = GatewayPair::from_url(&site_url);

    let upgrade_url = match upgrade_url_flag {
        Some(s) => s,
        None => {
            let default_shown = if existing.upgrade_url.is_empty() {
                pair.upgrade_url()
            } else {
                existing.upgrade_url.as_str()
            };
            let s = prompter.ask(&format!("upgrade_url (default: {default_shown}): "))?;
            if s.is_empty() {
                if existing.upgrade_url.is_empty() {
                    pair.upgrade_url().to_string()
                } else {
                    existing.upgrade_url.clone()
                }
            } else {
                s
            }
        }
    };

    let compute = match compute_flag {
        Some(s) => parse_compute(&s).map_err(anyhow::Error::msg)?,
        None => {
            let default = if existing.compute.is_empty() {
                "auto"
            } else {
                existing.compute.as_str()
            };
            prompter.ask_choice("compute", &["auto", "cpu", "cuda"], default)?
        }
    };

    let (hf_token, modelscope_api_token) = prompt_regional_hub_token(prompter, pair, &existing)?;

    let cfg = AriaConfig {
        site_url,
        upgrade_url,
        compute,
        hf_token,
        modelscope_api_token,
    };
    config::save_config(&cfg)?;
    println!("wrote {}", config::engine_yml_path()?.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_env::with_home;
    use serial_test::serial;
    use std::collections::VecDeque;

    struct ScriptedPrompter {
        answers: VecDeque<String>,
        choices: VecDeque<String>,
        secrets: VecDeque<String>,
        asked: Vec<String>,
    }

    impl ScriptedPrompter {
        fn new(answers: &[&str], choices: &[&str], secrets: &[&str]) -> Self {
            Self {
                answers: answers.iter().map(|s| s.to_string()).collect(),
                choices: choices.iter().map(|s| s.to_string()).collect(),
                secrets: secrets.iter().map(|s| s.to_string()).collect(),
                asked: Vec::new(),
            }
        }
    }

    impl Prompter for ScriptedPrompter {
        fn ask(&mut self, label: &str) -> io::Result<String> {
            self.asked.push(label.to_string());
            Ok(self
                .answers
                .pop_front()
                .unwrap_or_default()
                .trim()
                .to_string())
        }

        fn ask_choice(
            &mut self,
            label: &str,
            _allowed: &[&str],
            default: &str,
        ) -> io::Result<String> {
            self.asked.push(label.to_string());
            let raw = self.choices.pop_front().unwrap_or_default();
            Ok(if raw.is_empty() {
                default.to_string()
            } else {
                raw
            })
        }

        fn ask_secret(&mut self, label: &str) -> io::Result<String> {
            self.asked.push(label.to_string());
            Ok(self.secrets.pop_front().unwrap_or_default())
        }
    }

    #[test]
    fn redact_secret_hides_short_and_trims_long() {
        assert_eq!(redact_secret(""), "(not set)");
        assert_eq!(redact_secret("short"), "********");
        assert_eq!(redact_secret("12345678"), "********");
        assert_eq!(redact_secret("hf_abcdefghijkl"), "hf_a…ijkl");
    }

    #[test]
    #[serial]
    fn interactive_flow_defaults_to_detected_gateway() {
        with_home(|home| {
            let mut p = ScriptedPrompter::new(&["", ""], &[""], &["ms-secret-token"]);
            // site_url flag → .cn so ModelScope is the regional hub
            cmd_setup_with(
                &mut p,
                false,
                false,
                Some("https://ariacompute.cn".into()),
                None,
                None,
            )
            .unwrap();

            let cfg = config::load_config().unwrap();
            assert_eq!(cfg.site_url, "https://ariacompute.cn");
            assert_eq!(cfg.upgrade_url, "https://gitee.com/ariacompute");
            assert_eq!(cfg.compute, "auto");
            assert_eq!(cfg.modelscope_api_token, "ms-secret-token");
            assert!(cfg.hf_token.is_empty());
            assert!(home.join("engine.yml").is_file());
            assert!(p.asked.iter().any(|l| l.contains("modelscope_api_token")));
        });
    }

    #[test]
    #[serial]
    fn interactive_flow_keeps_existing_values_on_empty_input() {
        with_home(|home| {
            config::save_config(&AriaConfig {
                site_url: "https://ariacompute.com".into(),
                upgrade_url: "https://github.com/ariacompute".into(),
                compute: "cuda".into(),
                hf_token: "old-hf".into(),
                modelscope_api_token: String::new(),
            })
            .unwrap();

            let mut p = ScriptedPrompter::new(&["", ""], &[], &["new-hf-token"]);
            cmd_setup_with(&mut p, false, false, None, None, None).unwrap();

            let cfg = config::load_config().unwrap();
            assert_eq!(cfg.site_url, "https://ariacompute.com");
            assert_eq!(cfg.upgrade_url, "https://github.com/ariacompute");
            assert_eq!(cfg.compute, "cuda");
            assert_eq!(cfg.hf_token, "new-hf-token");
            assert!(home.join("engine.yml").is_file());
        });
    }

    #[test]
    #[serial]
    fn flags_override_prompts() {
        with_home(|_home| {
            let mut p = ScriptedPrompter::new(&[], &[], &[""]);
            cmd_setup_with(
                &mut p,
                false,
                false,
                Some("https://ariacompute.com".into()),
                Some("https://github.com/ariacompute".into()),
                Some("CPU".into()),
            )
            .unwrap();
            let cfg = config::load_config().unwrap();
            assert_eq!(cfg.compute, "cpu");
            assert_eq!(cfg.upgrade_url, "https://github.com/ariacompute");
            // empty secret keeps the previous (empty) token
            assert!(cfg.hf_token.is_empty());
        });
    }

    #[test]
    #[serial]
    fn invalid_compute_flag_is_rejected() {
        with_home(|_home| {
            let mut p = ScriptedPrompter::new(&[], &[], &[]);
            let err = cmd_setup_with(
                &mut p,
                false,
                false,
                Some("https://ariacompute.com".into()),
                None,
                Some("tpu".into()),
            )
            .unwrap_err();
            assert!(err.to_string().contains("invalid compute"));
        });
    }

    #[test]
    #[serial]
    fn status_prints_redacted_config_and_clear_removes_it() {
        with_home(|home| {
            config::save_config(&AriaConfig {
                site_url: "https://ariacompute.com".into(),
                upgrade_url: String::new(),
                compute: "cpu".into(),
                hf_token: "hf_0123456789".into(),
                modelscope_api_token: String::new(),
            })
            .unwrap();

            let mut p = ScriptedPrompter::new(&[], &[], &[]);
            cmd_setup_with(&mut p, true, false, None, None, None).unwrap();
            assert!(home.join("engine.yml").is_file());

            let mut p = ScriptedPrompter::new(&[], &[], &[]);
            cmd_setup_with(&mut p, false, true, None, None, None).unwrap();
            assert!(!home.join("engine.yml").exists());
        });
    }

    #[test]
    fn choice_prompt_retries_until_allowed() {
        let mut p = ScriptedPrompter::new(&["gpu", " CUDA "], &[], &[]);
        let value = ask_choice(&mut p, "compute", &["auto", "cpu", "cuda"], "auto").unwrap();
        assert_eq!(value, "cuda");

        let mut p = ScriptedPrompter::new(&[""], &[], &[]);
        assert_eq!(
            ask_choice(&mut p, "compute", &["auto", "cpu"], "cpu").unwrap(),
            "cpu"
        );
    }

    #[test]
    fn hub_token_prompt_uses_the_regional_field() {
        let existing = AriaConfig {
            hf_token: "keep-hf".into(),
            modelscope_api_token: "keep-ms".into(),
            ..Default::default()
        };
        let mut p = ScriptedPrompter::new(&[], &[], &["cn-token"]);
        let (hf, ms) = prompt_regional_hub_token(&mut p, GatewayPair::CN, &existing).unwrap();
        assert_eq!(hf, "keep-hf");
        assert_eq!(ms, "cn-token");

        let mut p = ScriptedPrompter::new(&[], &[], &["com-token"]);
        let (hf, ms) = prompt_regional_hub_token(&mut p, GatewayPair::INTL, &existing).unwrap();
        assert_eq!(hf, "com-token");
        assert_eq!(ms, "keep-ms");
    }

    #[test]
    #[serial]
    fn status_reports_empty_values_for_a_fresh_home() {
        with_home(|home| {
            let mut p = ScriptedPrompter::new(&[], &[], &[]);
            cmd_setup_with(&mut p, true, false, None, None, None).unwrap();
            // the fresh home has no engine.yml, so nothing is written by --status
            assert!(!home.join("engine.yml").exists());
            assert!(p.asked.is_empty(), "status must not prompt");
        });
    }

    #[test]
    #[serial]
    fn interactive_flow_falls_back_to_detected_defaults() {
        with_home(|home| {
            // every answer empty → locale-detected gateway + default compute
            let mut p = ScriptedPrompter::new(&["", ""], &[""], &[""]);
            cmd_setup_with(&mut p, false, false, None, None, None).unwrap();

            let cfg = config::load_config().unwrap();
            assert!(!cfg.site_url.is_empty());
            assert!(!cfg.upgrade_url.is_empty());
            assert_eq!(cfg.compute, "auto");
            assert!(cfg.hf_token.is_empty() || cfg.modelscope_api_token.is_empty());
            assert!(home.join("engine.yml").is_file());
        });
    }

    #[test]
    #[serial]
    fn interactive_flow_accepts_an_explicit_compute_choice() {
        with_home(|_home| {
            let mut p = ScriptedPrompter::new(&["", ""], &["cuda"], &[""]);
            cmd_setup_with(&mut p, false, false, None, None, None).unwrap();
            assert_eq!(config::load_config().unwrap().compute, "cuda");
        });
    }

    #[test]
    #[serial]
    fn interactive_flow_takes_typed_site_and_upgrade_urls() {
        with_home(|_home| {
            let mut p = ScriptedPrompter::new(
                &["https://ariacompute.com", "https://github.com/custom"],
                &[""],
                &[""],
            );
            cmd_setup_with(&mut p, false, false, None, None, None).unwrap();
            let cfg = config::load_config().unwrap();
            assert_eq!(cfg.site_url, "https://ariacompute.com");
            assert_eq!(cfg.upgrade_url, "https://github.com/custom");
        });
    }

    #[test]
    #[serial]
    fn interactive_flow_defaults_compute_when_the_saved_value_is_blank() {
        with_home(|_home| {
            config::save_config(&AriaConfig {
                site_url: "https://ariacompute.com".into(),
                upgrade_url: "https://github.com/ariacompute".into(),
                compute: String::new(),
                hf_token: String::new(),
                modelscope_api_token: String::new(),
            })
            .unwrap();
            let mut p = ScriptedPrompter::new(&["", ""], &[""], &[""]);
            cmd_setup_with(&mut p, false, false, None, None, None).unwrap();
            assert_eq!(config::load_config().unwrap().compute, "auto");
        });
    }
}
