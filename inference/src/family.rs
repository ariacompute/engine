use aria_kernel::EngineError;
use std::path::Path;

/// Delivery phase for a registered family (requirements §1.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FamilyPhase {
    A,
    B,
    C,
}

/// Architecture class for graph / loader hooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchClass {
    /// Dense decoder-only LLM (Gemma / Qwen / LFM / …).
    TextDense,
    /// MoE text (LFM2-8B-A1B / Inkling); Session top-k router + expert FFN.
    TextMoE,
    /// Vision-language.
    VL,
    /// Vision-language-action.
    VLA,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FamilyEntry {
    pub path: &'static str,
    pub base_model: &'static str,
    pub phase: FamilyPhase,
    pub arch: ArchClass,
}

/// Full registry mirroring model/requirements.md §1.1.
pub const FAMILY_REGISTRY: &[FamilyEntry] = &[
    FamilyEntry {
        path: "qwen/qwen3-0.6b",
        base_model: "Qwen/Qwen3-0.6B",
        phase: FamilyPhase::B,
        arch: ArchClass::TextDense,
    },
    FamilyEntry {
        path: "qwen/qwen3-1.7b",
        base_model: "Qwen/Qwen3-1.7B",
        phase: FamilyPhase::B,
        arch: ArchClass::TextDense,
    },
    FamilyEntry {
        path: "qwen/qwen3.5-0.8b",
        base_model: "Qwen/Qwen3.5-0.8B",
        phase: FamilyPhase::B,
        arch: ArchClass::TextDense,
    },
    FamilyEntry {
        path: "qwen/qwen3.5-2b",
        base_model: "Qwen/Qwen3.5-2B",
        phase: FamilyPhase::B,
        arch: ArchClass::TextDense,
    },
    FamilyEntry {
        path: "gemma/gemma-3-270m-it",
        base_model: "google/gemma-3-270m-it",
        phase: FamilyPhase::B,
        arch: ArchClass::TextDense,
    },
    FamilyEntry {
        path: "gemma/gemma-3-1b-it",
        base_model: "google/gemma-3-1b-it",
        phase: FamilyPhase::B,
        arch: ArchClass::TextDense,
    },
    FamilyEntry {
        path: "gemma/gemma-3n-e2b-it",
        base_model: "google/gemma-3n-E2B-it",
        phase: FamilyPhase::C,
        arch: ArchClass::VL,
    },
    FamilyEntry {
        path: "gemma/gemma-3n-e4b-it",
        base_model: "google/gemma-3n-E4B-it",
        phase: FamilyPhase::C,
        arch: ArchClass::VL,
    },
    FamilyEntry {
        path: "gemma/gemma-4-e2b-it",
        base_model: "google/gemma-4-E2B-it",
        phase: FamilyPhase::A,
        arch: ArchClass::VL, // full VL in stage C; tiny text path available from A
    },
    FamilyEntry {
        path: "gemma/gemma-4-e4b-it",
        base_model: "google/gemma-4-E4B-it",
        phase: FamilyPhase::C,
        arch: ArchClass::VL,
    },
    FamilyEntry {
        path: "lfm/lfm2-350m",
        base_model: "LiquidAI/LFM2-350M",
        phase: FamilyPhase::B,
        arch: ArchClass::TextDense,
    },
    FamilyEntry {
        path: "lfm/lfm2-700m",
        base_model: "LiquidAI/LFM2-700M",
        phase: FamilyPhase::B,
        arch: ArchClass::TextDense,
    },
    FamilyEntry {
        path: "lfm/lfm2-1.2b",
        base_model: "LiquidAI/LFM2-1.2B",
        phase: FamilyPhase::B,
        arch: ArchClass::TextDense,
    },
    FamilyEntry {
        path: "lfm/lfm2-2.6b",
        base_model: "LiquidAI/LFM2-2.6B",
        phase: FamilyPhase::B,
        arch: ArchClass::TextDense,
    },
    FamilyEntry {
        path: "lfm/lfm2-8b-a1b",
        base_model: "LiquidAI/LFM2-8B-A1B",
        phase: FamilyPhase::B,
        arch: ArchClass::TextMoE,
    },
    FamilyEntry {
        path: "lfm/lfm2-vl-450m",
        base_model: "LiquidAI/LFM2-VL-450M",
        phase: FamilyPhase::C,
        arch: ArchClass::VL,
    },
    FamilyEntry {
        path: "lfm/lfm2.5-350m",
        base_model: "LiquidAI/LFM2.5-350M",
        phase: FamilyPhase::B,
        arch: ArchClass::TextDense,
    },
    FamilyEntry {
        path: "lfm/lfm2.5-1.2b-instruct",
        base_model: "LiquidAI/LFM2.5-1.2B-Instruct",
        phase: FamilyPhase::B,
        arch: ArchClass::TextDense,
    },
    FamilyEntry {
        path: "lfm/lfm2.5-1.2b-thinking",
        base_model: "LiquidAI/LFM2.5-1.2B-Thinking",
        phase: FamilyPhase::B,
        arch: ArchClass::TextDense,
    },
    FamilyEntry {
        path: "lfm/lfm2.5-2.6b",
        base_model: "LiquidAI/LFM2.5-2.6B",
        phase: FamilyPhase::B,
        arch: ArchClass::TextDense,
    },
    FamilyEntry {
        path: "lfm/lfm2.5-vl-1.6b",
        base_model: "LiquidAI/LFM2.5-VL-1.6B",
        phase: FamilyPhase::C,
        arch: ArchClass::VL,
    },
    FamilyEntry {
        path: "nanbeige/nanbeige4.2-3b",
        base_model: "Nanbeige/Nanbeige4.2-3B",
        phase: FamilyPhase::B,
        arch: ArchClass::TextDense,
    },
    FamilyEntry {
        path: "bonsai/bonsai-27b",
        base_model: "prism-ml/Bonsai-27B-unpacked",
        phase: FamilyPhase::B,
        arch: ArchClass::TextDense,
    },
    FamilyEntry {
        path: "inkling/inkling-small",
        base_model: "thinkingmachines/Inkling-Small",
        phase: FamilyPhase::B,
        arch: ArchClass::TextMoE,
    },
    FamilyEntry {
        path: "openvla/openvla-7b",
        base_model: "openvla/openvla-7b",
        phase: FamilyPhase::C,
        arch: ArchClass::VLA,
    },
    FamilyEntry {
        path: "openpi/openpi-pi0-3b",
        base_model: "lerobot/pi0_base",
        phase: FamilyPhase::C,
        arch: ArchClass::VLA,
    },
    FamilyEntry {
        path: "openpi/openpi-pi0.5-3b",
        base_model: "lerobot/pi05_base",
        phase: FamilyPhase::C,
        arch: ArchClass::VLA,
    },
    FamilyEntry {
        path: "lingbot/lingbot-vla-v2-6b",
        base_model: "robbyant/lingbot-vla-v2-6b",
        phase: FamilyPhase::C,
        arch: ArchClass::VLA,
    },
];

/// Runtime family handle used by Session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Family {
    pub path: &'static str,
    pub arch: ArchClass,
    pub phase: FamilyPhase,
}

impl Family {
    pub fn path(self) -> &'static str {
        self.path
    }

    pub fn uses_text_decoder(self) -> bool {
        matches!(
            self.arch,
            ArchClass::TextDense | ArchClass::TextMoE | ArchClass::VL | ArchClass::VLA
        )
    }

    pub fn is_moe(self) -> bool {
        self.arch == ArchClass::TextMoE
    }
}

pub fn lookup_family(path: &str) -> Result<&'static FamilyEntry, EngineError> {
    FAMILY_REGISTRY
        .iter()
        .find(|e| e.path == path)
        .ok_or_else(|| EngineError::UnsupportedFamily(path.to_string()))
}

/// Map a bundle directory or `slug_q4` name onto a registry path.
pub fn infer_family_path(hint: &str) -> Option<&'static str> {
    let name = Path::new(hint)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(hint);
    let slug = strip_quant_suffix(name);
    if slug.is_empty() {
        return None;
    }
    if let Ok(e) = lookup_family(slug) {
        return Some(e.path);
    }
    let mut best: Option<&'static str> = None;
    let mut best_len = 0usize;
    for e in FAMILY_REGISTRY {
        let tail = e.path.rsplit('/').next().unwrap_or(e.path);
        if slug == tail && tail.len() > best_len {
            best = Some(e.path);
            best_len = tail.len();
        }
    }
    best
}

fn strip_quant_suffix(name: &str) -> &str {
    let mut s = name;
    if let Some(stripped) = s.strip_suffix("_tiny") {
        s = stripped;
    }
    if let Some(idx) = s.rfind("_q") {
        let suffix = &s[idx + 2..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit() || c == '.') {
            s = &s[..idx];
        }
    }
    s
}

/// Qwen3 (not 3.5 dual-RoPE) uses θ=1e6. Bundles that omit `rope_theta` hit serde
/// default 10000 and produce garbage completions.
pub fn effective_rope_theta(family_path: &str, configured: f32) -> f32 {
    let p = family_path.to_ascii_lowercase();
    if p.contains("qwen") && !p.contains("qwen3.5") && (configured - 10_000.0).abs() < 0.5 {
        1_000_000.0
    } else {
        configured
    }
}

pub fn family_phase(path: &str) -> Result<FamilyPhase, EngineError> {
    Ok(lookup_family(path)?.phase)
}

/// Graph hook id for an architecture class (stage B/C dispatch).
pub fn graph_hook(arch: ArchClass) -> &'static str {
    match arch {
        ArchClass::TextDense => "text_dense_decoder",
        ArchClass::TextMoE => "text_moe_decoder",
        ArchClass::VL => "vl_text_plus_vision",
        ArchClass::VLA => "vla_text_vision_action",
    }
}

/// Stage A only: gemma-4-e2b-it.
pub fn require_stage_a(path: &str) -> Result<Family, EngineError> {
    let e = lookup_family(path)?;
    if e.phase != FamilyPhase::A {
        return Err(EngineError::UnsupportedFamily(format!(
            "{} is phase {:?}; stage A only runs gemma/gemma-4-e2b-it",
            path, e.phase
        )));
    }
    Ok(Family {
        path: e.path,
        arch: e.arch,
        phase: e.phase,
    })
}

/// Stage B: text / MoE families (+ stage-A golden path).
pub fn require_stage_b(path: &str) -> Result<Family, EngineError> {
    let e = lookup_family(path)?;
    let ok = match (e.phase, e.arch) {
        (FamilyPhase::A, _) => e.path == "gemma/gemma-4-e2b-it",
        (FamilyPhase::B, ArchClass::TextDense | ArchClass::TextMoE) => true,
        _ => false,
    };
    if !ok {
        return Err(EngineError::UnsupportedFamily(format!(
            "{} (phase {:?}, arch {:?}) is not a stage-B text/MoE family",
            path, e.phase, e.arch
        )));
    }
    Ok(Family {
        path: e.path,
        arch: e.arch,
        phase: e.phase,
    })
}

/// Any registered family that can run the shared text decoder (+ stage C extras).
pub fn require_runnable(path: &str) -> Result<Family, EngineError> {
    let e = lookup_family(path)?;
    Ok(Family {
        path: e.path,
        arch: e.arch,
        phase: e.phase,
    })
}

/// Representative path per arch class for tiny E2E tests.
pub fn arch_class_representatives() -> &'static [(&'static str, ArchClass)] {
    &[
        ("gemma/gemma-3-270m-it", ArchClass::TextDense),
        ("qwen/qwen3.5-2b", ArchClass::TextDense),
        ("lfm/lfm2-350m", ArchClass::TextDense),
        ("lfm/lfm2-8b-a1b", ArchClass::TextMoE),
        ("nanbeige/nanbeige4.2-3b", ArchClass::TextDense),
        ("bonsai/bonsai-27b", ArchClass::TextDense),
        ("inkling/inkling-small", ArchClass::TextMoE),
        ("lfm/lfm2-vl-450m", ArchClass::VL),
        ("openvla/openvla-7b", ArchClass::VLA),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirror model `tests/test_families.py` EXPECTED path → base_model lock table.
    const EXPECTED_BASE_MODELS: &[(&str, &str)] = &[
        ("qwen/qwen3-0.6b", "Qwen/Qwen3-0.6B"),
        ("qwen/qwen3-1.7b", "Qwen/Qwen3-1.7B"),
        ("qwen/qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B"),
        ("qwen/qwen3.5-2b", "Qwen/Qwen3.5-2B"),
        ("gemma/gemma-3-270m-it", "google/gemma-3-270m-it"),
        ("gemma/gemma-3-1b-it", "google/gemma-3-1b-it"),
        ("gemma/gemma-3n-e2b-it", "google/gemma-3n-E2B-it"),
        ("gemma/gemma-3n-e4b-it", "google/gemma-3n-E4B-it"),
        ("gemma/gemma-4-e2b-it", "google/gemma-4-E2B-it"),
        ("gemma/gemma-4-e4b-it", "google/gemma-4-E4B-it"),
        ("lfm/lfm2-350m", "LiquidAI/LFM2-350M"),
        ("lfm/lfm2-700m", "LiquidAI/LFM2-700M"),
        ("lfm/lfm2-1.2b", "LiquidAI/LFM2-1.2B"),
        ("lfm/lfm2-2.6b", "LiquidAI/LFM2-2.6B"),
        ("lfm/lfm2-8b-a1b", "LiquidAI/LFM2-8B-A1B"),
        ("lfm/lfm2-vl-450m", "LiquidAI/LFM2-VL-450M"),
        ("lfm/lfm2.5-350m", "LiquidAI/LFM2.5-350M"),
        ("lfm/lfm2.5-1.2b-instruct", "LiquidAI/LFM2.5-1.2B-Instruct"),
        ("lfm/lfm2.5-1.2b-thinking", "LiquidAI/LFM2.5-1.2B-Thinking"),
        ("lfm/lfm2.5-2.6b", "LiquidAI/LFM2.5-2.6B"),
        ("lfm/lfm2.5-vl-1.6b", "LiquidAI/LFM2.5-VL-1.6B"),
        ("nanbeige/nanbeige4.2-3b", "Nanbeige/Nanbeige4.2-3B"),
        ("bonsai/bonsai-27b", "prism-ml/Bonsai-27B-unpacked"),
        ("inkling/inkling-small", "thinkingmachines/Inkling-Small"),
        ("openvla/openvla-7b", "openvla/openvla-7b"),
        ("openpi/openpi-pi0-3b", "lerobot/pi0_base"),
        ("openpi/openpi-pi0.5-3b", "lerobot/pi05_base"),
        ("lingbot/lingbot-vla-v2-6b", "robbyant/lingbot-vla-v2-6b"),
    ];

    #[test]
    fn registry_matches_model_expected() {
        assert_eq!(FAMILY_REGISTRY.len(), EXPECTED_BASE_MODELS.len());
        for (path, base) in EXPECTED_BASE_MODELS {
            let e = lookup_family(path).unwrap_or_else(|_| panic!("missing {path}"));
            assert_eq!(e.base_model, *base, "{path}");
        }
        // Every registry row appears in EXPECTED.
        for e in FAMILY_REGISTRY {
            assert!(
                EXPECTED_BASE_MODELS.iter().any(|(p, b)| *p == e.path && *b == e.base_model),
                "unexpected registry entry {}",
                e.path
            );
        }
    }

    #[test]
    fn registry_phase_gates() {
        assert!(require_stage_b("qwen/qwen3.5-2b").is_ok());
        assert!(require_stage_b("lfm/lfm2-8b-a1b").is_ok());
        assert!(matches!(
            require_stage_b("openvla/openvla-7b"),
            Err(EngineError::UnsupportedFamily(_))
        ));
        assert!(require_runnable("openvla/openvla-7b").is_ok());
        assert_eq!(graph_hook(ArchClass::TextMoE), "text_moe_decoder");
        assert_eq!(
            require_stage_a("gemma/gemma-4-e2b-it").unwrap().path(),
            "gemma/gemma-4-e2b-it"
        );
        assert_eq!(lookup_family("lfm/lfm2-8b-a1b").unwrap().arch, ArchClass::TextMoE);
        assert_eq!(lookup_family("lfm/lfm2-vl-450m").unwrap().arch, ArchClass::VL);
        assert_eq!(lookup_family("openvla/openvla-7b").unwrap().arch, ArchClass::VLA);
    }

    #[test]
    fn infer_family_from_bundle_dirname() {
        assert_eq!(
            infer_family_path("qwen3-0.6b_q4"),
            Some("qwen/qwen3-0.6b")
        );
        assert_eq!(
            infer_family_path("/home/ubuntu/.ariacompute/models/qwen3-0.6b_q4"),
            Some("qwen/qwen3-0.6b")
        );
        assert_eq!(
            infer_family_path("gemma-4-e2b-it_q8"),
            Some("gemma/gemma-4-e2b-it")
        );
        assert!(infer_family_path("totally-unknown_q4").is_none());
    }

    #[test]
    fn qwen3_rope_theta_not_llama_default() {
        assert!((effective_rope_theta("qwen/qwen3-0.6b", 10_000.0) - 1_000_000.0).abs() < 1.0);
        assert!((effective_rope_theta("qwen/qwen3-0.6b", 1_000_000.0) - 1_000_000.0).abs() < 1.0);
        assert!((effective_rope_theta("gemma/gemma-4-e2b-it", 10_000.0) - 10_000.0).abs() < 1.0);
        assert!((effective_rope_theta("qwen/qwen3.5-2b", 10_000.0) - 10_000.0).abs() < 1.0);
    }
}
