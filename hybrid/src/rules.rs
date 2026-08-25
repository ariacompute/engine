//! Rule routing layer (fast path, zero LLM cost).
//!
//! Deterministic ordered rules over `RequestKind` + `RouteSignal` + Pareto
//! thresholds. Hard constraints decide immediately with high confidence;
//! requests in the complexity "neighborhood" of the mode cutoff (or agentic
//! cross-domain tasks) yield `action: None` + `need_semantic: true`, handing
//! off to the semantic routing layer. Rule outcomes stay consistent with
//! `Router::project()` for every hard constraint.

use crate::policy::{chat_policy, ChatPolicy};
use crate::route::{ExecutionMode, ParetoMode, RouteAction, RouteSignal};

/// Fraction of the context limit at/above which a prompt counts as long-context.
const LONG_CONTEXT_RATIO: f64 = 0.8;
/// Complexity neighborhood half-width around the mode cutoff (±10%).
const NEIGHBORHOOD: f32 = 0.1;

/// Request kind classified from prompt text + signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestKind {
    /// Very short prompt, no task keywords — low-latency class.
    Inline,
    /// Regular chat.
    Chat,
    /// Agentic / multi-step / cross-domain task.
    Agent,
    /// Prompt near or over the local context budget.
    LongContext,
    /// Local engine cannot serve the modality.
    Media,
}

/// Agentic / cross-domain keywords (EN + ZH).
const AGENT_KEYWORDS: &[&str] = &[
    "refactor",
    "architecture",
    "multi-step",
    "multistep",
    "agent",
    "debug",
    "plan",
    "implement",
    "code review",
    "重构",
    "架构",
    "规划",
    "调试",
    "多步",
    "智能体",
    "跨域",
    "实现",
    "帮我写",
];

/// Short non-greeting prompts that still need the semantic layer.
/// Intent labels stay in the semantic JSON; rules only decide whether to ask.
const SEMANTIC_HINT_KEYWORDS: &[&str] = &[
    // Knowledge / explanatory
    "introduce",
    "explain",
    "describe",
    "compare",
    "overview",
    "tutorial",
    "what is",
    "what are",
    "how to",
    "why ",
    "介绍",
    "解释",
    "讲解",
    "说明",
    "什么是",
    "如何",
    // Reasoning / multi-step
    "prove",
    "derive",
    "theorem",
    "证明",
    "推导",
    // Code generation (agent words live in AGENT_KEYWORDS)
    "write a",
    "写一个",
    "写函数",
    // Math / STEM
    "solve",
    "equation",
    "calculate",
    "解方程",
    "计算",
    // Translate / rewrite / summarize
    "translate",
    "paraphrase",
    "summarize",
    "summary",
    "翻译",
    "总结",
    "摘要",
    "改写",
    // Creative
    "poem",
    "write a story",
    "roleplay",
    "role-play",
    "role play",
    "诗",
    "小说",
    "角色扮演",
    "写一封",
    // Compare / recommend
    "recommend",
    "which is",
    "哪个好",
    "选型",
    // Constrained format
    "as json",
    "in json",
    "json only",
    "schema",
    "markdown table",
    "只输出",
    // Professional advice (privacy_sensitive still hard-local)
    "medical advice",
    "legal advice",
    "诊断",
    "律师",
    "投资建议",
];

fn prompt_has_any(prompt: &str, lower: &str, keys: &[&str]) -> bool {
    keys.iter().any(|k| lower.contains(k) || prompt.contains(k))
}

/// Classify the request from prompt + signals (pure, O(len(prompt))).
pub fn classify(signal: &RouteSignal, prompt: &str) -> RequestKind {
    if signal.modality_unsupported_locally {
        return RequestKind::Media;
    }
    if signal.context_limit != u32::MAX
        && signal.context_limit > 0
        && (signal.context_tokens as f64) >= (signal.context_limit as f64) * LONG_CONTEXT_RATIO
    {
        return RequestKind::LongContext;
    }
    let lower = prompt.to_ascii_lowercase();
    if prompt_has_any(prompt, &lower, AGENT_KEYWORDS) {
        return RequestKind::Agent;
    }
    let chat_hint = prompt.contains('?')
        || prompt.contains('？')
        || prompt_has_any(prompt, &lower, SEMANTIC_HINT_KEYWORDS);
    if chat_hint || prompt.chars().count() >= 80 {
        RequestKind::Chat
    } else {
        RequestKind::Inline
    }
}

/// Aggregated outcome of the rule layer.
#[derive(Debug, Clone, PartialEq)]
pub struct RuleDecision {
    /// `None` = undecided (hand off to semantic layer when `need_semantic`).
    pub action: Option<RouteAction>,
    pub confidence: f32,
    pub reason: String,
    pub need_semantic: bool,
}

impl RuleDecision {
    fn decided(action: RouteAction, confidence: f32, reason: &str) -> Self {
        Self {
            action: Some(action),
            confidence,
            reason: reason.into(),
            need_semantic: false,
        }
    }

    fn undecided(reason: &str) -> Self {
        Self {
            action: None,
            confidence: 0.5,
            reason: reason.into(),
            need_semantic: true,
        }
    }
}

/// Deterministic rule chain (fast path).
#[derive(Debug, Clone, Copy)]
pub struct RuleEngine {
    pub execution: ExecutionMode,
    pub mode: ParetoMode,
    pub upgrade_after_failures: u32,
}

impl RuleEngine {
    pub fn new(execution: ExecutionMode, mode: ParetoMode, upgrade_after_failures: u32) -> Self {
        Self {
            execution,
            mode,
            upgrade_after_failures,
        }
    }

    /// Complexity at/above which the mode prefers cloud (mirrors Router).
    pub fn complexity_cutoff(&self) -> f32 {
        match self.mode {
            ParetoMode::Cost => 0.90,
            ParetoMode::Balance => 0.75,
            ParetoMode::Intelligence => 0.40,
        }
    }

    /// Evaluate the ordered rule chain. First decisive rule wins.
    pub fn evaluate(&self, signal: &RouteSignal, prompt: &str) -> RuleDecision {
        // R1/R2: hard local constraints.
        if self.execution == ExecutionMode::Device {
            return RuleDecision::decided(RouteAction::Local, 1.0, "rule:execution_device");
        }
        if signal.privacy_sensitive {
            return RuleDecision::decided(RouteAction::Local, 1.0, "rule:privacy_sensitive");
        }
        // R3: hard cloud constraint.
        if self.execution == ExecutionMode::Cloud {
            return RuleDecision::decided(RouteAction::CloudHandoff, 1.0, "rule:execution_cloud");
        }

        // R4..R7: signal-driven hard upgrades (cloud unavailable → must local).
        let hard_cloud = |available: bool, reason: &str, conf: f32| {
            if available {
                RuleDecision::decided(RouteAction::CloudHandoff, conf, reason)
            } else {
                RuleDecision::decided(
                    RouteAction::Local,
                    0.9,
                    match reason {
                        "rule:force_cloud" => "rule:force_cloud_but_unavailable",
                        "rule:modality_unsupported" => "rule:modality_but_unavailable",
                        "rule:context_overflow" => "rule:context_overflow_but_unavailable",
                        "rule:chat_prefer_cloud" => "rule:chat_prefer_cloud_but_unavailable",
                        _ => "rule:failures_upgrade_but_unavailable",
                    },
                )
            }
        };
        if signal.force_cloud {
            return hard_cloud(signal.cloud_available, "rule:force_cloud", 0.99);
        }
        if signal.modality_unsupported_locally {
            return hard_cloud(signal.cloud_available, "rule:modality_unsupported", 0.99);
        }
        let context_overflow = signal.context_tokens > signal.context_limit;
        if context_overflow {
            return hard_cloud(signal.cloud_available, "rule:context_overflow", 0.95);
        }
        if signal.consecutive_local_failures >= self.upgrade_after_failures {
            return hard_cloud(signal.cloud_available, "rule:local_failures_upgrade", 0.9);
        }

        // R8/R9: kind-driven uncertainty → semantic layer.
        let kind = classify(signal, prompt);
        if kind == RequestKind::LongContext {
            return RuleDecision::undecided("rule:long_context_near_limit");
        }
        if kind == RequestKind::Agent {
            return RuleDecision::undecided("rule:agent_task");
        }
        if kind == RequestKind::Chat {
            // Chat policy is mode-driven (see §3.4.2): Cost asks the semantic
            // layer; Balance/Intelligence prefer cloud without the 800ms
            // classifier (timeout or action=local would keep a 1B device model).
            match chat_policy(self.mode) {
                ChatPolicy::NeedSemantic => {
                    return RuleDecision::undecided("rule:chat_task");
                }
                ChatPolicy::PreferCloud => {
                    return hard_cloud(signal.cloud_available, "rule:chat_prefer_cloud", 0.75);
                }
            }
        }

        // R10/R11: complexity neighborhood → semantic; clearly above → cloud.
        let cutoff = self.complexity_cutoff();
        let low = cutoff * (1.0 - NEIGHBORHOOD);
        let high = cutoff * (1.0 + NEIGHBORHOOD);
        if signal.complexity >= high {
            return hard_cloud(signal.cloud_available, "rule:high_complexity", 0.8);
        }
        if signal.complexity >= low {
            return RuleDecision::undecided("rule:complexity_near_cutoff");
        }

        // R12: greetings / tiny prompts stay local (semantic skipped).
        RuleDecision::decided(RouteAction::Local, 0.9, "rule:inline_local")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> RuleEngine {
        RuleEngine::new(ExecutionMode::Hybrid, ParetoMode::Balance, 2)
    }

    fn sig() -> RouteSignal {
        RouteSignal::from_confidence(0.95)
    }

    #[test]
    fn hard_constraints_decide_immediately() {
        let e = engine();
        let mut s = sig();
        s.privacy_sensitive = true;
        let d = e.evaluate(&s, "anything");
        assert_eq!(d.action, Some(RouteAction::Local));
        assert!(!d.need_semantic);
        assert_eq!(d.reason, "rule:privacy_sensitive");

        let e = RuleEngine::new(ExecutionMode::Device, ParetoMode::Balance, 2);
        let d = e.evaluate(&sig(), "hi");
        assert_eq!(d.action, Some(RouteAction::Local));
        assert_eq!(d.reason, "rule:execution_device");

        let e = RuleEngine::new(ExecutionMode::Cloud, ParetoMode::Balance, 2);
        let d = e.evaluate(&sig(), "hi");
        assert_eq!(d.action, Some(RouteAction::CloudHandoff));
        assert_eq!(d.reason, "rule:execution_cloud");
    }

    #[test]
    fn hard_upgrades_and_unavailable_fallback() {
        let e = engine();
        let mut s = sig();
        s.force_cloud = true;
        let d = e.evaluate(&s, "hi");
        assert_eq!(d.action, Some(RouteAction::CloudHandoff));
        assert_eq!(d.reason, "rule:force_cloud");

        s.cloud_available = false;
        let d = e.evaluate(&s, "hi");
        assert_eq!(d.action, Some(RouteAction::Local));
        assert_eq!(d.reason, "rule:force_cloud_but_unavailable");

        let mut s = sig();
        s.modality_unsupported_locally = true;
        let d = e.evaluate(&s, "hi");
        assert_eq!(d.action, Some(RouteAction::CloudHandoff));

        let mut s = sig();
        s.context_tokens = 100;
        s.context_limit = 50;
        let d = e.evaluate(&s, "hi");
        assert_eq!(d.action, Some(RouteAction::CloudHandoff));
        assert_eq!(d.reason, "rule:context_overflow");

        let mut s = sig();
        s.consecutive_local_failures = 2;
        let d = e.evaluate(&s, "hi");
        assert_eq!(d.action, Some(RouteAction::CloudHandoff));
        assert_eq!(d.reason, "rule:local_failures_upgrade");
    }

    #[test]
    fn classify_kinds() {
        let mut s = sig();
        assert_eq!(classify(&s, "hi"), RequestKind::Inline);
        assert_eq!(classify(&s, &"x".repeat(200)), RequestKind::Chat);
        assert_eq!(
            classify(&s, "Introduce Rust/C/C++ languages"),
            RequestKind::Chat
        );
        assert_eq!(classify(&s, "什么是 Rust？"), RequestKind::Chat);
        assert_eq!(
            classify(&s, "Prove sqrt(2) is irrational"),
            RequestKind::Chat
        );
        assert_eq!(
            classify(&s, "Write a Rust mutex wrapper"),
            RequestKind::Chat
        );
        assert_eq!(classify(&s, "Solve x^2+1=0"), RequestKind::Chat);
        assert_eq!(classify(&s, "Translate this to French"), RequestKind::Chat);
        assert_eq!(classify(&s, "写一首关于秋天的诗"), RequestKind::Chat);
        assert_eq!(classify(&s, "recommend a laptop"), RequestKind::Chat);
        assert_eq!(classify(&s, "Reply as JSON only"), RequestKind::Chat);
        assert_eq!(
            classify(&s, "legal advice on a contract"),
            RequestKind::Chat
        );
        assert_eq!(
            classify(&s, "implement binary search in Rust"),
            RequestKind::Agent
        );
        assert_eq!(classify(&s, "帮我写一个排序函数"), RequestKind::Agent);
        assert_eq!(
            classify(&s, "please refactor this architecture"),
            RequestKind::Agent
        );
        assert_eq!(classify(&s, "帮我规划一下重构方案"), RequestKind::Agent);
        s.context_tokens = 90;
        s.context_limit = 100;
        assert_eq!(classify(&s, "hi"), RequestKind::LongContext);
        let mut s = sig();
        s.modality_unsupported_locally = true;
        assert_eq!(classify(&s, "hi"), RequestKind::Media);
    }

    #[test]
    fn agent_and_long_context_trigger_semantic() {
        let e = engine();
        let d = e.evaluate(&sig(), "please refactor this module");
        assert_eq!(d.action, None);
        assert!(d.need_semantic);
        assert_eq!(d.reason, "rule:agent_task");

        let d = e.evaluate(&sig(), "Introduce Rust/C/C++ languages");
        assert_eq!(d.action, Some(RouteAction::CloudHandoff));
        assert!(!d.need_semantic);
        assert_eq!(d.reason, "rule:chat_prefer_cloud");

        let mut s = sig();
        s.context_tokens = 90;
        s.context_limit = 100;
        let d = e.evaluate(&s, "hi");
        assert_eq!(d.action, None);
        assert!(d.need_semantic);
        assert_eq!(d.reason, "rule:long_context_near_limit");
    }

    #[test]
    fn complexity_neighborhood_triggers_semantic() {
        let e = engine(); // Balance cutoff 0.75 → neighborhood [0.675, 0.825)
        let mut s = sig();
        s.complexity = 0.7;
        let d = e.evaluate(&s, "hi");
        assert_eq!(d.action, None);
        assert!(d.need_semantic);
        assert_eq!(d.reason, "rule:complexity_near_cutoff");

        // Clearly above neighborhood → confident cloud.
        s.complexity = 0.9;
        let d = e.evaluate(&s, "hi");
        assert_eq!(d.action, Some(RouteAction::CloudHandoff));
        assert_eq!(d.reason, "rule:high_complexity");

        // Above neighborhood but cloud unavailable → local.
        s.cloud_available = false;
        let d = e.evaluate(&s, "hi");
        assert_eq!(d.action, Some(RouteAction::Local));

        // Clearly below neighborhood → local.
        let mut s = sig();
        s.complexity = 0.5;
        let d = e.evaluate(&s, "hi");
        assert_eq!(d.action, Some(RouteAction::Local));
        assert_eq!(d.reason, "rule:inline_local");
    }

    #[test]
    fn mode_cutoffs_shift_neighborhood() {
        let cost = RuleEngine::new(ExecutionMode::Hybrid, ParetoMode::Cost, 2);
        let intel = RuleEngine::new(ExecutionMode::Hybrid, ParetoMode::Intelligence, 2);
        let mut s = sig();
        // 0.89 is inside Cost neighborhood [0.81, 0.99) → semantic;
        // above Intelligence neighborhood high (0.44) → cloud.
        s.complexity = 0.89;
        assert!(cost.evaluate(&s, "hi").need_semantic);
        assert_eq!(
            intel.evaluate(&s, "hi").action,
            Some(RouteAction::CloudHandoff)
        );
        assert!((cost.complexity_cutoff() - 0.90).abs() < 1e-6);
        assert!((intel.complexity_cutoff() - 0.40).abs() < 1e-6);
    }

    #[test]
    fn default_local_by_kind() {
        let e = engine();
        let d = e.evaluate(&sig(), "hello");
        assert_eq!(d.action, Some(RouteAction::Local));
        assert_eq!(d.reason, "rule:inline_local");
        assert!((d.confidence - 0.9).abs() < 1e-6);
        assert!(!d.need_semantic);

        let d = e.evaluate(&sig(), &"plain chat ".repeat(20));
        assert_eq!(d.action, Some(RouteAction::CloudHandoff));
        assert!(!d.need_semantic);
        assert_eq!(d.reason, "rule:chat_prefer_cloud");

        let cost = RuleEngine::new(ExecutionMode::Hybrid, ParetoMode::Cost, 2);
        let d = cost.evaluate(&sig(), "Introduce Rust/C/C++ languages");
        assert_eq!(d.action, None);
        assert!(d.need_semantic);
        assert_eq!(d.reason, "rule:chat_task");
    }

    #[test]
    fn semantic_hint_categories_need_semantic() {
        let e = engine();
        let cases: &[(&str, Option<RouteAction>, &str, bool)] = &[
            (
                "Prove sqrt(2) is irrational",
                Some(RouteAction::CloudHandoff),
                "rule:chat_prefer_cloud",
                false,
            ),
            (
                "Write a Rust mutex wrapper",
                Some(RouteAction::CloudHandoff),
                "rule:chat_prefer_cloud",
                false,
            ),
            (
                "Solve x^2+1=0",
                Some(RouteAction::CloudHandoff),
                "rule:chat_prefer_cloud",
                false,
            ),
            (
                "summarize this paragraph",
                Some(RouteAction::CloudHandoff),
                "rule:chat_prefer_cloud",
                false,
            ),
            (
                "写一首关于秋天的诗",
                Some(RouteAction::CloudHandoff),
                "rule:chat_prefer_cloud",
                false,
            ),
            (
                "recommend a laptop",
                Some(RouteAction::CloudHandoff),
                "rule:chat_prefer_cloud",
                false,
            ),
            (
                "Reply as JSON only",
                Some(RouteAction::CloudHandoff),
                "rule:chat_prefer_cloud",
                false,
            ),
            (
                "legal advice on a contract",
                Some(RouteAction::CloudHandoff),
                "rule:chat_prefer_cloud",
                false,
            ),
            ("implement binary search", None, "rule:agent_task", true),
        ];
        for (prompt, action, reason, need_sem) in cases {
            let d = e.evaluate(&sig(), prompt);
            assert_eq!(d.need_semantic, *need_sem, "prompt {prompt:?}");
            assert_eq!(d.action, *action, "prompt {prompt:?}");
            assert_eq!(d.reason, *reason, "prompt {prompt:?}");
        }

        let mut no_cloud = sig();
        no_cloud.cloud_available = false;
        let d = e.evaluate(&no_cloud, "Introduce Rust/C/C++ languages");
        assert_eq!(d.action, Some(RouteAction::Local));
        assert_eq!(d.reason, "rule:chat_prefer_cloud_but_unavailable");
        for greet in ["hi", "Hello", "ok", "thanks", "谢谢"] {
            let d = e.evaluate(&sig(), greet);
            assert!(
                !d.need_semantic,
                "greeting {greet:?} must stay inline local"
            );
            assert_eq!(d.action, Some(RouteAction::Local));
            assert_eq!(d.reason, "rule:inline_local");
        }

        let mut s = sig();
        s.privacy_sensitive = true;
        let d = e.evaluate(&s, "legal advice on a contract");
        assert!(!d.need_semantic);
        assert_eq!(d.action, Some(RouteAction::Local));
        assert_eq!(d.reason, "rule:privacy_sensitive");
    }
}
