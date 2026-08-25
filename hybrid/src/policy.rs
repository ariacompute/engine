//! Effective routing axes: execution × mode × semantic.
//!
//! `compute` is orthogonal (local GEMM only) and is not represented here.
//! See requirements.md §3.4.2.

use crate::route::{ExecutionMode, ParetoMode};

/// How hybrid Chat tasks (knowledge / intro / long chat) are decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatPolicy {
    /// Cost: consult the semantic layer.
    NeedSemantic,
    /// Balance / Intelligence: prefer cloud without the 800ms classifier.
    PreferCloud,
}

/// Chat policy for hybrid execution (ignored when `execution` is not Hybrid).
pub fn chat_policy(mode: ParetoMode) -> ChatPolicy {
    match mode {
        ParetoMode::Cost => ChatPolicy::NeedSemantic,
        ParetoMode::Balance | ParetoMode::Intelligence => ChatPolicy::PreferCloud,
    }
}

/// Snapshot of switches that actually affect routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectiveRouting {
    pub execution: ExecutionMode,
    pub mode: ParetoMode,
    /// Config / CLI semantic switch (may be unused).
    pub semantic_enabled: bool,
    pub cloud_available: bool,
}

impl EffectiveRouting {
    pub fn new(
        execution: ExecutionMode,
        mode: ParetoMode,
        semantic_enabled: bool,
        cloud_available: bool,
    ) -> Self {
        Self {
            execution,
            mode,
            semantic_enabled,
            cloud_available,
        }
    }

    /// Semantic slow path can actually be consulted.
    pub fn semantic_applicable(&self) -> bool {
        self.execution == ExecutionMode::Hybrid && self.cloud_available && self.semantic_enabled
    }

    pub fn chat_policy(&self) -> ChatPolicy {
        chat_policy(self.mode)
    }

    /// Serve / log label: `on` | `off` | `n/a`.
    pub fn semantic_label(&self) -> &'static str {
        if self.execution != ExecutionMode::Hybrid || !self.cloud_available {
            "n/a"
        } else if self.semantic_enabled {
            "on"
        } else {
            "off"
        }
    }

    /// Serve / log label; `unused` when execution is not hybrid.
    pub fn mode_label(&self) -> &'static str {
        if self.execution != ExecutionMode::Hybrid {
            "unused"
        } else {
            self.mode.as_str()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route::{RouteAction, RouteLayer, RouteSignal, Router};
    use crate::rules::RuleEngine;

    #[test]
    fn chat_policy_follows_mode() {
        assert_eq!(chat_policy(ParetoMode::Cost), ChatPolicy::NeedSemantic);
        assert_eq!(chat_policy(ParetoMode::Balance), ChatPolicy::PreferCloud);
        assert_eq!(
            chat_policy(ParetoMode::Intelligence),
            ChatPolicy::PreferCloud
        );
    }

    #[test]
    fn semantic_applicable_requires_hybrid_cloud_and_switch() {
        let on = EffectiveRouting::new(ExecutionMode::Hybrid, ParetoMode::Balance, true, true);
        assert!(on.semantic_applicable());
        assert_eq!(on.semantic_label(), "on");
        assert_eq!(on.mode_label(), "balance");

        let no_cloud =
            EffectiveRouting::new(ExecutionMode::Hybrid, ParetoMode::Balance, true, false);
        assert!(!no_cloud.semantic_applicable());
        assert_eq!(no_cloud.semantic_label(), "n/a");

        let device = EffectiveRouting::new(ExecutionMode::Device, ParetoMode::Balance, true, true);
        assert!(!device.semantic_applicable());
        assert_eq!(device.semantic_label(), "n/a");
        assert_eq!(device.mode_label(), "unused");

        let off = EffectiveRouting::new(ExecutionMode::Hybrid, ParetoMode::Balance, false, true);
        assert!(!off.semantic_applicable());
        assert_eq!(off.semantic_label(), "off");
    }

    #[test]
    fn composition_matrix_hello_chat_agent() {
        let hello = "hi";
        let intro = "Introduce Rust/C/C++ languages";
        let agent = "please refactor this module";
        let sig = RouteSignal::from_confidence(0.95);

        let device = RuleEngine::new(ExecutionMode::Device, ParetoMode::Balance, 2);
        for p in [hello, intro, agent] {
            let d = device.evaluate(&sig, p);
            assert_eq!(d.action, Some(RouteAction::Local), "{p}");
            assert!(!d.need_semantic, "{p}");
            assert_eq!(d.reason, "rule:execution_device");
        }

        let cloud = RuleEngine::new(ExecutionMode::Cloud, ParetoMode::Cost, 2);
        for p in [hello, intro, agent] {
            let d = cloud.evaluate(&sig, p);
            assert_eq!(d.action, Some(RouteAction::CloudHandoff), "{p}");
            assert!(!d.need_semantic, "{p}");
            assert_eq!(d.reason, "rule:execution_cloud");
        }

        let bal = RuleEngine::new(ExecutionMode::Hybrid, ParetoMode::Balance, 2);
        let d = bal.evaluate(&sig, hello);
        assert_eq!(d.action, Some(RouteAction::Local));
        assert_eq!(d.reason, "rule:inline_local");
        let d = bal.evaluate(&sig, intro);
        assert_eq!(d.action, Some(RouteAction::CloudHandoff));
        assert!(!d.need_semantic);
        assert_eq!(d.reason, "rule:chat_prefer_cloud");
        let d = bal.evaluate(&sig, agent);
        assert_eq!(d.action, None);
        assert!(d.need_semantic);
        assert_eq!(d.reason, "rule:agent_task");

        let cost = RuleEngine::new(ExecutionMode::Hybrid, ParetoMode::Cost, 2);
        let d = cost.evaluate(&sig, hello);
        assert_eq!(d.reason, "rule:inline_local");
        let d = cost.evaluate(&sig, intro);
        assert_eq!(d.action, None);
        assert!(d.need_semantic);
        assert_eq!(d.reason, "rule:chat_task");
        let d = cost.evaluate(&sig, agent);
        assert_eq!(d.reason, "rule:agent_task");

        let intel = RuleEngine::new(ExecutionMode::Hybrid, ParetoMode::Intelligence, 2);
        let d = intel.evaluate(&sig, intro);
        assert_eq!(d.action, Some(RouteAction::CloudHandoff));
        assert_eq!(d.reason, "rule:chat_prefer_cloud");

        let mut no_cloud = sig.clone();
        no_cloud.cloud_available = false;
        let d = bal.evaluate(&no_cloud, intro);
        assert_eq!(d.action, Some(RouteAction::Local));
        assert_eq!(d.reason, "rule:chat_prefer_cloud_but_unavailable");
    }

    #[test]
    fn p0_route_ignores_prompt_and_chat_policy() {
        let r = Router::new().unwrap();
        let sig = RouteSignal::from_confidence(0.95);
        let d = r.route(&sig);
        assert_eq!(d.action, RouteAction::Local);
        assert_eq!(d.layer, RouteLayer::Rules);
        assert!(!d.semantic_consulted);
        let rules = RuleEngine::new(ExecutionMode::Hybrid, ParetoMode::Balance, 2);
        let rd = rules.evaluate(&sig, "Introduce Rust/C/C++ languages");
        assert_eq!(rd.action, Some(RouteAction::CloudHandoff));
        assert_eq!(rd.reason, "rule:chat_prefer_cloud");
    }
}
