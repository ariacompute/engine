//! Signal → projection → decision router (P0/P1) + two-layer hybrid entry (P2).

use crate::health::{BackendKind, HealthTracker};
use crate::rules::RuleEngine;
use crate::semantic::{SemanticRouter, SEMANTIC_ACCEPT_THRESHOLD};
use aria_kernel::EngineError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Policy contract version embedded in every decision (replay / Outcome).
pub const POLICY_VERSION: &str = "hybrid-p0p1-v1";

/// Where inference is allowed to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// Signal → projection hybrid (default).
    #[default]
    Hybrid,
    /// Force on-device Local (never hand off).
    Device,
    /// Force cloud handoff (never run local decode for chat routing).
    Cloud,
}

impl ExecutionMode {
    /// Parse execution mode: `hybrid` (default) | `device` | `cloud`.
    pub fn parse(raw: &str) -> Result<Self, EngineError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "hybrid" => Ok(Self::Hybrid),
            "device" => Ok(Self::Device),
            "cloud" => Ok(Self::Cloud),
            other => Err(EngineError::InvalidParam(format!(
                "hybrid_execution must be hybrid|device|cloud, got {other:?}"
            ))),
        }
    }
}

/// Pareto position on the cost–intelligence frontier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ParetoMode {
    /// Prefer on-device; higher complexity cutoff before handoff.
    Cost,
    /// Signal-driven auto routing (default complexity cutoff).
    #[default]
    Balance,
    /// Prefer cloud; lower complexity cutoff before handoff.
    Intelligence,
}

/// Final dispatch target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteAction {
    Local,
    CloudHandoff,
}

/// Thin projection bands (vLLM-SR-inspired, binary hybrid).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionBand {
    MustLocal,
    LocalOk,
    PreferCloud,
}

/// Which routing layer produced the decision (P2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RouteLayer {
    /// Deterministic rule layer (fast path; also the P0/P1 `route()` path).
    #[default]
    Rules,
    /// Semantic layer (slow path; cloud intent classification adopted).
    Semantic,
}

/// Request-level routing inputs (P1 signal surface).
#[derive(Debug, Clone, PartialEq)]
pub struct RouteSignal {
    /// Reserved confidence in `[0, 1]` (API compat; not used for handoff).
    pub confidence: f32,
    /// Task complexity heuristic in `[0, 1]`.
    pub complexity: f32,
    pub context_tokens: u32,
    pub context_limit: u32,
    /// Local engine cannot serve this modality (VL/VLA/tool gap).
    pub modality_unsupported_locally: bool,
    /// Consecutive local failures in this task/session (reask / upgrade).
    pub consecutive_local_failures: u32,
    /// Privacy / authz: never leave device.
    pub privacy_sensitive: bool,
    /// Cloud backend reachable (key/mock present, not timed out).
    pub cloud_available: bool,
    pub session_id: Option<String>,
    /// Test / explicit override (e.g. prompt contains FORCE_CLOUD).
    pub force_cloud: bool,
}

impl RouteSignal {
    pub fn from_confidence(confidence: f32) -> Self {
        Self {
            confidence,
            complexity: 0.0,
            context_tokens: 0,
            context_limit: u32::MAX,
            modality_unsupported_locally: false,
            consecutive_local_failures: 0,
            privacy_sensitive: false,
            cloud_available: true,
            session_id: None,
            force_cloud: false,
        }
    }
}

/// Explained routing decision (P0; P2 adds `#[serde(default)]` layer fields).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteDecision {
    pub action: RouteAction,
    pub reason: String,
    pub policy_version: String,
    pub fallback: RouteAction,
    pub projection: ProjectionBand,
    pub mode: ParetoMode,
    /// P2: which layer produced the decision (`rules` for the P0/P1 path).
    #[serde(default)]
    pub layer: RouteLayer,
    /// P2: decision confidence in `[0, 1]` (P0/P1 path carries signal confidence).
    #[serde(default)]
    pub confidence: f32,
    /// P2: whether the semantic layer was consulted.
    #[serde(default)]
    pub semantic_consulted: bool,
    /// P2: semantic round-trip latency when consulted.
    #[serde(default)]
    pub semantic_latency_ms: Option<u64>,
}

/// Post-execution record for feedback / future learning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteOutcome {
    pub task_id: String,
    pub session_id: Option<String>,
    pub action: RouteAction,
    pub reason: String,
    pub policy_version: String,
    pub mode: ParetoMode,
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub latency_ms: Option<u64>,
    pub cloud_handoff: bool,
    pub user_corrected: Option<bool>,
    pub validation_ok: Option<bool>,
    /// P2: which layer produced the routed decision.
    #[serde(default)]
    pub layer: RouteLayer,
    /// P2: decision confidence.
    #[serde(default)]
    pub confidence: f32,
    /// P2: whether the semantic layer was consulted.
    #[serde(default)]
    pub semantic_consulted: bool,
    /// P2: semantic round-trip latency when consulted.
    #[serde(default)]
    pub semantic_latency_ms: Option<u64>,
}

/// In-process outcome log (P0).
#[derive(Debug, Default, Clone)]
pub struct OutcomeStore {
    inner: Arc<Mutex<Vec<RouteOutcome>>>,
}

impl OutcomeStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&self, outcome: RouteOutcome) {
        self.inner
            .lock()
            .expect("outcome store poisoned")
            .push(outcome);
    }

    pub fn len(&self) -> usize {
        self.inner.lock().expect("outcome store poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn recent(&self, n: usize) -> Vec<RouteOutcome> {
        let guard = self.inner.lock().expect("outcome store poisoned");
        let start = guard.len().saturating_sub(n);
        guard[start..].to_vec()
    }

    pub fn clear(&self) {
        self.inner.lock().expect("outcome store poisoned").clear();
    }
}

#[derive(Debug, Default)]
struct StickinessState {
    /// session_id → last committed action.
    last: HashMap<String, RouteAction>,
}

/// Hybrid router: signals → projection → sticky decision.
#[derive(Debug, Clone)]
pub struct Router {
    pub execution: ExecutionMode,
    pub mode: ParetoMode,
    pub upgrade_after_failures: u32,
    pub policy_version: String,
    stickiness: Arc<Mutex<StickinessState>>,
    pub outcomes: OutcomeStore,
}

impl Router {
    pub fn new() -> Result<Self, EngineError> {
        Ok(Self {
            execution: ExecutionMode::Hybrid,
            mode: ParetoMode::Balance,
            upgrade_after_failures: 2,
            policy_version: POLICY_VERSION.to_string(),
            stickiness: Arc::new(Mutex::new(StickinessState::default())),
            outcomes: OutcomeStore::new(),
        })
    }

    pub fn with_mode(mut self, mode: ParetoMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_execution(mut self, execution: ExecutionMode) -> Self {
        self.execution = execution;
        self
    }

    /// Complexity at/above which hybrid mode prefers cloud.
    pub fn complexity_cutoff(&self) -> f32 {
        match self.mode {
            ParetoMode::Cost => 0.90,
            ParetoMode::Balance => 0.75,
            ParetoMode::Intelligence => 0.40,
        }
    }

    /// Project raw signals into a routing band (no stickiness yet).
    pub fn project(&self, signal: &RouteSignal) -> (ProjectionBand, String) {
        if self.execution == ExecutionMode::Device {
            return (ProjectionBand::MustLocal, "execution_device".into());
        }
        if signal.privacy_sensitive {
            return (ProjectionBand::MustLocal, "privacy_sensitive".into());
        }
        if self.execution == ExecutionMode::Cloud {
            // Force handoff; unavailable cloud still PreferCloud so chat errors
            // instead of silently decoding locally.
            return (ProjectionBand::PreferCloud, "execution_cloud".into());
        }

        let context_overflow = signal.context_tokens > signal.context_limit;
        let need_cloud = signal.force_cloud
            || signal.modality_unsupported_locally
            || context_overflow
            || signal.consecutive_local_failures >= self.upgrade_after_failures
            || signal.complexity >= self.complexity_cutoff();

        if need_cloud {
            if !signal.cloud_available {
                return (
                    ProjectionBand::MustLocal,
                    "prefer_cloud_but_unavailable".into(),
                );
            }
            let reason = if signal.force_cloud {
                "force_cloud"
            } else if signal.modality_unsupported_locally {
                "modality_unsupported_locally"
            } else if context_overflow {
                "context_overflow"
            } else if signal.consecutive_local_failures >= self.upgrade_after_failures {
                "local_failures_upgrade"
            } else {
                "high_complexity"
            };
            return (ProjectionBand::PreferCloud, reason.into());
        }

        (ProjectionBand::LocalOk, "local_ok".into())
    }

    pub fn route(&self, signal: &RouteSignal) -> RouteDecision {
        let (projection, proj_reason) = self.project(signal);
        let tentative = match projection {
            ProjectionBand::MustLocal => RouteAction::Local,
            ProjectionBand::LocalOk => RouteAction::Local,
            ProjectionBand::PreferCloud => RouteAction::CloudHandoff,
        };
        let fallback = match tentative {
            RouteAction::Local => RouteAction::CloudHandoff,
            RouteAction::CloudHandoff => RouteAction::Local,
        };

        let (action, reason) = self.apply_stickiness(signal, projection, tentative, &proj_reason);

        RouteDecision {
            action,
            reason,
            policy_version: self.policy_version.clone(),
            fallback,
            projection,
            mode: self.mode,
            layer: RouteLayer::Rules,
            confidence: signal.confidence,
            semantic_consulted: false,
            semantic_latency_ms: None,
        }
    }

    /// Backward-compatible confidence-only entry.
    pub fn route_confidence(&self, confidence: f32) -> RouteDecision {
        self.route(&RouteSignal::from_confidence(confidence))
    }

    /// Two-layer hybrid routing (P2): rule fast path → semantic slow path
    /// (only when rules are undecided) → health fallback flip → stickiness.
    ///
    /// With a disabled/unavailable semantic layer and healthy backends this
    /// degrades to exactly the P0/P1 `route()` outcome.
    pub async fn route_hybrid(
        &self,
        signal: &RouteSignal,
        prompt: &str,
        semantic: &SemanticRouter,
        health: &HealthTracker,
    ) -> RouteDecision {
        let engine = RuleEngine::new(self.execution, self.mode, self.upgrade_after_failures);
        let rule = engine.evaluate(signal, prompt);
        let (projection, proj_reason) = self.project(signal);

        let mut layer = RouteLayer::Rules;
        let mut confidence = rule.confidence;
        let mut semantic_consulted = false;
        let mut semantic_latency_ms = None;
        let mut policy_version = self.policy_version.clone();
        let mut reason = rule.reason.clone();

        let mut tentative = match rule.action {
            Some(action) => action,
            None => {
                let mut resolved = None;
                if rule.need_semantic && semantic.is_enabled() {
                    semantic_consulted = true;
                    let started = std::time::Instant::now();
                    let outcome = semantic.consult(signal, prompt).await;
                    semantic_latency_ms = Some(started.elapsed().as_millis() as u64);
                    if let Some(sd) = outcome {
                        let conflicts = matches!(
                            (projection, sd.action),
                            (ProjectionBand::MustLocal, RouteAction::CloudHandoff)
                        ) || (self.execution == ExecutionMode::Cloud
                            && sd.action == RouteAction::Local);
                        if sd.confidence >= SEMANTIC_ACCEPT_THRESHOLD && !conflicts {
                            resolved = Some(sd.action);
                            layer = RouteLayer::Semantic;
                            confidence = sd.confidence;
                            reason = format!("semantic:{}:{}", sd.intent, sd.reason);
                            policy_version = format!("{}+semantic", self.policy_version);
                        }
                    }
                }
                match resolved {
                    Some(action) => action,
                    None => {
                        // Degrade to the plain projection outcome (≡ route()).
                        reason = if semantic_consulted {
                            format!("semantic_fallback:{proj_reason}")
                        } else {
                            proj_reason.clone()
                        };
                        match projection {
                            ProjectionBand::MustLocal | ProjectionBand::LocalOk => {
                                RouteAction::Local
                            }
                            ProjectionBand::PreferCloud => RouteAction::CloudHandoff,
                        }
                    }
                }
            }
        };

        // Health fallback flip: soft decisions only, never MustLocal.
        if projection != ProjectionBand::MustLocal {
            match tentative {
                RouteAction::CloudHandoff
                    if !health.healthy(BackendKind::Cloud)
                        && health.healthy(BackendKind::Local) =>
                {
                    reason = format!("{reason}|health_flip:cloud_unhealthy");
                    tentative = RouteAction::Local;
                }
                RouteAction::Local
                    if !health.healthy(BackendKind::Local)
                        && health.healthy(BackendKind::Cloud)
                        && signal.cloud_available =>
                {
                    reason = format!("{reason}|health_flip:local_unhealthy");
                    tentative = RouteAction::CloudHandoff;
                }
                _ => {}
            }
        }

        let fallback = match tentative {
            RouteAction::Local => RouteAction::CloudHandoff,
            RouteAction::CloudHandoff => RouteAction::Local,
        };
        let (action, reason) = self.apply_stickiness(signal, projection, tentative, &reason);

        RouteDecision {
            action,
            reason,
            policy_version,
            fallback,
            projection,
            mode: self.mode,
            layer,
            confidence,
            semantic_consulted,
            semantic_latency_ms,
        }
    }

    fn apply_stickiness(
        &self,
        signal: &RouteSignal,
        projection: ProjectionBand,
        tentative: RouteAction,
        proj_reason: &str,
    ) -> (RouteAction, String) {
        let Some(sid) = signal.session_id.as_deref() else {
            self.commit_session(None, tentative);
            return (tentative, proj_reason.to_string());
        };

        let mut guard = self.stickiness.lock().expect("stickiness poisoned");
        if let Some(prev) = guard.last.get(sid).copied() {
            match (prev, projection, tentative) {
                // Hard local always wins.
                (_, ProjectionBand::MustLocal, _) => {
                    guard.last.insert(sid.to_string(), RouteAction::Local);
                    return (RouteAction::Local, format!("sticky_override:{proj_reason}"));
                }
                // Stay on cloud once handed off (unless MustLocal above).
                (RouteAction::CloudHandoff, _, _) => {
                    return (
                        RouteAction::CloudHandoff,
                        format!("session_sticky_cloud:{proj_reason}"),
                    );
                }
                // Stay local unless hard upgrade (failures / modality / force /
                // overflow / execution=cloud). P2: also guards semantic-layer
                // soft upgrades where projection may be LocalOk.
                (RouteAction::Local, _, RouteAction::CloudHandoff) => {
                    let hard_upgrade = self.execution == ExecutionMode::Cloud
                        || signal.force_cloud
                        || signal.modality_unsupported_locally
                        || signal.context_tokens > signal.context_limit
                        || signal.consecutive_local_failures >= self.upgrade_after_failures;
                    if hard_upgrade {
                        guard
                            .last
                            .insert(sid.to_string(), RouteAction::CloudHandoff);
                        return (
                            RouteAction::CloudHandoff,
                            format!("session_upgrade:{proj_reason}"),
                        );
                    }
                    return (
                        RouteAction::Local,
                        format!("session_sticky_local:{proj_reason}"),
                    );
                }
                (RouteAction::Local, _, RouteAction::Local) => {
                    return (
                        RouteAction::Local,
                        format!("session_sticky_local:{proj_reason}"),
                    );
                }
            }
        }

        guard.last.insert(sid.to_string(), tentative);
        (tentative, proj_reason.to_string())
    }

    fn commit_session(&self, session_id: Option<&str>, action: RouteAction) {
        if let Some(sid) = session_id {
            self.stickiness
                .lock()
                .expect("stickiness poisoned")
                .last
                .insert(sid.to_string(), action);
        }
    }

    pub fn record_outcome(&self, outcome: RouteOutcome) {
        self.outcomes.record(outcome);
    }

    pub fn clear_session(&self, session_id: &str) {
        self.stickiness
            .lock()
            .expect("stickiness poisoned")
            .last
            .remove(session_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::{HealthEvent, HealthTracker};
    use crate::semantic::{FakeSemanticClient, SemanticClient, SemanticDecision, SemanticRouter};
    use std::future::Future;
    use std::pin::Pin;

    #[derive(Debug)]
    struct FakeSemantic {
        result: Option<SemanticDecision>,
    }

    impl FakeSemanticClient for FakeSemantic {
        fn consult<'a>(
            &'a self,
            _signal: &'a RouteSignal,
            _prompt: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<SemanticDecision, EngineError>> + Send + 'a>>
        {
            let r = self.result.clone();
            Box::pin(async move { r.ok_or_else(|| EngineError::Cloud("fake down".into())) })
        }
    }

    fn semantic_with(result: Option<SemanticDecision>) -> SemanticRouter {
        SemanticRouter::new(
            Some(SemanticClient::Fake(Arc::new(FakeSemantic { result }))),
            100,
            8,
        )
    }

    fn sd(action: RouteAction, confidence: f32) -> SemanticDecision {
        SemanticDecision {
            action,
            confidence,
            intent: "agent".into(),
            reason: "needs strong model".into(),
        }
    }

    /// Rule fast path: clear-local signal, semantic never consulted.
    #[tokio::test]
    async fn hybrid_rule_fast_path_skips_semantic() {
        let r = Router::new().unwrap();
        let sem = semantic_with(Some(sd(RouteAction::CloudHandoff, 0.99)));
        let health = HealthTracker::new();
        let sig = RouteSignal::from_confidence(0.95);
        let d = r.route_hybrid(&sig, "hi", &sem, &health).await;
        assert_eq!(d.action, RouteAction::Local);
        assert_eq!(d.layer, RouteLayer::Rules);
        assert!(!d.semantic_consulted);
        assert_eq!(d.semantic_latency_ms, None);
        assert_eq!(d.reason, "rule:inline_local");
        // Matches the P0/P1 route() action for the same signal.
        assert_eq!(d.action, r.route(&sig).action);
    }

    /// Undecided rules → semantic adopted (high confidence).
    #[tokio::test]
    async fn hybrid_semantic_adopted_on_uncertain_rules() {
        let r = Router::new().unwrap();
        let sem = semantic_with(Some(sd(RouteAction::CloudHandoff, 0.9)));
        let health = HealthTracker::new();
        let mut sig = RouteSignal::from_confidence(0.95);
        sig.complexity = 0.7; // Balance neighborhood [0.675, 0.825)
        let d = r.route_hybrid(&sig, "hi", &sem, &health).await;
        assert_eq!(d.action, RouteAction::CloudHandoff);
        assert_eq!(d.layer, RouteLayer::Semantic);
        assert!(d.semantic_consulted);
        assert!(d.semantic_latency_ms.is_some());
        assert!(d.policy_version.ends_with("+semantic"));
        assert!(d.reason.starts_with("semantic:agent:"));
        assert!((d.confidence - 0.9).abs() < 1e-6);
    }

    /// Low-confidence semantic answer rejected → degrade to projection.
    #[tokio::test]
    async fn hybrid_semantic_low_confidence_rejected() {
        let r = Router::new().unwrap();
        let sem = semantic_with(Some(sd(RouteAction::CloudHandoff, 0.4)));
        let health = HealthTracker::new();
        let mut sig = RouteSignal::from_confidence(0.95);
        sig.complexity = 0.7;
        let d = r.route_hybrid(&sig, "hi", &sem, &health).await;
        assert_eq!(d.action, RouteAction::Local);
        assert_eq!(d.layer, RouteLayer::Rules);
        assert!(d.semantic_consulted);
        assert!(d.reason.contains("semantic_fallback"));
        assert_eq!(d.policy_version, POLICY_VERSION);
    }

    /// Semantic failure / timeout → silent degrade to rule-layer outcome.
    #[tokio::test]
    async fn hybrid_semantic_failure_silent_degrade() {
        let r = Router::new().unwrap();
        let sem = semantic_with(None); // fake always errors
        let health = HealthTracker::new();
        let mut sig = RouteSignal::from_confidence(0.95);
        sig.complexity = 0.7;
        let d = r.route_hybrid(&sig, "hi", &sem, &health).await;
        assert_eq!(d.action, RouteAction::Local);
        assert_eq!(d.layer, RouteLayer::Rules);
        assert!(d.semantic_consulted);
        assert!(d.reason.contains("semantic_fallback"));
    }

    /// Semantic cloud answer conflicting with MustLocal is never adopted.
    #[tokio::test]
    async fn hybrid_semantic_never_overrides_hard_local() {
        let r = Router::new().unwrap();
        let sem = semantic_with(Some(sd(RouteAction::CloudHandoff, 0.99)));
        let health = HealthTracker::new();
        let mut sig = RouteSignal::from_confidence(0.95);
        sig.complexity = 0.78; // ≥ cutoff but in neighborhood → rules undecided
        sig.cloud_available = false; // project() → MustLocal
        let d = r.route_hybrid(&sig, "hi", &sem, &health).await;
        assert_eq!(d.action, RouteAction::Local);
        assert_eq!(d.projection, ProjectionBand::MustLocal);
        assert_eq!(d.layer, RouteLayer::Rules);
    }

    /// Health flip: cloud handoff flips local when cloud backend unhealthy.
    #[tokio::test]
    async fn hybrid_health_flip_cloud_to_local() {
        let r = Router::new().unwrap();
        let sem = SemanticRouter::disabled();
        let health = HealthTracker::new();
        for _ in 0..3 {
            health.record(BackendKind::Cloud, HealthEvent::Failure);
        }
        let mut sig = RouteSignal::from_confidence(0.99);
        sig.force_cloud = true;
        let d = r.route_hybrid(&sig, "hi", &sem, &health).await;
        assert_eq!(d.action, RouteAction::Local);
        assert!(d.reason.contains("health_flip:cloud_unhealthy"));
    }

    /// Health flip: local decision flips cloud when local backend unhealthy.
    #[tokio::test]
    async fn hybrid_health_flip_local_to_cloud() {
        let r = Router::new().unwrap();
        let sem = SemanticRouter::disabled();
        let health = HealthTracker::new();
        for _ in 0..4 {
            health.record(BackendKind::Local, HealthEvent::Failure);
        }
        let sig = RouteSignal::from_confidence(0.95);
        let d = r.route_hybrid(&sig, "hi", &sem, &health).await;
        assert_eq!(d.action, RouteAction::CloudHandoff);
        assert!(d.reason.contains("health_flip:local_unhealthy"));
    }

    /// Hard constraints are never flipped by health.
    #[tokio::test]
    async fn hybrid_health_never_flips_must_local() {
        let r = Router::new().unwrap();
        let sem = SemanticRouter::disabled();
        let health = HealthTracker::new();
        for _ in 0..4 {
            health.record(BackendKind::Local, HealthEvent::Failure);
        }
        let mut sig = RouteSignal::from_confidence(0.95);
        sig.privacy_sensitive = true;
        let d = r.route_hybrid(&sig, "hi", &sem, &health).await;
        assert_eq!(d.action, RouteAction::Local);
        assert_eq!(d.projection, ProjectionBand::MustLocal);
        assert!(!d.reason.contains("health_flip"));
    }

    /// Disabled semantic layer ≡ P0/P1 route() across a signal battery.
    #[tokio::test]
    async fn hybrid_disabled_semantic_matches_route() {
        let r = Router::new().unwrap();
        let sem = SemanticRouter::disabled();
        let health = HealthTracker::new();
        let cases: Vec<(RouteSignal, &str)> = vec![
            (RouteSignal::from_confidence(0.95), "hi"),
            (
                RouteSignal::from_confidence(0.95),
                "please refactor this module",
            ),
            (
                {
                    let mut s = RouteSignal::from_confidence(0.95);
                    s.complexity = 0.7;
                    s
                },
                "hi",
            ),
            (
                {
                    let mut s = RouteSignal::from_confidence(0.95);
                    s.complexity = 0.9;
                    s
                },
                "hi",
            ),
            (
                {
                    let mut s = RouteSignal::from_confidence(0.95);
                    s.force_cloud = true;
                    s
                },
                "hi",
            ),
        ];
        for (sig, prompt) in cases {
            let hybrid = r.route_hybrid(&sig, prompt, &sem, &health).await;
            let plain = r.route(&sig);
            assert_eq!(hybrid.action, plain.action, "prompt={prompt:?}");
            assert_eq!(hybrid.layer, RouteLayer::Rules);
            assert!(!hybrid.semantic_consulted);
        }
    }

    /// Stickiness still applies after the two-layer pipeline.
    #[tokio::test]
    async fn hybrid_stickiness_preserved() {
        let r = Router::new().unwrap();
        let sem = SemanticRouter::disabled();
        let health = HealthTracker::new();
        let mut sig = RouteSignal::from_confidence(0.95);
        sig.session_id = Some("p2-stick".into());
        let d1 = r.route_hybrid(&sig, "hi", &sem, &health).await;
        assert_eq!(d1.action, RouteAction::Local);

        // Soft semantic upgrade is blocked by sticky local.
        let sem = semantic_with(Some(sd(RouteAction::CloudHandoff, 0.9)));
        sig.complexity = 0.7;
        let d2 = r.route_hybrid(&sig, "hi", &sem, &health).await;
        assert_eq!(d2.action, RouteAction::Local);
        assert!(d2.reason.contains("sticky_local"));

        // Hard force_cloud upgrades.
        sig.force_cloud = true;
        let d3 = r.route_hybrid(&sig, "hi", &sem, &health).await;
        assert_eq!(d3.action, RouteAction::CloudHandoff);
        r.clear_session("p2-stick");
    }
}
