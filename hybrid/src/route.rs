//! Signal → projection → decision router (P0/P1).

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
    /// Parse `ARIA_HYBRID_EXECUTION`: `hybrid` (default) | `device` | `cloud`.
    pub fn parse(raw: &str) -> Result<Self, EngineError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "hybrid" => Ok(Self::Hybrid),
            "device" => Ok(Self::Device),
            "cloud" => Ok(Self::Cloud),
            other => Err(EngineError::InvalidParam(format!(
                "ARIA_HYBRID_EXECUTION must be hybrid|device|cloud, got {other:?}"
            ))),
        }
    }
}

/// Pareto position on the cost–intelligence frontier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ParetoMode {
    /// Prefer on-device; harder to hand off.
    Cost,
    /// Default threshold.
    #[default]
    Balance,
    /// Easier cloud handoff for higher quality.
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

/// Request-level routing inputs (P1 signal surface).
#[derive(Debug, Clone, PartialEq)]
pub struct RouteSignal {
    /// Model confidence in `[0, 1]` (higher → stay local when other signals allow).
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

/// Explained routing decision (P0).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteDecision {
    pub action: RouteAction,
    pub reason: String,
    pub policy_version: String,
    pub fallback: RouteAction,
    pub projection: ProjectionBand,
    pub mode: ParetoMode,
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
    pub threshold: f32,
    pub execution: ExecutionMode,
    pub mode: ParetoMode,
    pub upgrade_after_failures: u32,
    pub policy_version: String,
    stickiness: Arc<Mutex<StickinessState>>,
    pub outcomes: OutcomeStore,
}

impl Router {
    pub fn new(threshold: f32) -> Result<Self, EngineError> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(EngineError::InvalidParam(
                "confidence threshold must be in [0,1]".into(),
            ));
        }
        Ok(Self {
            threshold,
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

    /// Effective confidence cutoff: `confidence < threshold` → prefer cloud.
    pub fn effective_threshold(&self) -> f32 {
        match self.mode {
            ParetoMode::Cost => (self.threshold * 0.5).clamp(0.0, 1.0),
            ParetoMode::Balance => self.threshold,
            ParetoMode::Intelligence => (self.threshold + 0.25).clamp(0.0, 1.0),
        }
    }

    /// Project raw signals into a routing band (no stickiness yet).
    pub fn project(&self, signal: &RouteSignal) -> (ProjectionBand, String) {
        if self.execution == ExecutionMode::Device {
            return (ProjectionBand::MustLocal, "execution_device".into());
        }
        if signal.privacy_sensitive {
            return (
                ProjectionBand::MustLocal,
                "privacy_sensitive".into(),
            );
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
            || signal.complexity >= 0.75
            || signal.confidence < self.effective_threshold()
            || signal.consecutive_local_failures >= self.upgrade_after_failures;

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
            } else if signal.complexity >= 0.75 {
                "high_complexity"
            } else {
                "low_confidence"
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
        }
    }

    /// Backward-compatible confidence-only entry.
    pub fn route_confidence(&self, confidence: f32) -> RouteDecision {
        self.route(&RouteSignal::from_confidence(confidence))
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
                // Stay local unless hard PreferCloud upgrade (failures / modality / force).
                (RouteAction::Local, ProjectionBand::PreferCloud, RouteAction::CloudHandoff) => {
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
                _ => {}
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
