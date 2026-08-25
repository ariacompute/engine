//! Hybrid router: local inference vs cloud handoff (signal → projection → decision).
//!
//! P2 adds the two-layer hybrid routing entry `Router::route_hybrid`:
//! rule layer (fast path) + semantic layer (slow path) + health fallback.

mod health;
mod policy;
mod route;
mod rules;
mod semantic;

pub use health::{BackendKind, HealthEvent, HealthSnapshot, HealthTracker, HEALTHY_THRESHOLD};
pub use policy::{chat_policy, ChatPolicy, EffectiveRouting};
pub use route::{
    ExecutionMode, OutcomeStore, ParetoMode, ProjectionBand, RouteAction, RouteDecision,
    RouteLayer, RouteOutcome, RouteSignal, Router, POLICY_VERSION,
};
pub use rules::{classify, RequestKind, RuleDecision, RuleEngine};
pub use semantic::{
    CloudSemanticClient, FakeSemanticClient, SemanticClient, SemanticDecision, SemanticRouter,
    DEFAULT_SEMANTIC_CACHE_SIZE, DEFAULT_SEMANTIC_TIMEOUT_MS, SEMANTIC_ACCEPT_THRESHOLD,
    SEMANTIC_CACHE_TTL,
};

use aria_kernel::EngineError;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Model id posted to Aria gateway on cloud handoff.
pub const CLOUD_GATEWAY_MODEL: &str = "ariacompute/ariamodel";

/// Gateway thinking models often take >25s for a full answer; 5s would cut them off.
pub const DEFAULT_CLOUD_CHAT_TIMEOUT_MS: u64 = 60_000;

/// Estimate hybrid routing signals from prompt text and model context limit.
///
/// - `context_tokens`: ~chars/4 heuristic (OpenAI-style rough estimate)
/// - `complexity` in `[0,1]`: length score + keyword boost (EN/ZH)
pub fn estimate_route_signals(prompt: &str, context_limit: u32) -> (f32, u32) {
    let chars = prompt.chars().count();
    let context_tokens = chars.div_ceil(4) as u32;
    let _ = context_limit; // caller uses for overflow; kept for API clarity

    // Length: short prompts stay easy; long prompts ramp toward hard.
    let length_score = if chars == 0 {
        0.0
    } else if chars < 80 {
        0.15
    } else if chars < 240 {
        0.35
    } else if chars < 800 {
        0.55
    } else if chars < 2000 {
        0.75
    } else {
        0.90
    };

    let lower = prompt.to_ascii_lowercase();
    let keywords = [
        "analyze",
        "analyse",
        "step-by-step",
        "step by step",
        "multi-step",
        "multistep",
        "reason",
        "reasoning",
        "plan",
        "compare",
        "debug",
        "refactor",
        "architecture",
        "分析",
        "推理",
        "逐步",
        "多步",
        "对比",
        "规划",
        "架构",
        "调试",
    ];
    let mut boost = 0.0f32;
    for k in keywords {
        if lower.contains(k) || prompt.contains(k) {
            boost += 0.18;
        }
    }
    boost = boost.min(0.55);
    let complexity = (length_score + boost).clamp(0.0, 1.0);
    (complexity, context_tokens)
}

#[derive(Debug, Clone)]
pub struct CloudClient {
    pub base_url: String,
    pub api_key: String,
    pub timeout_ms: u64,
    /// When set, `chat` returns this JSON without HTTP (Stage A tests).
    pub mock: Option<MockMode>,
}

#[derive(Debug, Clone)]
pub enum MockMode {
    Success(Value),
    FailStatus(u16),
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudChatRequest {
    pub model: String,
    pub messages: Vec<CloudMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Semantic classifier must not spend the 800ms budget on thinking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_thinking: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudMessage {
    pub role: String,
    pub content: String,
}

impl CloudClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            timeout_ms: DEFAULT_CLOUD_CHAT_TIMEOUT_MS,
            mock: None,
        }
    }

    pub fn with_mock(mut self, mock: MockMode) -> Self {
        self.mock = Some(mock);
        self
    }

    /// True when a mock is configured or an API key is present.
    pub fn is_available(&self) -> bool {
        self.mock.is_some() || !self.api_key.is_empty()
    }

    pub async fn chat(&self, req: &CloudChatRequest) -> Result<Value, EngineError> {
        if let Some(mock) = &self.mock {
            return match mock {
                MockMode::Success(v) => Ok(v.clone()),
                MockMode::FailStatus(code) => {
                    Err(EngineError::Cloud(format!("mock non-2xx status {code}")))
                }
                MockMode::Timeout => Err(EngineError::Cloud("mock timeout".into())),
            };
        }
        if self.api_key.is_empty() {
            return Err(EngineError::Cloud("cloud API key not set".into()));
        }
        let url = format!(
            "{}/v1/chat/completions",
            self.base_url.trim_end_matches('/')
        );
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(self.timeout_ms))
            .build()
            .map_err(|e| EngineError::Cloud(e.to_string()))?;
        let resp = client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(req)
            .send()
            .await
            .map_err(|e| EngineError::Cloud(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(EngineError::Cloud(format!("HTTP {status}")));
        }
        resp.json::<Value>()
            .await
            .map_err(|e| EngineError::Cloud(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn cloud_gateway_model_id() {
        assert_eq!(CLOUD_GATEWAY_MODEL, "ariacompute/ariamodel");
        assert_eq!(DEFAULT_CLOUD_CHAT_TIMEOUT_MS, 60_000);
        let req = CloudChatRequest {
            model: CLOUD_GATEWAY_MODEL.to_string(),
            messages: vec![],
            max_tokens: Some(8),
            enable_thinking: None,
        };
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["model"], "ariacompute/ariamodel");
        let omit = CloudChatRequest {
            model: CLOUD_GATEWAY_MODEL.to_string(),
            messages: vec![],
            max_tokens: None,
            enable_thinking: None,
        };
        let v = serde_json::to_value(&omit).unwrap();
        assert!(v.get("max_tokens").is_none());
        let classifier = CloudChatRequest {
            model: CLOUD_GATEWAY_MODEL.to_string(),
            messages: vec![],
            max_tokens: Some(128),
            enable_thinking: Some(false),
        };
        let v = serde_json::to_value(&classifier).unwrap();
        assert_eq!(v["enable_thinking"], false);
    }

    #[test]
    fn router_default_stays_local_on_confidence_only() {
        let r = Router::new().unwrap();
        // Confidence is unused for handoff; complexity defaults to 0.
        assert_eq!(r.route_confidence(0.9).action, RouteAction::Local);
        assert_eq!(r.route_confidence(0.1).action, RouteAction::Local);
        let mut r2 = Router::new().unwrap();
        r2.execution = ExecutionMode::Device;
        assert_eq!(r2.route_confidence(0.0).action, RouteAction::Local);
    }

    #[test]
    fn execution_mode_device_and_cloud() {
        assert_eq!(ExecutionMode::parse("").unwrap(), ExecutionMode::Hybrid);
        assert_eq!(
            ExecutionMode::parse("hybrid").unwrap(),
            ExecutionMode::Hybrid
        );
        assert_eq!(
            ExecutionMode::parse("DEVICE").unwrap(),
            ExecutionMode::Device
        );
        assert_eq!(ExecutionMode::parse("cloud").unwrap(), ExecutionMode::Cloud);
        assert!(ExecutionMode::parse("gpu").is_err());

        let device = Router::new().unwrap().with_execution(ExecutionMode::Device);
        let d = device.route(&RouteSignal {
            force_cloud: true,
            confidence: 0.0,
            ..RouteSignal::from_confidence(0.0)
        });
        assert_eq!(d.action, RouteAction::Local);
        assert_eq!(d.reason, "execution_device");

        let cloud = Router::new().unwrap().with_execution(ExecutionMode::Cloud);
        let c = cloud.route(&RouteSignal::from_confidence(0.99));
        assert_eq!(c.action, RouteAction::CloudHandoff);
        assert_eq!(c.reason, "execution_cloud");

        // Privacy still wins over cloud execution.
        let mut priv_sig = RouteSignal::from_confidence(0.0);
        priv_sig.privacy_sensitive = true;
        let p = cloud.route(&priv_sig);
        assert_eq!(p.action, RouteAction::Local);
        assert_eq!(p.reason, "privacy_sensitive");

        // Cloud mode with unavailable backend still PreferCloud (handoff path).
        let mut no_cloud = RouteSignal::from_confidence(0.99);
        no_cloud.cloud_available = false;
        let u = cloud.route(&no_cloud);
        assert_eq!(u.action, RouteAction::CloudHandoff);
        assert_eq!(u.reason, "execution_cloud");
    }

    #[test]
    fn pareto_modes_complexity_cutoffs() {
        let base = Router::new().unwrap();
        let cost = Router::new().unwrap().with_mode(ParetoMode::Cost);
        let intel = Router::new().unwrap().with_mode(ParetoMode::Intelligence);
        assert!((base.complexity_cutoff() - 0.75).abs() < 1e-6);
        assert!((cost.complexity_cutoff() - 0.90).abs() < 1e-6);
        assert!((intel.complexity_cutoff() - 0.40).abs() < 1e-6);

        // Mid complexity 0.5 → Cost/Balance local, Intelligence cloud.
        let mid = |mode: ParetoMode| {
            let r = Router::new().unwrap().with_mode(mode);
            let mut sig = RouteSignal::from_confidence(0.95);
            sig.complexity = 0.5;
            r.route(&sig).action
        };
        assert_eq!(mid(ParetoMode::Cost), RouteAction::Local);
        assert_eq!(mid(ParetoMode::Balance), RouteAction::Local);
        assert_eq!(mid(ParetoMode::Intelligence), RouteAction::CloudHandoff);

        // High complexity 0.8 → Cost local, Balance/Intelligence cloud.
        let high = |mode: ParetoMode| {
            let r = Router::new().unwrap().with_mode(mode);
            let mut sig = RouteSignal::from_confidence(0.95);
            sig.complexity = 0.8;
            r.route(&sig).action
        };
        assert_eq!(high(ParetoMode::Cost), RouteAction::Local);
        assert_eq!(high(ParetoMode::Balance), RouteAction::CloudHandoff);
        assert_eq!(high(ParetoMode::Intelligence), RouteAction::CloudHandoff);
    }

    #[test]
    fn hard_constraints_and_projection() {
        let r = Router::new().unwrap();
        let mut sig = RouteSignal::from_confidence(0.9);
        sig.privacy_sensitive = true;
        let d = r.route(&sig);
        assert_eq!(d.action, RouteAction::Local);
        assert_eq!(d.projection, ProjectionBand::MustLocal);
        assert!(d.reason.contains("privacy"));

        let mut sig = RouteSignal::from_confidence(0.9);
        sig.modality_unsupported_locally = true;
        sig.cloud_available = true;
        let d = r.route(&sig);
        assert_eq!(d.action, RouteAction::CloudHandoff);
        assert_eq!(d.projection, ProjectionBand::PreferCloud);

        let mut sig = RouteSignal::from_confidence(0.0);
        sig.complexity = 0.99;
        sig.cloud_available = false;
        let d = r.route(&sig);
        assert_eq!(d.action, RouteAction::Local);
        assert_eq!(d.projection, ProjectionBand::MustLocal);
        assert!(d.reason.contains("unavailable"));

        let mut sig = RouteSignal::from_confidence(0.9);
        sig.context_tokens = 100;
        sig.context_limit = 50;
        let d = r.route(&sig);
        assert_eq!(d.action, RouteAction::CloudHandoff);
        assert!(d.reason.contains("context"));
    }

    #[test]
    fn context_overflow_and_force_cloud_all_modes() {
        for mode in [
            ParetoMode::Cost,
            ParetoMode::Balance,
            ParetoMode::Intelligence,
        ] {
            let r = Router::new().unwrap().with_mode(mode);
            let mut overflow = RouteSignal::from_confidence(0.95);
            overflow.complexity = 0.0;
            overflow.context_tokens = 100;
            overflow.context_limit = 50;
            assert_eq!(
                r.route(&overflow).action,
                RouteAction::CloudHandoff,
                "overflow {mode:?}"
            );

            let mut force = RouteSignal::from_confidence(0.95);
            force.complexity = 0.0;
            force.force_cloud = true;
            assert_eq!(
                r.route(&force).action,
                RouteAction::CloudHandoff,
                "force {mode:?}"
            );
        }
    }

    #[test]
    fn session_stickiness_and_upgrade() {
        let r = Router::new().unwrap();
        let mut sig = RouteSignal::from_confidence(0.9);
        sig.session_id = Some("s1".into());
        let d1 = r.route(&sig);
        assert_eq!(d1.action, RouteAction::Local);

        // Soft prefer-cloud (high complexity) should stick to Local.
        sig.complexity = 0.9;
        let d2 = r.route(&sig);
        assert_eq!(d2.action, RouteAction::Local);
        assert!(d2.reason.contains("sticky_local"));

        // Hard upgrade via failures.
        sig.consecutive_local_failures = 2;
        let d3 = r.route(&sig);
        assert_eq!(d3.action, RouteAction::CloudHandoff);
        assert!(d3.reason.contains("upgrade"));

        // After cloud, stick to cloud.
        sig.complexity = 0.0;
        sig.consecutive_local_failures = 0;
        let d4 = r.route(&sig);
        assert_eq!(d4.action, RouteAction::CloudHandoff);
        assert!(d4.reason.contains("sticky_cloud"));

        // Privacy forces local.
        sig.privacy_sensitive = true;
        let d5 = r.route(&sig);
        assert_eq!(d5.action, RouteAction::Local);
        assert_eq!(d5.projection, ProjectionBand::MustLocal);
    }

    #[test]
    fn decision_carries_policy_metadata() {
        let r = Router::new().unwrap();
        let mut sig = RouteSignal::from_confidence(0.95);
        sig.complexity = 0.9;
        let d = r.route(&sig);
        assert_eq!(d.action, RouteAction::CloudHandoff);
        assert_eq!(d.policy_version, POLICY_VERSION);
        assert_eq!(d.fallback, RouteAction::Local);
        assert_eq!(d.mode, ParetoMode::Balance);
        assert!(!d.reason.is_empty());
    }

    #[test]
    fn outcome_store_records() {
        let r = Router::new().unwrap();
        let mut sig = RouteSignal::from_confidence(0.95);
        sig.complexity = 0.9;
        let d = r.route(&sig);
        r.record_outcome(RouteOutcome {
            task_id: "t1".into(),
            session_id: None,
            action: d.action,
            reason: d.reason.clone(),
            policy_version: d.policy_version.clone(),
            mode: d.mode,
            input_tokens: Some(10),
            output_tokens: Some(4),
            latency_ms: Some(12),
            cloud_handoff: true,
            user_corrected: Some(false),
            validation_ok: Some(true),
            layer: RouteLayer::Rules,
            confidence: 0.95,
            semantic_consulted: false,
            semantic_latency_ms: None,
        });
        assert_eq!(r.outcomes.len(), 1);
        let recent = r.outcomes.recent(1);
        assert_eq!(recent[0].task_id, "t1");
        assert!(recent[0].cloud_handoff);
    }

    #[test]
    fn high_complexity_prefers_cloud() {
        let r = Router::new().unwrap();
        let mut sig = RouteSignal::from_confidence(0.95);
        sig.complexity = 0.8;
        let d = r.route(&sig);
        assert_eq!(d.action, RouteAction::CloudHandoff);
        assert!(d.reason.contains("complexity"));
    }

    #[test]
    fn estimate_route_signals_length_and_keywords() {
        let (c_short, tok) = estimate_route_signals("hi", 4096);
        assert!(c_short < 0.4);
        assert!(tok > 0);
        let (c_kw, _) = estimate_route_signals(
            "Please analyze and reason step-by-step about this plan",
            4096,
        );
        assert!(c_kw >= 0.40);
        assert!(c_kw < 0.75);
        let long = "x".repeat(900);
        let (c_long, _) = estimate_route_signals(&long, 4096);
        assert!(c_long >= 0.75);
        assert!(c_long < 0.90);
    }

    #[tokio::test]
    async fn mock_cloud_ok_and_fail() {
        let ok = CloudClient::new("http://127.0.0.1:9", "").with_mock(MockMode::Success(
            json!({"choices":[{"message":{"content":"hi"}}]}),
        ));
        assert!(ok.is_available());
        let v = ok
            .chat(&CloudChatRequest {
                model: "x".into(),
                messages: vec![CloudMessage {
                    role: "user".into(),
                    content: "a".into(),
                }],
                max_tokens: Some(8),
                enable_thinking: None,
            })
            .await
            .unwrap();
        assert!(v["choices"][0]["message"]["content"].is_string());

        let bad = CloudClient::new("http://127.0.0.1:9", "").with_mock(MockMode::FailStatus(503));
        let err = bad
            .chat(&CloudChatRequest {
                model: "x".into(),
                messages: vec![],
                max_tokens: None,
                enable_thinking: None,
            })
            .await
            .unwrap_err();
        assert!(matches!(err, EngineError::Cloud(_)));

        let to = CloudClient::new("http://127.0.0.1:9", "").with_mock(MockMode::Timeout);
        let err = to
            .chat(&CloudChatRequest {
                model: "x".into(),
                messages: vec![],
                max_tokens: None,
                enable_thinking: None,
            })
            .await
            .unwrap_err();
        assert!(matches!(err, EngineError::Cloud(_)));
        assert!(err.to_string().contains("timeout"));
    }

    #[test]
    fn router_new_ok() {
        assert!(Router::new().is_ok());
        let r = Router::new().unwrap();
        assert_eq!(r.route_confidence(0.0).action, RouteAction::Local);
        let intel = Router::new().unwrap().with_mode(ParetoMode::Intelligence);
        let mut sig = RouteSignal::from_confidence(0.95);
        sig.complexity = 0.5;
        assert_eq!(intel.route(&sig).action, RouteAction::CloudHandoff);
    }

    #[test]
    fn force_cloud_and_modality_without_cloud() {
        let r = Router::new().unwrap();
        let mut sig = RouteSignal::from_confidence(0.99);
        sig.force_cloud = true;
        let d = r.route(&sig);
        assert_eq!(d.action, RouteAction::CloudHandoff);
        assert!(d.reason.contains("force_cloud"));
        assert_eq!(d.fallback, RouteAction::Local);

        let mut sig = RouteSignal::from_confidence(0.99);
        sig.modality_unsupported_locally = true;
        sig.cloud_available = false;
        let d = r.route(&sig);
        assert_eq!(d.action, RouteAction::Local);
        assert_eq!(d.projection, ProjectionBand::MustLocal);
        assert!(d.reason.contains("unavailable"));
    }

    #[test]
    fn project_api_and_custom_upgrade_threshold() {
        let mut r = Router::new().unwrap();
        r.upgrade_after_failures = 3;
        let mut sig = RouteSignal::from_confidence(0.9);
        sig.consecutive_local_failures = 2;
        let (band, reason) = r.project(&sig);
        assert_eq!(band, ProjectionBand::LocalOk);
        assert_eq!(reason, "local_ok");

        sig.consecutive_local_failures = 3;
        let (band, reason) = r.project(&sig);
        assert_eq!(band, ProjectionBand::PreferCloud);
        assert!(reason.contains("failures"));
    }

    #[test]
    fn sessions_are_isolated_and_clearable() {
        let r = Router::new().unwrap();
        let mut a = RouteSignal::from_confidence(0.9);
        a.session_id = Some("a".into());
        assert_eq!(r.route(&a).action, RouteAction::Local);

        // Upgrade session a to cloud.
        a.force_cloud = true;
        assert_eq!(r.route(&a).action, RouteAction::CloudHandoff);

        // Session b starts fresh local.
        let mut b = RouteSignal::from_confidence(0.9);
        b.session_id = Some("b".into());
        assert_eq!(r.route(&b).action, RouteAction::Local);

        // a still sticky cloud.
        a.force_cloud = false;
        a.confidence = 0.99;
        assert_eq!(r.route(&a).action, RouteAction::CloudHandoff);

        r.clear_session("a");
        assert_eq!(r.route(&a).action, RouteAction::Local);
    }

    #[test]
    fn sticky_soft_complexity_stays_local() {
        let r = Router::new().unwrap();
        let mut sig = RouteSignal::from_confidence(0.95);
        sig.session_id = Some("sticky-cplx".into());
        assert_eq!(r.route(&sig).action, RouteAction::Local);

        sig.complexity = 0.9;
        let d = r.route(&sig);
        assert_eq!(d.action, RouteAction::Local);
        assert!(d.reason.contains("sticky_local"));
    }

    #[test]
    fn outcome_store_recent_clear_and_empty() {
        let store = OutcomeStore::new();
        assert!(store.is_empty());
        for i in 0..5 {
            store.record(RouteOutcome {
                task_id: format!("t{i}"),
                session_id: None,
                action: RouteAction::Local,
                reason: "local_ok".into(),
                policy_version: POLICY_VERSION.into(),
                mode: ParetoMode::Cost,
                input_tokens: None,
                output_tokens: None,
                latency_ms: None,
                cloud_handoff: false,
                user_corrected: None,
                validation_ok: None,
                layer: RouteLayer::Rules,
                confidence: 0.0,
                semantic_consulted: false,
                semantic_latency_ms: None,
            });
        }
        assert_eq!(store.len(), 5);
        let recent = store.recent(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].task_id, "t3");
        assert_eq!(recent[1].task_id, "t4");
        assert_eq!(store.recent(10).len(), 5);
        store.clear();
        assert!(store.is_empty());
    }

    #[test]
    fn decision_and_outcome_serde_roundtrip() {
        let r = Router::new().unwrap().with_mode(ParetoMode::Intelligence);
        let mut sig = RouteSignal::from_confidence(0.95);
        sig.complexity = 0.5;
        let d = r.route(&sig);
        let v = serde_json::to_value(&d).unwrap();
        let back: RouteDecision = serde_json::from_value(v).unwrap();
        assert_eq!(back.action, d.action);
        assert_eq!(back.mode, ParetoMode::Intelligence);
        assert_eq!(back.projection, d.projection);

        let outcome = RouteOutcome {
            task_id: "x".into(),
            session_id: Some("s".into()),
            action: RouteAction::CloudHandoff,
            reason: "high_complexity".into(),
            policy_version: POLICY_VERSION.into(),
            mode: ParetoMode::Balance,
            input_tokens: Some(1),
            output_tokens: Some(2),
            latency_ms: Some(3),
            cloud_handoff: true,
            user_corrected: Some(true),
            validation_ok: Some(false),
            layer: RouteLayer::Semantic,
            confidence: 0.8,
            semantic_consulted: true,
            semantic_latency_ms: Some(42),
        };
        let v = serde_json::to_value(&outcome).unwrap();
        let back: RouteOutcome = serde_json::from_value(v).unwrap();
        assert_eq!(back, outcome);
    }

    #[test]
    fn cloud_client_availability_without_credentials() {
        let c = CloudClient::new("http://127.0.0.1:9", "");
        assert!(!c.is_available());
        assert_eq!(c.timeout_ms, DEFAULT_CLOUD_CHAT_TIMEOUT_MS);
    }

    #[tokio::test]
    async fn cloud_client_missing_api_key() {
        let c = CloudClient::new("http://127.0.0.1:9", "");
        let err = c
            .chat(&CloudChatRequest {
                model: "x".into(),
                messages: vec![],
                max_tokens: None,
                enable_thinking: None,
            })
            .await
            .unwrap_err();
        assert!(matches!(err, EngineError::Cloud(_)));
        assert!(err.to_string().contains("cloud API key not set"));
    }
}
