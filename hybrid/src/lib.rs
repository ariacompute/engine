//! Hybrid router: local inference vs cloud handoff (signal → projection → decision).

mod route;

pub use route::{
    OutcomeStore, ParetoMode, ProjectionBand, RouteAction, RouteDecision, RouteOutcome,
    RouteSignal, Router, POLICY_VERSION,
};

use aria_kernel::EngineError;
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudMessage {
    pub role: String,
    pub content: String,
}

impl CloudClient {
    pub fn from_env(base_url: impl Into<String>) -> Self {
        let api_key = std::env::var("ARIA_HYBRID_CLOUD_API_KEY").unwrap_or_default();
        Self {
            base_url: base_url.into(),
            api_key,
            timeout_ms: 5_000,
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
                MockMode::FailStatus(code) => Err(EngineError::Cloud(format!(
                    "mock non-2xx status {code}"
                ))),
                MockMode::Timeout => Err(EngineError::Cloud("mock timeout".into())),
            };
        }
        if self.api_key.is_empty() {
            return Err(EngineError::Cloud(
                "ARIA_HYBRID_CLOUD_API_KEY not set".into(),
            ));
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
    fn router_threshold_balance() {
        let r = Router::new(0.5).unwrap();
        assert_eq!(
            r.route_confidence(0.9).action,
            RouteAction::Local
        );
        assert_eq!(
            r.route_confidence(0.1).action,
            RouteAction::CloudHandoff
        );
        let mut r2 = Router::new(0.5).unwrap();
        r2.on_device_only = true;
        assert_eq!(r2.route_confidence(0.0).action, RouteAction::Local);
        assert!(Router::new(1.5).is_err());
    }

    #[test]
    fn pareto_modes_shift_threshold() {
        let base = Router::new(0.5).unwrap();
        let cost = Router::new(0.5).unwrap().with_mode(ParetoMode::Cost);
        let intel = Router::new(0.5)
            .unwrap()
            .with_mode(ParetoMode::Intelligence);
        assert!((base.effective_threshold() - 0.5).abs() < 1e-6);
        assert!((cost.effective_threshold() - 0.25).abs() < 1e-6);
        assert!((intel.effective_threshold() - 0.75).abs() < 1e-6);

        // conf=0.4 → Balance handoff, Cost stays local, Intelligence handoff
        assert_eq!(cost.route_confidence(0.4).action, RouteAction::Local);
        assert_eq!(
            base.route_confidence(0.4).action,
            RouteAction::CloudHandoff
        );
        assert_eq!(
            intel.route_confidence(0.4).action,
            RouteAction::CloudHandoff
        );
        // conf=0.6 → only Intelligence wants cloud
        assert_eq!(base.route_confidence(0.6).action, RouteAction::Local);
        assert_eq!(
            intel.route_confidence(0.6).action,
            RouteAction::CloudHandoff
        );
    }

    #[test]
    fn hard_constraints_and_projection() {
        let r = Router::new(0.5).unwrap();
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
    fn session_stickiness_and_upgrade() {
        let r = Router::new(0.5).unwrap();
        let mut sig = RouteSignal::from_confidence(0.9);
        sig.session_id = Some("s1".into());
        let d1 = r.route(&sig);
        assert_eq!(d1.action, RouteAction::Local);

        // Soft prefer-cloud (low conf) should stick to Local.
        sig.confidence = 0.1;
        let d2 = r.route(&sig);
        assert_eq!(d2.action, RouteAction::Local);
        assert!(d2.reason.contains("sticky_local"));

        // Hard upgrade via failures.
        sig.consecutive_local_failures = 2;
        let d3 = r.route(&sig);
        assert_eq!(d3.action, RouteAction::CloudHandoff);
        assert!(d3.reason.contains("upgrade"));

        // After cloud, stick to cloud.
        sig.confidence = 0.99;
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
        let r = Router::new(0.5).unwrap();
        let d = r.route_confidence(0.1);
        assert_eq!(d.policy_version, POLICY_VERSION);
        assert_eq!(d.fallback, RouteAction::Local);
        assert_eq!(d.mode, ParetoMode::Balance);
        assert!(!d.reason.is_empty());
    }

    #[test]
    fn outcome_store_records() {
        let r = Router::new(0.5).unwrap();
        let d = r.route_confidence(0.1);
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
        });
        assert_eq!(r.outcomes.len(), 1);
        let recent = r.outcomes.recent(1);
        assert_eq!(recent[0].task_id, "t1");
        assert!(recent[0].cloud_handoff);
    }

    #[test]
    fn high_complexity_prefers_cloud() {
        let r = Router::new(0.5).unwrap();
        let mut sig = RouteSignal::from_confidence(0.95);
        sig.complexity = 0.8;
        let d = r.route(&sig);
        assert_eq!(d.action, RouteAction::CloudHandoff);
        assert!(d.reason.contains("complexity"));
    }

    #[tokio::test]
    async fn mock_cloud_ok_and_fail() {
        let ok = CloudClient::from_env("http://127.0.0.1:9").with_mock(MockMode::Success(
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
            })
            .await
            .unwrap();
        assert!(v["choices"][0]["message"]["content"].is_string());

        let bad = CloudClient::from_env("http://127.0.0.1:9").with_mock(MockMode::FailStatus(503));
        let err = bad
            .chat(&CloudChatRequest {
                model: "x".into(),
                messages: vec![],
                max_tokens: None,
            })
            .await
            .unwrap_err();
        assert!(matches!(err, EngineError::Cloud(_)));

        let to = CloudClient::from_env("http://127.0.0.1:9").with_mock(MockMode::Timeout);
        let err = to
            .chat(&CloudChatRequest {
                model: "x".into(),
                messages: vec![],
                max_tokens: None,
            })
            .await
            .unwrap_err();
        assert!(matches!(err, EngineError::Cloud(_)));
        assert!(err.to_string().contains("timeout"));
    }

    #[test]
    fn invalid_threshold() {
        assert!(matches!(
            Router::new(-0.1),
            Err(EngineError::InvalidParam(_))
        ));
        assert!(matches!(
            Router::new(1.01),
            Err(EngineError::InvalidParam(_))
        ));
    }

    #[test]
    fn threshold_boundaries_ok() {
        assert!(Router::new(0.0).is_ok());
        assert!(Router::new(1.0).is_ok());
        let r0 = Router::new(0.0).unwrap();
        // Cost mode still 0; never handoff on confidence alone.
        assert_eq!(r0.route_confidence(0.0).action, RouteAction::Local);
        let r1 = Router::new(1.0)
            .unwrap()
            .with_mode(ParetoMode::Intelligence);
        // effective threshold clamped to 1.0 → conf < 1.0 handoff
        assert_eq!(
            r1.route_confidence(0.99).action,
            RouteAction::CloudHandoff
        );
    }

    #[test]
    fn force_cloud_and_modality_without_cloud() {
        let r = Router::new(0.5).unwrap();
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
        let mut r = Router::new(0.5).unwrap();
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
        let r = Router::new(0.5).unwrap();
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
        let r = Router::new(0.5).unwrap();
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
        let r = Router::new(0.5)
            .unwrap()
            .with_mode(ParetoMode::Intelligence);
        let d = r.route_confidence(0.1);
        let v = serde_json::to_value(&d).unwrap();
        let back: RouteDecision = serde_json::from_value(v).unwrap();
        assert_eq!(back.action, d.action);
        assert_eq!(back.mode, ParetoMode::Intelligence);
        assert_eq!(back.projection, d.projection);

        let outcome = RouteOutcome {
            task_id: "x".into(),
            session_id: Some("s".into()),
            action: RouteAction::CloudHandoff,
            reason: "low_confidence".into(),
            policy_version: POLICY_VERSION.into(),
            mode: ParetoMode::Balance,
            input_tokens: Some(1),
            output_tokens: Some(2),
            latency_ms: Some(3),
            cloud_handoff: true,
            user_corrected: Some(true),
            validation_ok: Some(false),
        };
        let v = serde_json::to_value(&outcome).unwrap();
        let back: RouteOutcome = serde_json::from_value(v).unwrap();
        assert_eq!(back, outcome);
    }

    #[test]
    fn cloud_client_availability_without_credentials() {
        let c = CloudClient::from_env("http://127.0.0.1:9");
        assert!(!c.is_available());
    }

    #[tokio::test]
    async fn cloud_client_missing_api_key() {
        let c = CloudClient::from_env("http://127.0.0.1:9");
        let err = c
            .chat(&CloudChatRequest {
                model: "x".into(),
                messages: vec![],
                max_tokens: None,
            })
            .await
            .unwrap_err();
        assert!(matches!(err, EngineError::Cloud(_)));
        assert!(err.to_string().contains("ARIA_HYBRID_CLOUD_API_KEY"));
    }
}
