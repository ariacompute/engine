//! Hybrid router: local inference vs cloud handoff by confidence.

use aria_kernel::EngineError;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RouteDecision {
    Local,
    CloudHandoff,
}

#[derive(Debug, Clone)]
pub struct Router {
    pub threshold: f32,
    pub on_device_only: bool,
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
            on_device_only: false,
        })
    }

    pub fn route(&self, confidence: f32) -> RouteDecision {
        if self.on_device_only {
            return RouteDecision::Local;
        }
        if confidence < self.threshold {
            RouteDecision::CloudHandoff
        } else {
            RouteDecision::Local
        }
    }
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
    fn router_threshold() {
        let r = Router::new(0.5).unwrap();
        assert_eq!(r.route(0.9), RouteDecision::Local);
        assert_eq!(r.route(0.1), RouteDecision::CloudHandoff);
        let mut r2 = Router::new(0.5).unwrap();
        r2.on_device_only = true;
        assert_eq!(r2.route(0.0), RouteDecision::Local);
        assert!(Router::new(1.5).is_err());
    }

    #[tokio::test]
    async fn mock_cloud_ok_and_fail() {
        let ok = CloudClient::from_env("http://127.0.0.1:9").with_mock(MockMode::Success(
            json!({"choices":[{"message":{"content":"hi"}}]}),
        ));
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
}
