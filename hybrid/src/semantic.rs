//! Semantic routing layer (slow path, P2).
//!
//! Consulted only when the rule layer is undecided. The production client
//! reuses [`CloudClient`] to ask the cloud gateway for a structured JSON
//! intent classification (`{"action","confidence","intent","reason"}`).
//! Decisions are cached (normalized-prompt hash key + TTL + capacity cap) and
//! deduplicated in-flight (singleflight). Disabled / no credentials / timeout
//! / parse failure → `None`, silently degrading to the rule layer. Never
//! panics; errors are surfaced only as `None` at this boundary.

use crate::route::{RouteAction, RouteSignal};
use crate::{CloudChatRequest, CloudClient, CloudMessage, CLOUD_GATEWAY_MODEL};
use aria_kernel::EngineError;
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Minimum semantic confidence for the router to adopt a slow-path decision.
pub const SEMANTIC_ACCEPT_THRESHOLD: f32 = 0.6;
/// Default per-consult timeout.
pub const DEFAULT_SEMANTIC_TIMEOUT_MS: u64 = 800;
/// Default decision-cache capacity.
pub const DEFAULT_SEMANTIC_CACHE_SIZE: usize = 512;
/// Decision-cache TTL.
pub const SEMANTIC_CACHE_TTL: Duration = Duration::from_secs(60);
/// Prompt characters considered for the cache key / cloud request.
const PROMPT_KEY_CHARS: usize = 256;

const SEMANTIC_SYSTEM_PROMPT: &str = "You are a routing classifier for a hybrid local/cloud inference engine. \
Decide whether the user request should run on the small on-device model (\"local\") or be handed off to the \
cloud flagship model (\"cloud\"). Do not think step by step. Answer with STRICT JSON only, no markdown, no prose: \
{\"action\":\"local\"|\"cloud\",\"confidence\":0.0-1.0,\"intent\":\"short intent label\",\"reason\":\"short reason\"}.";

/// Structured slow-path routing decision.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticDecision {
    pub action: RouteAction,
    pub confidence: f32,
    pub intent: String,
    pub reason: String,
}

/// Test-injection point for the semantic layer.
pub trait FakeSemanticClient: Send + Sync + fmt::Debug {
    fn consult<'a>(
        &'a self,
        signal: &'a RouteSignal,
        prompt: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<SemanticDecision, EngineError>> + Send + 'a>>;
}

/// Production semantic client: reuses the cloud gateway chat client.
#[derive(Debug, Clone)]
pub struct CloudSemanticClient {
    cloud: CloudClient,
}

impl CloudSemanticClient {
    pub fn new(cloud: CloudClient) -> Self {
        Self { cloud }
    }

    pub fn from_client(cloud: &CloudClient) -> Self {
        Self {
            cloud: cloud.clone(),
        }
    }

    pub fn is_available(&self) -> bool {
        self.cloud.is_available()
    }

    async fn consult(
        &self,
        signal: &RouteSignal,
        prompt: &str,
    ) -> Result<SemanticDecision, EngineError> {
        let req = CloudChatRequest {
            model: CLOUD_GATEWAY_MODEL.to_string(),
            messages: vec![
                CloudMessage {
                    role: "system".into(),
                    content: SEMANTIC_SYSTEM_PROMPT.into(),
                },
                CloudMessage {
                    role: "user".into(),
                    content: semantic_user_prompt(signal, prompt),
                },
            ],
            max_tokens: Some(128),
            enable_thinking: Some(false),
        };
        let v = self.cloud.chat(&req).await?;
        let msg = &v["choices"][0]["message"];
        let content = msg["content"]
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .or_else(|| msg["reasoning"].as_str())
            .ok_or_else(|| EngineError::Cloud("semantic response missing content".into()))?;
        parse_semantic_decision(content)
    }
}

/// Semantic client dispatch (production cloud or injected fake).
#[derive(Debug, Clone)]
pub enum SemanticClient {
    Cloud(CloudSemanticClient),
    Fake(Arc<dyn FakeSemanticClient>),
}

impl SemanticClient {
    pub fn is_available(&self) -> bool {
        match self {
            Self::Cloud(c) => c.is_available(),
            Self::Fake(_) => true,
        }
    }

    async fn consult(
        &self,
        signal: &RouteSignal,
        prompt: &str,
    ) -> Result<SemanticDecision, EngineError> {
        match self {
            Self::Cloud(c) => c.consult(signal, prompt).await,
            Self::Fake(f) => f.consult(signal, prompt).await,
        }
    }
}

/// Normalize a prompt for cache-keying: trim, collapse whitespace, lowercase, truncate.
fn normalize_prompt(prompt: &str) -> String {
    let collapsed: String = prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed
        .to_lowercase()
        .chars()
        .take(PROMPT_KEY_CHARS)
        .collect()
}

/// FNV-1a over the normalized prompt + coarse signal buckets.
fn cache_key(signal: &RouteSignal, prompt: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    let mut mix = |byte: u8| {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    };
    for b in normalize_prompt(prompt).as_bytes() {
        mix(*b);
    }
    // Coarse buckets keep near-identical requests from fragmenting the cache.
    mix((signal.complexity * 10.0) as u8);
    mix((signal.context_tokens / 512) as u8);
    mix(signal.context_limit.wrapping_div(4096) as u8);
    hash
}

/// Parse + validate the strict JSON decision from the cloud response.
fn parse_semantic_decision(content: &str) -> Result<SemanticDecision, EngineError> {
    let mut raw = content.trim();
    if let Some(rest) = raw.strip_prefix("```") {
        raw = rest.strip_prefix("json").unwrap_or(rest);
        raw = raw.strip_suffix("```").unwrap_or(raw).trim();
    }
    if let (Some(start), Some(end)) = (raw.find('{'), raw.rfind('}')) {
        if end > start {
            raw = &raw[start..=end];
        }
    }
    let v: Value = serde_json::from_str(raw)
        .map_err(|e| EngineError::Cloud(format!("invalid semantic JSON: {e}")))?;
    let action = match v["action"].as_str() {
        Some("local") => RouteAction::Local,
        Some("cloud") | Some("cloud_handoff") => RouteAction::CloudHandoff,
        other => {
            return Err(EngineError::Cloud(format!(
                "invalid semantic action: {other:?}"
            )))
        }
    };
    let confidence = v["confidence"]
        .as_f64()
        .filter(|c| (0.0..=1.0).contains(c))
        .ok_or_else(|| EngineError::Cloud("invalid semantic confidence".into()))?
        as f32;
    let intent = v["intent"].as_str().unwrap_or("unknown").to_string();
    let reason = v["reason"].as_str().unwrap_or("").to_string();
    Ok(SemanticDecision {
        action,
        confidence,
        intent,
        reason,
    })
}

fn semantic_user_prompt(signal: &RouteSignal, prompt: &str) -> String {
    format!(
        "signals: complexity={:.2} context_tokens={} context_limit={}\nrequest:\n{}",
        signal.complexity,
        signal.context_tokens,
        signal.context_limit,
        normalize_prompt(prompt)
    )
}

#[derive(Debug, Default)]
struct SemanticCache {
    entries: HashMap<u64, (Instant, SemanticDecision)>,
    capacity: usize,
    ttl: Duration,
}

impl SemanticCache {
    fn get(&mut self, key: u64) -> Option<SemanticDecision> {
        match self.entries.get(&key) {
            Some((ts, d)) if ts.elapsed() <= self.ttl => Some(d.clone()),
            Some(_) => {
                self.entries.remove(&key);
                None
            }
            None => None,
        }
    }

    fn insert(&mut self, key: u64, decision: SemanticDecision) {
        if self.capacity == 0 {
            return;
        }
        if self.entries.len() >= self.capacity && !self.entries.contains_key(&key) {
            // Evict the oldest entry (capacity is small; linear scan is fine).
            if let Some(oldest) = self
                .entries
                .iter()
                .max_by_key(|(_, (ts, _))| ts.elapsed())
                .map(|(k, _)| *k)
            {
                self.entries.remove(&oldest);
            }
        }
        self.entries.insert(key, (Instant::now(), decision));
    }
}

/// Slow-path semantic router with cache + singleflight + timeout.
#[derive(Debug, Clone)]
pub struct SemanticRouter {
    enabled: bool,
    timeout: Duration,
    client: Option<SemanticClient>,
    cache: Arc<Mutex<SemanticCache>>,
    inflight: Arc<Mutex<HashMap<u64, Arc<tokio::sync::Mutex<()>>>>>,
}

impl SemanticRouter {
    pub fn new(client: Option<SemanticClient>, timeout_ms: u64, cache_size: usize) -> Self {
        Self {
            enabled: true,
            timeout: Duration::from_millis(timeout_ms.max(1)),
            client,
            cache: Arc::new(Mutex::new(SemanticCache {
                entries: HashMap::new(),
                capacity: cache_size,
                ttl: SEMANTIC_CACHE_TTL,
            })),
            inflight: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Disabled router: `consult` always returns `None` (pure rule path).
    pub fn disabled() -> Self {
        Self::new(
            None,
            DEFAULT_SEMANTIC_TIMEOUT_MS,
            DEFAULT_SEMANTIC_CACHE_SIZE,
        )
        .with_enabled(false)
    }

    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
            && self
                .client
                .as_ref()
                .is_some_and(SemanticClient::is_available)
    }

    pub fn cache_len(&self) -> usize {
        self.cache
            .lock()
            .expect("semantic cache poisoned")
            .entries
            .len()
    }

    /// Consult the semantic layer. Any failure / disabled / timeout → `None`.
    pub async fn consult(&self, signal: &RouteSignal, prompt: &str) -> Option<SemanticDecision> {
        if !self.enabled {
            return None;
        }
        let client = self.client.clone()?;
        if !client.is_available() {
            return None;
        }
        let key = cache_key(signal, prompt);
        if let Some(d) = self.cache.lock().expect("semantic cache poisoned").get(key) {
            return Some(d);
        }

        // Singleflight: one in-flight cloud call per cache key.
        let guard = {
            let mut m = self.inflight.lock().expect("semantic inflight poisoned");
            m.entry(key)
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                .clone()
        };
        let _permit = guard.lock().await;
        // Re-check after winning the permit (another task may have filled it).
        if let Some(d) = self.cache.lock().expect("semantic cache poisoned").get(key) {
            self.inflight
                .lock()
                .expect("semantic inflight poisoned")
                .remove(&key);
            return Some(d);
        }

        let result = match tokio::time::timeout(self.timeout, client.consult(signal, prompt)).await
        {
            Ok(Ok(decision)) => Some(decision),
            Ok(Err(_)) | Err(_) => None,
        };
        if let Some(d) = &result {
            self.cache
                .lock()
                .expect("semantic cache poisoned")
                .insert(key, d.clone());
        }
        self.inflight
            .lock()
            .expect("semantic inflight poisoned")
            .remove(&key);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MockMode;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn sig() -> RouteSignal {
        RouteSignal::from_confidence(0.95)
    }

    fn decision(action: RouteAction, confidence: f32) -> SemanticDecision {
        SemanticDecision {
            action,
            confidence,
            intent: "test".into(),
            reason: "fake".into(),
        }
    }

    #[derive(Debug)]
    struct StaticFake {
        result: Result<SemanticDecision, ()>,
        calls: Arc<AtomicUsize>,
        sleep_ms: u64,
    }

    impl FakeSemanticClient for StaticFake {
        fn consult<'a>(
            &'a self,
            _signal: &'a RouteSignal,
            _prompt: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<SemanticDecision, EngineError>> + Send + 'a>>
        {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                if self.sleep_ms > 0 {
                    tokio::time::sleep(Duration::from_millis(self.sleep_ms)).await;
                }
                match &self.result {
                    Ok(d) => Ok(d.clone()),
                    Err(()) => Err(EngineError::Cloud("fake error".into())),
                }
            })
        }
    }

    fn fake_router(
        result: Result<SemanticDecision, ()>,
        calls: Arc<AtomicUsize>,
        sleep_ms: u64,
    ) -> SemanticRouter {
        SemanticRouter::new(
            Some(SemanticClient::Fake(Arc::new(StaticFake {
                result,
                calls,
                sleep_ms,
            }))),
            100,
            8,
        )
    }

    #[test]
    fn parse_valid_and_fenced_json() {
        let d = parse_semantic_decision(
            r#"{"action":"cloud","confidence":0.9,"intent":"reasoning","reason":"hard"}"#,
        )
        .unwrap();
        assert_eq!(d.action, RouteAction::CloudHandoff);
        assert!((d.confidence - 0.9).abs() < 1e-6);
        assert_eq!(d.intent, "reasoning");

        let fenced = "```json\n{\"action\":\"local\",\"confidence\":0.7}\n```";
        let d = parse_semantic_decision(fenced).unwrap();
        assert_eq!(d.action, RouteAction::Local);
        assert_eq!(d.intent, "unknown");

        let mixed = "noise {\"action\":\"cloud\",\"confidence\":0.8} trailing";
        let d = parse_semantic_decision(mixed).unwrap();
        assert_eq!(d.action, RouteAction::CloudHandoff);
    }

    #[test]
    fn parse_rejects_invalid_payloads() {
        assert!(parse_semantic_decision("not json").is_err());
        assert!(parse_semantic_decision(r#"{"action":"maybe","confidence":0.5}"#).is_err());
        assert!(parse_semantic_decision(r#"{"action":"local","confidence":1.5}"#).is_err());
        assert!(parse_semantic_decision(r#"{"action":"local"}"#).is_err());
        assert!(parse_semantic_decision(r#"{"confidence":0.5}"#).is_err());
    }

    #[tokio::test]
    async fn disabled_or_unavailable_returns_none() {
        let calls = Arc::new(AtomicUsize::new(0));
        let r = fake_router(Ok(decision(RouteAction::Local, 0.9)), calls.clone(), 0)
            .with_enabled(false);
        assert!(r.consult(&sig(), "hi").await.is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        let r = SemanticRouter::new(None, 100, 8);
        assert!(!r.is_enabled());
        assert!(r.consult(&sig(), "hi").await.is_none());

        // Cloud client without credentials → unavailable short-circuit.
        let r = SemanticRouter::new(
            Some(SemanticClient::Cloud(CloudSemanticClient::new(
                CloudClient::new("http://127.0.0.1:9", ""),
            ))),
            100,
            8,
        );
        assert!(!r.is_enabled());
        assert!(r.consult(&sig(), "hi").await.is_none());
    }

    #[tokio::test]
    async fn consult_success_and_cache_hit() {
        let calls = Arc::new(AtomicUsize::new(0));
        let r = fake_router(
            Ok(decision(RouteAction::CloudHandoff, 0.9)),
            calls.clone(),
            0,
        );
        let d = r.consult(&sig(), "hello").await.unwrap();
        assert_eq!(d.action, RouteAction::CloudHandoff);
        let d2 = r.consult(&sig(), "hello").await.unwrap();
        assert_eq!(d2, d);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(r.cache_len(), 1);
    }

    #[tokio::test]
    async fn cache_ttl_expiry_refetches() {
        let calls = Arc::new(AtomicUsize::new(0));
        let r = fake_router(Ok(decision(RouteAction::Local, 0.9)), calls.clone(), 0);
        r.cache.lock().unwrap().ttl = Duration::from_millis(0);
        r.consult(&sig(), "hello").await.unwrap();
        r.consult(&sig(), "hello").await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn cache_capacity_evicts_oldest() {
        let calls = Arc::new(AtomicUsize::new(0));
        let r = SemanticRouter::new(
            Some(SemanticClient::Fake(Arc::new(StaticFake {
                result: Ok(decision(RouteAction::Local, 0.9)),
                calls: calls.clone(),
                sleep_ms: 0,
            }))),
            100,
            1,
        );
        r.consult(&sig(), "prompt-a").await.unwrap();
        r.consult(&sig(), "prompt-b").await.unwrap();
        assert_eq!(r.cache_len(), 1);
        // prompt-a was evicted → refetch.
        r.consult(&sig(), "prompt-a").await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn singleflight_deduplicates_inflight() {
        let calls = Arc::new(AtomicUsize::new(0));
        let r = fake_router(Ok(decision(RouteAction::Local, 0.9)), calls.clone(), 50);
        let s = sig();
        let (a, b) = tokio::join!(r.consult(&s, "same"), r.consult(&s, "same"));
        assert!(a.is_some());
        assert!(b.is_some());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn timeout_and_error_degrade_to_none() {
        let calls = Arc::new(AtomicUsize::new(0));
        let r = fake_router(Ok(decision(RouteAction::Local, 0.9)), calls.clone(), 500);
        assert!(r.consult(&sig(), "slow").await.is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(r.cache_len(), 0);

        let calls = Arc::new(AtomicUsize::new(0));
        let r = fake_router(Err(()), calls.clone(), 0);
        assert!(r.consult(&sig(), "err").await.is_none());
        assert_eq!(r.cache_len(), 0);
    }

    #[tokio::test]
    async fn cloud_semantic_client_parses_gateway_json() {
        let cloud = CloudClient::new("http://127.0.0.1:9", "").with_mock(MockMode::Success(
            json!({"choices":[{"message":{"content":"{\"action\":\"cloud\",\"confidence\":0.88,\"intent\":\"agent\",\"reason\":\"multi-step\"}"}}]}),
        ));
        let c = CloudSemanticClient::new(cloud);
        assert!(c.is_available());
        let d = c.consult(&sig(), "refactor this").await.unwrap();
        assert_eq!(d.action, RouteAction::CloudHandoff);
        assert!((d.confidence - 0.88).abs() < 1e-6);
        assert_eq!(d.intent, "agent");

        // Non-JSON content → error → router degrades to None.
        let cloud = CloudClient::new("http://127.0.0.1:9", "").with_mock(MockMode::Success(
            json!({"choices":[{"message":{"content":"plain chat answer"}}]}),
        ));
        let r = SemanticRouter::new(
            Some(SemanticClient::Cloud(CloudSemanticClient::new(cloud))),
            100,
            8,
        );
        assert!(r.consult(&sig(), "hi").await.is_none());

        // Gateway failure → None.
        let cloud = CloudClient::new("http://127.0.0.1:9", "").with_mock(MockMode::FailStatus(503));
        let r = SemanticRouter::new(
            Some(SemanticClient::Cloud(CloudSemanticClient::new(cloud))),
            100,
            8,
        );
        assert!(r.consult(&sig(), "hi").await.is_none());
    }

    #[test]
    fn cache_key_normalizes_prompt() {
        let s = sig();
        assert_eq!(
            cache_key(&s, "  Hello   world "),
            cache_key(&s, "hello world")
        );
        let long = "x".repeat(400);
        assert_eq!(cache_key(&s, &long), cache_key(&s, &long[..256]));
        assert_ne!(cache_key(&s, "alpha"), cache_key(&s, "beta"));
    }
}
