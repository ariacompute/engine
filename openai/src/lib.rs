//! OpenAI-compatible HTTP surface (chat / embeddings / ASR / tools / RAG).

pub mod config;
pub mod download;
pub mod gateway_detect;
pub mod upgrade;

use aria_hybrid::{
    estimate_route_signals, CloudChatRequest, CloudClient, CloudMessage, ExecutionMode, RouteAction,
    RouteOutcome, RouteSignal, Router, CLOUD_GATEWAY_MODEL,
};
use aria_inference::{rag_pack_context, GenerateOpts, Session, SessionBuilder};
use aria_kernel::EngineError;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router as AxumRouter};
use futures_util::stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone)]
pub struct AppState {
    pub session: Arc<Mutex<Session>>,
    pub router: Router,
    pub cloud: CloudClient,
    pub model_id: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatTool {
    #[serde(rename = "type")]
    pub type_: Option<String>,
    pub function: Option<ChatToolFunction>,
}

#[derive(Debug, Deserialize)]
pub struct ChatToolFunction {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<ChatTool>>,
    /// Optional RAG snippets (stage C).
    #[serde(default)]
    pub rag_snippets: Option<Vec<String>>,
    /// Optional hybrid session id for sticky Local/Cloud routing.
    #[serde(default)]
    pub session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub input: EmbeddingInput,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Text(String),
    Batch(Vec<String>),
}

#[derive(Debug, Deserialize)]
pub struct TranscriptionRequest {
    /// Base64-encoded PCM16 LE mono (stage C stub; real multipart later).
    pub file_b64: String,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
}

pub fn app(state: AppState) -> AxumRouter {
    AxumRouter::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/audio/transcriptions", post(audio_transcriptions))
        .route("/v1/embeddings", post(embeddings))
        .with_state(state)
}

async fn list_models(State(st): State<AppState>) -> Json<Value> {
    let ids = advertised_model_ids(&st);
    let data: Vec<Value> = ids
        .into_iter()
        .map(|id| {
            json!({
                "id": id,
                "object": "model",
                "owned_by": "aria"
            })
        })
        .collect();
    Json(json!({
        "object": "list",
        "data": data
    }))
}

/// Model ids exposed on `GET /v1/models`.
/// - `cloud`: only gateway id `ariacompute/ariamodel`
/// - `device`: local bundle id
/// - `hybrid`: local + gateway id when cloud credentials are configured
fn advertised_model_ids(st: &AppState) -> Vec<String> {
    match st.router.execution {
        ExecutionMode::Cloud => vec![CLOUD_GATEWAY_MODEL.to_string()],
        ExecutionMode::Device => vec![st.model_id.clone()],
        ExecutionMode::Hybrid => {
            let mut ids = vec![st.model_id.clone()];
            if st.cloud.is_available() {
                let cloud = CLOUD_GATEWAY_MODEL.to_string();
                if !ids.iter().any(|id| id == &cloud) {
                    ids.push(cloud);
                }
            }
            ids
        }
    }
}

async fn chat_completions(
    State(st): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    match handle_chat(&st, req).await {
        Ok(r) => r,
        Err(e) => engine_err(e),
    }
}

async fn embeddings(State(st): State<AppState>, Json(req): Json<EmbeddingRequest>) -> Response {
    match handle_embeddings(&st, req) {
        Ok(v) => Json(v).into_response(),
        Err(e) => engine_err(e),
    }
}

async fn audio_transcriptions(
    State(st): State<AppState>,
    Json(req): Json<TranscriptionRequest>,
) -> Response {
    match handle_transcription(&st, req) {
        Ok(v) => Json(v).into_response(),
        Err(e) => engine_err(e),
    }
}

fn handle_embeddings(st: &AppState, req: EmbeddingRequest) -> Result<Value, EngineError> {
    let texts = match req.input {
        EmbeddingInput::Text(s) => vec![s],
        EmbeddingInput::Batch(v) => v,
    };
    let sess = st
        .session
        .lock()
        .map_err(|e| EngineError::Io(e.to_string()))?;
    let mut data = Vec::new();
    for (i, t) in texts.iter().enumerate() {
        let emb = sess.embed_text(t)?;
        data.push(json!({
            "object": "embedding",
            "index": i,
            "embedding": emb
        }));
    }
    Ok(json!({
        "object": "list",
        "data": data,
        "model": req.model.unwrap_or_else(|| st.model_id.clone()),
        "usage": { "prompt_tokens": 0, "total_tokens": 0 }
    }))
}

fn handle_transcription(st: &AppState, req: TranscriptionRequest) -> Result<Value, EngineError> {
    let pcm = decode_b64(&req.file_b64)?;
    let sess = st
        .session
        .lock()
        .map_err(|e| EngineError::Io(e.to_string()))?;
    let text = sess.transcribe_pcm16le(&pcm)?;
    Ok(json!({
        "text": text,
        "model": req.model.unwrap_or_else(|| st.model_id.clone())
    }))
}

fn decode_b64(s: &str) -> Result<Vec<u8>, EngineError> {
    // Minimal base64 decode without extra dep (std-only alphabet).
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            b'=' => Some(0),
            _ => None,
        }
    }
    let bytes = s.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err(EngineError::Format("invalid base64 length".into()));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let (a, b, c, d) = (
            val(chunk[0]).ok_or_else(|| EngineError::Format("b64".into()))?,
            val(chunk[1]).ok_or_else(|| EngineError::Format("b64".into()))?,
            val(chunk[2]).ok_or_else(|| EngineError::Format("b64".into()))?,
            val(chunk[3]).ok_or_else(|| EngineError::Format("b64".into()))?,
        );
        out.push((a << 2) | (b >> 4));
        if chunk[2] != b'=' {
            out.push((b << 4) | (c >> 2));
        }
        if chunk[3] != b'=' {
            out.push((c << 6) | d);
        }
    }
    Ok(out)
}

async fn handle_chat(st: &AppState, req: ChatCompletionRequest) -> Result<Response, EngineError> {
    let max_tokens = req.max_tokens.unwrap_or(16) as usize;
    if max_tokens == 0 {
        return Err(EngineError::InvalidParam("max_tokens must be > 0".into()));
    }
    let user_text = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.as_str())
        .unwrap_or("");

    let prompt_text = if let Some(snips) = &req.rag_snippets {
        if !snips.is_empty() {
            rag_pack_context(snips, user_text)
        } else {
            user_text.to_string()
        }
    } else {
        user_text.to_string()
    };

    // Tool-call short-circuit: if tools present and user asks to call, return tool_calls JSON.
    if let Some(tools) = &req.tools {
        if !tools.is_empty() && user_text.contains("CALL_TOOL") {
            let name = tools
                .iter()
                .find_map(|t| t.function.as_ref().map(|f| f.name.clone()))
                .unwrap_or_else(|| "unknown".into());
            let created = now_secs();
            return Ok(Json(json!({
                "id": format!("chatcmpl-{created}"),
                "object": "chat.completion",
                "created": created,
                "model": st.model_id,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": format!("call_{created}"),
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": "{\"query\":\"CALL_TOOL\"}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            }))
            .into_response());
        }
    }

    let force_cloud = prompt_text.contains("FORCE_CLOUD");
    let context_limit = {
        let sess = st
            .session
            .lock()
            .map_err(|e| EngineError::Io(e.to_string()))?;
        sess.config().context_length as u32
    };
    let (complexity, context_tokens) = estimate_route_signals(&prompt_text, context_limit);
    let signal = RouteSignal {
        confidence: 0.95,
        complexity,
        context_tokens,
        context_limit,
        modality_unsupported_locally: false,
        consecutive_local_failures: 0,
        privacy_sensitive: false,
        cloud_available: st.cloud.is_available(),
        session_id: req.session_id.clone(),
        force_cloud,
    };
    let decision = st.router.route(&signal);
    let started = std::time::Instant::now();

    match decision.action {
        RouteAction::CloudHandoff => {
            let cloud_req = CloudChatRequest {
                model: CLOUD_GATEWAY_MODEL.to_string(),
                messages: req
                    .messages
                    .iter()
                    .map(|m| CloudMessage {
                        role: m.role.clone(),
                        content: m.content.clone(),
                    })
                    .collect(),
                max_tokens: Some(max_tokens as u32),
            };
            let v = st.cloud.chat(&cloud_req).await?;
            st.router.record_outcome(RouteOutcome {
                task_id: format!("chat-{}", now_secs()),
                session_id: signal.session_id.clone(),
                action: decision.action,
                reason: decision.reason.clone(),
                policy_version: decision.policy_version.clone(),
                mode: decision.mode,
                input_tokens: None,
                output_tokens: None,
                latency_ms: Some(started.elapsed().as_millis() as u64),
                cloud_handoff: true,
                user_corrected: None,
                validation_ok: None,
            });
            Ok(Json(v).into_response())
        }
        RouteAction::Local => {
            let mut sess = st
                .session
                .lock()
                .map_err(|e| EngineError::Io(e.to_string()))?;
            let prompt = sess.encode_text(&prompt_text);
            let gen = sess.generate(
                &prompt,
                &GenerateOpts {
                    max_tokens,
                    temperature: req.temperature.unwrap_or(0.0),
                },
            )?;
            st.router.record_outcome(RouteOutcome {
                task_id: format!("chat-{}", now_secs()),
                session_id: signal.session_id.clone(),
                action: decision.action,
                reason: decision.reason.clone(),
                policy_version: decision.policy_version.clone(),
                mode: decision.mode,
                input_tokens: Some(prompt.len() as u32),
                output_tokens: Some(gen.tokens.len() as u32),
                latency_ms: Some(started.elapsed().as_millis() as u64),
                cloud_handoff: false,
                user_corrected: None,
                validation_ok: None,
            });
            let created = now_secs();
            if req.stream.unwrap_or(false) {
                let content = gen.text.clone();
                let model = st.model_id.clone();
                let chunk = json!({
                    "id": format!("chatcmpl-{created}"),
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": { "role": "assistant", "content": content },
                        "finish_reason": null
                    }]
                });
                let done = json!({
                    "id": format!("chatcmpl-{created}"),
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                });
                let events = stream::iter(vec![
                    Ok::<_, Infallible>(Event::default().data(chunk.to_string())),
                    Ok(Event::default().data(done.to_string())),
                    Ok(Event::default().data("[DONE]")),
                ]);
                Ok(Sse::new(events)
                    .keep_alive(KeepAlive::default())
                    .into_response())
            } else {
                Ok(Json(json!({
                    "id": format!("chatcmpl-{created}"),
                    "object": "chat.completion",
                    "created": created,
                    "model": st.model_id,
                    "choices": [{
                        "index": 0,
                        "message": { "role": "assistant", "content": gen.text },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": prompt.len(),
                        "completion_tokens": gen.tokens.len(),
                        "total_tokens": prompt.len() + gen.tokens.len()
                    }
                }))
                .into_response())
            }
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn engine_err(e: EngineError) -> Response {
    let (code, ty) = match &e {
        EngineError::InvalidParam(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
        EngineError::Unsupported(_) | EngineError::UnsupportedFamily(_) => {
            (StatusCode::NOT_IMPLEMENTED, "unsupported_error")
        }
        EngineError::Cloud(_) => (StatusCode::BAD_GATEWAY, "cloud_error"),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
    };
    let body = Json(ErrorBody {
        error: ErrorDetail {
            message: e.to_string(),
            r#type: ty.into(),
        },
    });
    (code, body).into_response()
}

pub fn build_state(
    model_dir: impl AsRef<std::path::Path>,
    router: Router,
    cloud: CloudClient,
) -> Result<AppState, EngineError> {
    build_state_with_family(model_dir, "gemma/gemma-4-e2b-it", router, cloud)
}

pub fn build_state_with_family(
    model_dir: impl AsRef<std::path::Path>,
    family: &str,
    router: Router,
    cloud: CloudClient,
) -> Result<AppState, EngineError> {
    let model_dir = model_dir.as_ref();
    let session = SessionBuilder::new()
        .model(model_dir)
        .family(family)
        .build()?;
    // Local OpenAI model id = bundle directory name (e.g. qwen3-0.6b_q4), not the
    // internal family path used for graph wiring (often still stage-A gemma).
    let local_id = model_dir
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| session.model_id().to_string());
    // Cloud-only execution advertises the gateway model id (ariacompute/ariamodel).
    let model_id = if router.execution == ExecutionMode::Cloud {
        CLOUD_GATEWAY_MODEL.to_string()
    } else {
        local_id
    };
    Ok(AppState {
        session: Arc::new(Mutex::new(session)),
        router,
        cloud,
        model_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aria_hybrid::{MockMode, ParetoMode, POLICY_VERSION};
    use aria_inference::fixture::write_tiny_q4_bundle;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    async fn body_json(res: Response) -> Value {
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn b64_encode(data: &[u8]) -> String {
        const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in data.chunks(3) {
            let mut n = (chunk[0] as u32) << 16;
            if chunk.len() > 1 {
                n |= (chunk[1] as u32) << 8;
            }
            if chunk.len() > 2 {
                n |= chunk[2] as u32;
            }
            out.push(T[((n >> 18) & 63) as usize] as char);
            out.push(T[((n >> 12) & 63) as usize] as char);
            if chunk.len() > 1 {
                out.push(T[((n >> 6) & 63) as usize] as char);
            } else {
                out.push('=');
            }
            if chunk.len() > 2 {
                out.push(T[(n & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
        out
    }

    #[tokio::test]
    async fn chat_and_models_and_sse() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let state = build_state(
            dir.path(),
            Router::new().unwrap(),
            CloudClient::new("http://127.0.0.1:9", "").with_mock(MockMode::Success(json!({
                "choices":[{"message":{"content":"cloud"}}]
            }))),
        )
        .unwrap();
        let app = app(state);

        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages":[{"role":"user","content":"hi"}],
                            "max_tokens": 2
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages":[{"role":"user","content":"hi"}],
                            "max_tokens": 2,
                            "stream": true
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.contains("data:"));
    }

    #[tokio::test]
    async fn embeddings_asr_tools_rag() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let state = build_state(
            dir.path(),
            Router::new().unwrap(),
            CloudClient::new("http://127.0.0.1:9", "").with_mock(MockMode::Success(json!({}))),
        )
        .unwrap();
        let app = app(state);

        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/embeddings")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({"input":"hello embedding"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert!(!v["data"][0]["embedding"].as_array().unwrap().is_empty());

        let pcm = [0u8, 1, 2, 3, 4, 5, 6, 7];
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audio/transcriptions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({"file_b64": b64_encode(&pcm)}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert!(!v["text"].as_str().unwrap().is_empty());

        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages":[{"role":"user","content":"please CALL_TOOL now"}],
                            "tools":[{"type":"function","function":{"name":"search"}}],
                            "max_tokens": 2
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "search"
        );

        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages":[{"role":"user","content":"q"}],
                            "rag_snippets":["fact one","fact two"],
                            "max_tokens": 2
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn hybrid_cloud_and_execution_modes() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let state = build_state(
            dir.path(),
            Router::new().unwrap(),
            CloudClient::new("http://127.0.0.1:9", "").with_mock(MockMode::Success(json!({
                "choices":[{"message":{"content":"from-cloud"}}]
            }))),
        )
        .unwrap();
        let svc = app(state);
        let res = svc
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages":[{"role":"user","content":"FORCE_CLOUD please"}],
                            "max_tokens": 2
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert_eq!(v["choices"][0]["message"]["content"], "from-cloud");

        // device: FORCE_CLOUD still stays local
        let router = Router::new()
            .unwrap()
            .with_execution(aria_hybrid::ExecutionMode::Device);
        let state = build_state(
            dir.path(),
            router,
            CloudClient::new("http://127.0.0.1:9", "").with_mock(MockMode::Success(json!({
                "choices":[{"message":{"content":"from-cloud"}}]
            }))),
        )
        .unwrap();
        let svc = app(state);
        let res = svc
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages":[{"role":"user","content":"FORCE_CLOUD please"}],
                            "max_tokens": 2
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert_ne!(v["choices"][0]["message"]["content"], "from-cloud");

        // cloud: plain prompt (no FORCE_CLOUD) still handoffs
        let router = Router::new()
            .unwrap()
            .with_execution(aria_hybrid::ExecutionMode::Cloud);
        let state = build_state(
            dir.path(),
            router,
            CloudClient::new("http://127.0.0.1:9", "").with_mock(MockMode::Success(json!({
                "choices":[{"message":{"content":"from-cloud"}}]
            }))),
        )
        .unwrap();
        assert_eq!(state.model_id, CLOUD_GATEWAY_MODEL);
        let svc = app(state);
        let res = svc
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let models = body_json(res).await;
        assert_eq!(models["data"].as_array().unwrap().len(), 1);
        assert_eq!(models["data"][0]["id"], CLOUD_GATEWAY_MODEL);

        let res = svc
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages":[{"role":"user","content":"hello locally please"}],
                            "max_tokens": 2
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert_eq!(v["choices"][0]["message"]["content"], "from-cloud");
    }

    #[tokio::test]
    async fn models_lists_cloud_gateway_id_not_family_path() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let local_name = dir.path().file_name().unwrap().to_str().unwrap().to_string();

        // cloud execution → only ariacompute/ariamodel (never gemma family path)
        let cloud_state = build_state(
            dir.path(),
            Router::new()
                .unwrap()
                .with_execution(ExecutionMode::Cloud),
            CloudClient::new("http://127.0.0.1:9", "sk-test").with_mock(MockMode::Success(
                json!({"choices":[{"message":{"content":"c"}}]}),
            )),
        )
        .unwrap();
        assert_eq!(cloud_state.model_id, CLOUD_GATEWAY_MODEL);
        let models = body_json(
            app(cloud_state)
                .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(models["data"].as_array().unwrap().len(), 1);
        assert_eq!(models["data"][0]["id"], CLOUD_GATEWAY_MODEL);
        assert_ne!(models["data"][0]["id"], "gemma/gemma-4-e2b-it");

        // hybrid → local bundle dir name + gateway id
        let hybrid_state = build_state(
            dir.path(),
            Router::new().unwrap(),
            CloudClient::new("http://127.0.0.1:9", "sk-test"),
        )
        .unwrap();
        assert_eq!(hybrid_state.model_id, local_name);
        let models = body_json(
            app(hybrid_state)
                .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
                .await
                .unwrap(),
        )
        .await;
        let ids: Vec<&str> = models["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec![local_name.as_str(), CLOUD_GATEWAY_MODEL]);
        assert!(!ids.contains(&"gemma/gemma-4-e2b-it"));
    }

    #[tokio::test]
    async fn invalid_max_tokens_and_cloud_timeout() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let state = build_state(
            dir.path(),
            Router::new().unwrap(),
            CloudClient::new("http://127.0.0.1:9", "").with_mock(MockMode::Timeout),
        )
        .unwrap();
        let app = app(state);

        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages":[{"role":"user","content":"hi"}],
                            "max_tokens": 0
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let v = body_json(res).await;
        assert_eq!(v["error"]["type"], "invalid_request_error");

        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages":[{"role":"user","content":"FORCE_CLOUD please"}],
                            "max_tokens": 2
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_GATEWAY);
        let v = body_json(res).await;
        assert_eq!(v["error"]["type"], "cloud_error");
    }

    #[tokio::test]
    async fn hybrid_records_outcome_local_and_cloud() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let state = build_state(
            dir.path(),
            Router::new().unwrap(),
            CloudClient::new("http://127.0.0.1:9", "").with_mock(MockMode::Success(json!({
                "choices":[{"message":{"content":"from-cloud"}}]
            }))),
        )
        .unwrap();
        let router = state.router.clone();
        let svc = app(state);

        let res = svc
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages":[{"role":"user","content":"stay local"}],
                            "max_tokens": 2,
                            "session_id": "out-1"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(router.outcomes.len(), 1);
        let local = &router.outcomes.recent(1)[0];
        assert!(!local.cloud_handoff);
        assert_eq!(local.session_id.as_deref(), Some("out-1"));
        assert_eq!(local.policy_version, POLICY_VERSION);
        assert!(local.input_tokens.unwrap_or(0) > 0);
        assert!(local.output_tokens.unwrap_or(0) > 0);

        let res = svc
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages":[{"role":"user","content":"FORCE_CLOUD"}],
                            "max_tokens": 2,
                            "session_id": "out-1"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(router.outcomes.len(), 2);
        let cloud = &router.outcomes.recent(1)[0];
        assert!(cloud.cloud_handoff);
        assert_eq!(cloud.action, aria_hybrid::RouteAction::CloudHandoff);
    }

    #[tokio::test]
    async fn hybrid_session_sticky_then_force_upgrade() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let state = build_state(
            dir.path(),
            Router::new().unwrap(),
            CloudClient::new("http://127.0.0.1:9", "").with_mock(MockMode::Success(json!({
                "choices":[{"message":{"content":"from-cloud"}}]
            }))),
        )
        .unwrap();
        let svc = app(state);

        // Seed session as Local.
        let res = svc
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages":[{"role":"user","content":"hello"}],
                            "max_tokens": 2,
                            "session_id": "stick-http"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert_ne!(v["choices"][0]["message"]["content"], "from-cloud");

        // Soft path without FORCE_CLOUD stays local (confidence 0.95).
        let res = svc
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages":[{"role":"user","content":"again"}],
                            "max_tokens": 2,
                            "session_id": "stick-http"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert_ne!(v["choices"][0]["message"]["content"], "from-cloud");

        // Hard FORCE_CLOUD upgrades the sticky session.
        let res = svc
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages":[{"role":"user","content":"FORCE_CLOUD now"}],
                            "max_tokens": 2,
                            "session_id": "stick-http"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert_eq!(v["choices"][0]["message"]["content"], "from-cloud");
    }

    #[tokio::test]
    async fn hybrid_intelligence_mode_handoff_without_force() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        // Keyword boost → complexity ≥ Intelligence cutoff (0.40), still < Balance (0.75).
        let router = Router::new()
            .unwrap()
            .with_mode(ParetoMode::Intelligence);
        let state = build_state(
            dir.path(),
            router,
            CloudClient::new("http://127.0.0.1:9", "").with_mock(MockMode::Success(json!({
                "choices":[{"message":{"content":"intel-cloud"}}]
            }))),
        )
        .unwrap();
        let svc = app(state);
        let res = svc
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages":[{"role":"user","content":"Please analyze and reason step-by-step about this plan"}],
                            "max_tokens": 2
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert_eq!(v["choices"][0]["message"]["content"], "intel-cloud");
    }

    #[tokio::test]
    async fn hybrid_cost_mode_stays_local_when_balance_would_handoff() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        // Length score 0.35 (80–239 chars) + 3 keywords → ~0.89: Balance handoffs,
        // Cost stays local; keep under tiny fixture context_length (64 tokens).
        let mut prompt = String::from("Please analyze and reason step-by-step. ");
        while prompt.chars().count() < 100 {
            prompt.push('x');
        }
        let balance = build_state(
            dir.path(),
            Router::new().unwrap(),
            CloudClient::new("http://127.0.0.1:9", "").with_mock(MockMode::Success(json!({
                "choices":[{"message":{"content":"balance-cloud"}}]
            }))),
        )
        .unwrap();
        let res = app(balance)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages":[{"role":"user","content": prompt}],
                            "max_tokens": 2
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let v = body_json(res).await;
        assert_eq!(v["choices"][0]["message"]["content"], "balance-cloud");

        let cost = build_state(
            dir.path(),
            Router::new().unwrap().with_mode(ParetoMode::Cost),
            CloudClient::new("http://127.0.0.1:9", "").with_mock(MockMode::Success(json!({
                "choices":[{"message":{"content":"cost-cloud"}}]
            }))),
        )
        .unwrap();
        let res = app(cost)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages":[{"role":"user","content": prompt}],
                            "max_tokens": 2
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let v = body_json(res).await;
        assert_ne!(v["choices"][0]["message"]["content"], "cost-cloud");
    }

    #[tokio::test]
    async fn hybrid_unavailable_cloud_stays_local_on_force() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        // No mock, no API key → cloud_available=false
        let state = build_state(
            dir.path(),
            Router::new().unwrap(),
            CloudClient::new("http://127.0.0.1:9", ""),
        )
        .unwrap();
        assert!(!state.cloud.is_available());
        let svc = app(state);
        let res = svc
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages":[{"role":"user","content":"FORCE_CLOUD please"}],
                            "max_tokens": 2
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        // Local path succeeds rather than Cloud error.
        let v = body_json(res).await;
        assert!(v["choices"][0]["message"]["content"].is_string());
    }
}
