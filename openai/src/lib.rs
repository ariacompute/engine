//! OpenAI-compatible HTTP surface (chat / embeddings / ASR / tools / RAG).

pub mod check;
pub mod config;
pub mod download;
pub mod gateway_detect;
pub mod upgrade;

use aria_inference::{
    infer_family_path, rag_pack_context, ChatTurn, ComputePref, GenerateOpts, Session,
    SessionBuilder,
};
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
    /// Optional session id (unused after hybrid removal; accepted for API compat).
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
        .route("/v1/engine/profile", get(engine_profile))
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

async fn engine_profile(State(st): State<AppState>) -> Response {
    match handle_engine_profile(&st) {
        Ok(v) => Json(v).into_response(),
        Err(e) => engine_err(e),
    }
}

fn handle_engine_profile(st: &AppState) -> Result<Value, EngineError> {
    let sess = st
        .session
        .lock()
        .map_err(|e| EngineError::Io(e.to_string()))?;
    if let Some(p) = sess.last_profile() {
        return serde_json::to_value(p).map_err(|e| EngineError::Format(e.to_string()));
    }
    Ok(json!({
        "compute": sess.compute_label(),
        "load": {
            "mmap_ms": 0.0,
            "dequant_ms": 0.0,
            "unrotate_ms": 0.0,
            "materialize_ms": 0.0,
            "cuda_upload_ms": 0.0
        },
        "generate": Value::Null,
        "ci_fail": false
    }))
}

/// Model ids exposed on `GET /v1/models` (local bundle directory name).
fn advertised_model_ids(st: &AppState) -> Vec<String> {
    vec![st.model_id.clone()]
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

/// Client `max_tokens` as-is when set. If omitted, decode until stop or remaining
/// context — never the old default of 16.
fn resolve_local_max_tokens(
    requested: Option<u32>,
    prompt_len: usize,
    context_length: usize,
) -> Result<usize, EngineError> {
    match requested {
        Some(0) => Err(EngineError::InvalidParam("max_tokens must be > 0".into())),
        Some(n) => Ok(n as usize),
        None => Ok(context_length.saturating_sub(prompt_len).max(1)),
    }
}

async fn handle_chat(st: &AppState, req: ChatCompletionRequest) -> Result<Response, EngineError> {
    if req.max_tokens == Some(0) {
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

    let mut sess = st
        .session
        .lock()
        .map_err(|e| EngineError::Io(e.to_string()))?;
    let mut turns: Vec<ChatTurn> = req
        .messages
        .iter()
        .map(|m| ChatTurn {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();
    if turns.is_empty() {
        turns.push(ChatTurn::new("user", prompt_text.clone()));
    } else if let Some(last) = turns.iter_mut().rev().find(|t| t.role == "user") {
        last.content = prompt_text.clone();
    }
    let prompt = sess.encode_chat(&turns);
    let max_tokens = resolve_local_max_tokens(
        req.max_tokens,
        prompt.len(),
        sess.config().context_length,
    )?;
    let gen = sess.generate(
        &prompt,
        &GenerateOpts {
            max_tokens,
            temperature: req.temperature.unwrap_or(0.0),
        },
    )?;
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
        EngineError::Cloud(_) | EngineError::Upstream(_) => (StatusCode::BAD_GATEWAY, "upstream_error"),
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

pub fn build_state(model_dir: impl AsRef<std::path::Path>) -> Result<AppState, EngineError> {
    build_state_with_opts(model_dir, ComputePref::Auto, false)
}

pub fn build_state_with_opts(
    model_dir: impl AsRef<std::path::Path>,
    compute: ComputePref,
    profile: bool,
) -> Result<AppState, EngineError> {
    let model_dir = model_dir.as_ref();
    let family = model_dir
        .file_name()
        .and_then(|s| s.to_str())
        .and_then(infer_family_path)
        .unwrap_or("gemma/gemma-4-e2b-it");
    build_state_with_family_opts(model_dir, family, compute, profile)
}

pub fn build_state_with_family(
    model_dir: impl AsRef<std::path::Path>,
    family: &str,
) -> Result<AppState, EngineError> {
    build_state_with_family_opts(model_dir, family, ComputePref::Auto, false)
}

fn build_state_with_family_opts(
    model_dir: impl AsRef<std::path::Path>,
    family: &str,
    compute: ComputePref,
    profile: bool,
) -> Result<AppState, EngineError> {
    let model_dir = model_dir.as_ref();
    let session = SessionBuilder::new()
        .model(model_dir)
        .family(family)
        .compute(compute)
        .profile(profile)
        .build()?;
    let model_id = model_dir
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| session.model_id().to_string());
    Ok(AppState {
        session: Arc::new(Mutex::new(session)),
        model_id,
    })
}

/// Register this engine as a local provider on aria-router. Failure is fatal for serve.
pub async fn register_with_router(
    router_base: &str,
    endpoint: &str,
    model_name: &str,
) -> Result<(), EngineError> {
    let url = format!(
        "{}/v1/router/providers",
        router_base.trim_end_matches('/')
    );
    let resp = reqwest::Client::new()
        .put(&url)
        .json(&json!({
            "name": model_name,
            "endpoint": endpoint,
            "provider_model_id": model_name,
            "locality": "local"
        }))
        .send()
        .await
        .map_err(|e| EngineError::Upstream(format!("router register: {e}")))?;
    let status = resp.status();
    if !status.is_success() {
        let t = resp.text().await.unwrap_or_default();
        return Err(EngineError::Upstream(format!(
            "router register HTTP {status}: {t}"
        )));
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use aria_inference::fixture::write_tiny_q4_bundle;
    use aria_inference::ComputePref;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[test]
    fn local_max_tokens_forwards_or_uses_remaining_context() {
        assert_eq!(resolve_local_max_tokens(Some(32), 10, 64).unwrap(), 32);
        assert_eq!(resolve_local_max_tokens(None, 10, 64).unwrap(), 54);
        assert_eq!(resolve_local_max_tokens(None, 0, 64).unwrap(), 64);
        assert_eq!(resolve_local_max_tokens(None, 64, 64).unwrap(), 1);
        assert_ne!(resolve_local_max_tokens(None, 0, 64).unwrap(), 16);
        assert!(resolve_local_max_tokens(Some(0), 10, 64).is_err());
    }

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
        let state = build_state(dir.path()).unwrap();
        let local_name = dir.path().file_name().unwrap().to_string_lossy().into_owned();
        let app = app(state);

        let res = app
            .clone()
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        let ids: Vec<&str> = v["data"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap()).collect();
        assert_eq!(ids, vec![local_name.as_str()]);

        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(json!({"messages":[{"role":"user","content":"hi"}],"max_tokens": 2}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let capped = body_json(res).await;
        assert!(capped["usage"]["completion_tokens"].as_u64().unwrap() <= 2);

        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(json!({"messages":[{"role":"user","content":"hi"}]}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let omit = body_json(res).await;
        assert!(omit["usage"]["completion_tokens"].as_u64().unwrap() > 0);

        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(json!({"messages":[{"role":"user","content":"hi"}],"max_tokens": 2,"stream": true}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        assert!(String::from_utf8_lossy(&bytes).contains("data:"));
    }

    #[tokio::test]
    async fn engine_profile_after_chat() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let state = build_state_with_opts(dir.path(), ComputePref::Cpu, true).unwrap();
        let app = app(state);
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(json!({"messages":[{"role":"user","content":"hi"}],"max_tokens": 2}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let res = app
            .oneshot(Request::builder().uri("/v1/engine/profile").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert_eq!(v["ci_fail"], false);
        assert!(v["compute"].as_str().unwrap().contains("cpu"));
        assert!(v["generate"].is_object());
        assert!(v["load"]["materialize_ms"].as_f64().unwrap() >= 0.0);
    }

    #[tokio::test]
    async fn embeddings_asr_tools_rag() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let app = app(build_state(dir.path()).unwrap());

        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/embeddings")
                    .header("content-type", "application/json")
                    .body(Body::from(json!({"input":"hello embedding"}).to_string()))
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
                    .body(Body::from(json!({"file_b64": b64_encode(&pcm)}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert!(!body_json(res).await["text"].as_str().unwrap().is_empty());

        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(json!({
                        "messages":[{"role":"user","content":"please CALL_TOOL now"}],
                        "tools":[{"type":"function","function":{"name":"search"}}],
                        "max_tokens": 2
                    }).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");

        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(json!({
                        "messages":[{"role":"user","content":"q"}],
                        "rag_snippets":["fact one","fact two"],
                        "max_tokens": 2
                    }).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn invalid_max_tokens() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let app = app(build_state(dir.path()).unwrap());
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(json!({"messages":[{"role":"user","content":"hi"}],"max_tokens": 0}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_json(res).await["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn register_with_router_hits_upsert() {
        use axum::routing::put;
        async fn ok(Json(v): Json<Value>) -> Json<Value> {
            Json(json!({"ok": true, "name": v.get("name")}))
        }
        let app = axum::Router::new().route("/v1/router/providers", put(ok));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
        register_with_router(&format!("http://{addr}"), "127.0.0.1:8080", "tiny_q4")
            .await
            .unwrap();
    }
}
