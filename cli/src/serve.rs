//! System One HTTP server (`POST /v1/systemone`).

use std::path::PathBuf;
use std::sync::Arc;

use ariacompute_core::contract::Track;
use ariacompute_core::packing::Record;
use ariacompute_core::systemone::{record_from_systemone_question, SystemOneRequest};
use ariacompute_dd::DecoderScorer;
use ariacompute_de::EncoderScorer;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Map, Value};

#[derive(Clone)]
pub struct ServeOpts {
    pub track: Track,
    pub checkpoint: PathBuf,
    pub model_name: String,
    pub bind: String,
    /// Local compute preference from setup: auto | cpu | cuda (logged; device selection for candle).
    pub compute: String,
}

/// One typed-decision backend behind the HTTP face.
pub trait Judge: Send + Sync {
    fn score(&self, record: &Record) -> Result<Value, String>;
}

struct EncoderJudge(EncoderScorer);

impl Judge for EncoderJudge {
    fn score(&self, record: &Record) -> Result<Value, String> {
        self.0.score_record(record).map_err(|e| e.to_string())
    }
}

struct DecoderJudge(DecoderScorer);

impl Judge for DecoderJudge {
    fn score(&self, record: &Record) -> Result<Value, String> {
        let out = self.0.score_record(record).map_err(|e| e.to_string())?;
        Ok(out.get("systemone").cloned().unwrap_or(out))
    }
}

struct AppState {
    judge: Arc<dyn Judge>,
    model_name: String,
}

pub fn build_router(opts: &ServeOpts) -> anyhow::Result<Router> {
    let judge: Arc<dyn Judge> = match opts.track {
        Track::Encoder => Arc::new(EncoderJudge(EncoderScorer::open(&opts.checkpoint)?)),
        Track::Decoder => Arc::new(DecoderJudge(DecoderScorer::open(Some(&opts.checkpoint))?)),
    };
    Ok(router_with(judge, opts.model_name.clone()))
}

/// Build the router around an injected judge (serve tests / embedding).
pub fn router_with(judge: Arc<dyn Judge>, model_name: impl Into<String>) -> Router {
    let state = Arc::new(AppState {
        judge,
        model_name: model_name.into(),
    });
    Router::new()
        .route("/health", get(health))
        .route("/", get(health))
        .route("/v1/models", get(models))
        .route("/v1/systemone", post(systemone))
        .with_state(state)
}

async fn health(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "ok": true,
        "service": "afm-engine",
        "model": st.model_name,
    }))
}

async fn models(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": st.model_name, "object": "model", "owned_by": "afm-d"}],
    }))
}

async fn systemone(
    State(st): State<Arc<AppState>>,
    Json(body): Json<SystemOneRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let mut answers = Map::new();
    for (qid, question) in &body.questions {
        let record = record_from_systemone_question(qid, &body.state, question).map_err(|e| {
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"detail": e.to_string()})),
            )
        })?;
        let ans = st
            .judge
            .score(&record)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e}))))?;
        answers.insert(qid.clone(), ans);
    }
    Ok(Json(json!({
        "answers": answers,
        "model": st.model_name,
    })))
}

/// Serve `app` on `listener` until `shutdown` resolves (then drain in-flight requests).
pub async fn serve_on<F>(
    listener: tokio::net::TcpListener,
    app: Router,
    shutdown: F,
) -> anyhow::Result<()>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

/// Production entry: shut down on Ctrl-C / SIGINT.
pub async fn run_serve(opts: ServeOpts) -> anyhow::Result<()> {
    // Best effort: if the signal driver is unavailable the server keeps serving
    // until the process is terminated some other way.
    run_serve_with(opts, async {
        let _ = tokio::signal::ctrl_c().await;
    })
    .await
}

/// Like `run_serve` with an injectable shutdown future (tests / embedding).
pub async fn run_serve_with<F>(opts: ServeOpts, shutdown: F) -> anyhow::Result<()>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let compute =
        ariacompute_core::config::parse_compute(&opts.compute).map_err(anyhow::Error::msg)?;
    let app = build_router(&opts)?;
    let listener = tokio::net::TcpListener::bind(&opts.bind).await?;
    let track = opts.track.as_str();
    tracing::info!(
        "listening on http://{} track={} model={} compute={}",
        opts.bind,
        track,
        opts.model_name,
        compute
    );
    serve_on(listener, app, shutdown).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    struct StubJudge {
        fail: bool,
    }

    impl Judge for StubJudge {
        fn score(&self, record: &Record) -> Result<Value, String> {
            if self.fail {
                return Err("scoring failed".into());
            }
            Ok(json!({"choice": record.options[0].name.clone()}))
        }
    }

    async fn call(app: Router, req: Request<Body>) -> (StatusCode, Value) {
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, body)
    }

    fn systemone_request(question: Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/systemone")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"state": "s", "questions": {"q1": question}}).to_string(),
            ))
            .unwrap()
    }

    fn choice_question() -> Value {
        json!({"type": "choice", "criteria": {"a": "alpha", "b": "beta"}})
    }

    #[tokio::test]
    async fn health_models_and_systemone_succeed() {
        let app = router_with(Arc::new(StubJudge { fail: false }), "afm-de");

        let (status, body) = call(
            app.clone(),
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["ok"], true);
        assert_eq!(body["model"], "afm-de");

        let (status, body) = call(
            app.clone(),
            Request::builder().uri("/").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["service"], "afm-engine");

        let (status, body) = call(
            app.clone(),
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["data"][0]["id"], "afm-de");
        assert_eq!(body["data"][0]["owned_by"], "afm-d");

        let (status, body) = call(app, systemone_request(choice_question())).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["model"], "afm-de");
        assert!(body["answers"]["q1"]["choice"].is_string());
    }

    #[tokio::test]
    async fn systemone_maps_bad_question_to_422() {
        let app = router_with(Arc::new(StubJudge { fail: false }), "afm-de");
        let (status, body) = call(app, systemone_request(json!({"type": "rank"}))).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(body["detail"]
            .as_str()
            .unwrap()
            .contains("unsupported systemone type"));
    }

    #[tokio::test]
    async fn systemone_maps_scoring_failure_to_500() {
        let app = router_with(Arc::new(StubJudge { fail: true }), "afm-de");
        let (status, body) = call(app, systemone_request(choice_question())).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body["error"], "scoring failed");
    }

    #[tokio::test]
    async fn build_router_supports_both_tracks() {
        let dir = tempfile::tempdir().unwrap();
        for track in [Track::Encoder, Track::Decoder] {
            let opts = ServeOpts {
                track,
                checkpoint: dir.path().to_path_buf(),
                model_name: format!("afm-{}", track.as_str()),
                bind: "127.0.0.1:0".into(),
                compute: "cpu".into(),
            };
            let app = build_router(&opts).unwrap();
            // scoring without weights must fail closed with a 500
            let (status, body) = call(app, systemone_request(choice_question())).await;
            assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{track:?}");
            let msg = body["error"].as_str().unwrap();
            assert!(msg.contains("weights"), "{msg}");
        }
    }

    #[tokio::test]
    async fn build_router_propagates_open_errors() {
        let dir = tempfile::tempdir().unwrap();
        let opts = ServeOpts {
            track: Track::Encoder,
            checkpoint: dir.path().join("missing"),
            model_name: "afm-de".into(),
            bind: "127.0.0.1:0".into(),
            compute: "cpu".into(),
        };
        let err = build_router(&opts)
            .map(|_| ())
            .expect_err("expected open error");
        assert!(err.to_string().contains("not a directory"));
    }

    #[test]
    fn decoder_judge_unwraps_the_systemone_block() {
        struct Stub;
        impl ariacompute_dd::LetterScorer for Stub {
            fn letter_logits(
                &mut self,
                _row: &Value,
            ) -> ariacompute_core::error::Result<(Vec<String>, Vec<f32>)> {
                Ok((vec!["a".into(), "b".into()], vec![0.2, 0.8]))
            }
        }
        let scorer = DecoderScorer::from_parts(
            ariacompute_dd::DecoderCheckpoint::open(None::<&std::path::Path>).unwrap(),
            Box::new(Stub),
        );
        let judge = DecoderJudge(scorer);
        let record = Record::from_value(&json!({
            "id": "1",
            "task": "choice",
            "instructions": "pick",
            "options": [{"name": "a", "description": "alpha"}, {"name": "b", "description": "beta"}],
            "state": "s"
        }))
        .unwrap();
        let ans = judge.score(&record).unwrap();
        assert_eq!(ans["choice"], "b");
        assert!(ans.get("systemone").is_none());
    }

    #[tokio::test]
    async fn run_serve_with_binds_and_stops_on_shutdown() {
        let dir = tempfile::tempdir().unwrap();
        let opts = ServeOpts {
            track: Track::Encoder,
            checkpoint: dir.path().to_path_buf(),
            model_name: "afm-de".into(),
            bind: "127.0.0.1:0".into(),
            compute: "auto".into(),
        };
        run_serve_with(opts, std::future::ready(())).await.unwrap();
    }

    #[tokio::test]
    async fn run_serve_rejects_bad_compute_and_bad_bind() {
        let dir = tempfile::tempdir().unwrap();
        let mut opts = ServeOpts {
            track: Track::Encoder,
            checkpoint: dir.path().to_path_buf(),
            model_name: "afm-de".into(),
            bind: "127.0.0.1:0".into(),
            compute: "gpu".into(),
        };
        let err = run_serve_with(opts.clone(), std::future::ready(()))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("invalid compute"));

        opts.compute = "cpu".into();
        opts.bind = "definitely-not-a-host:0".into();
        let err = run_serve_with(opts, std::future::ready(()))
            .await
            .unwrap_err();
        assert!(!err.to_string().is_empty());
    }
}
