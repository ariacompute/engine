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
}

enum Scorer {
    Encoder(Box<EncoderScorer>),
    Decoder(DecoderScorer),
}

struct AppState {
    scorer: Scorer,
    model_name: String,
}

pub fn build_router(opts: &ServeOpts) -> anyhow::Result<Router> {
    let scorer = match opts.track {
        Track::Encoder => Scorer::Encoder(Box::new(EncoderScorer::open(&opts.checkpoint)?)),
        Track::Decoder => Scorer::Decoder(DecoderScorer::open(Some(&opts.checkpoint))?),
    };
    let state = Arc::new(AppState {
        scorer,
        model_name: opts.model_name.clone(),
    });
    Ok(Router::new()
        .route("/health", get(health))
        .route("/", get(health))
        .route("/v1/models", get(models))
        .route("/v1/systemone", post(systemone))
        .with_state(state))
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
        let ans = score_one(&st.scorer, &record).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
        })?;
        answers.insert(qid.clone(), ans);
    }
    Ok(Json(json!({
        "answers": answers,
        "model": st.model_name,
    })))
}

fn score_one(scorer: &Scorer, record: &Record) -> Result<Value, String> {
    match scorer {
        Scorer::Encoder(s) => s.score_record(record).map_err(|e| e.to_string()),
        Scorer::Decoder(s) => {
            let out = s.score_record(record).map_err(|e| e.to_string())?;
            Ok(out
                .get("systemone")
                .cloned()
                .unwrap_or(out))
        }
    }
}

pub async fn run_serve(opts: ServeOpts) -> anyhow::Result<()> {
    let app = build_router(&opts)?;
    let listener = tokio::net::TcpListener::bind(&opts.bind).await?;
    tracing::info!(
        "listening on http://{} track={} model={}",
        opts.bind,
        opts.track.as_str(),
        opts.model_name
    );
    axum::serve(listener, app).await?;
    Ok(())
}
