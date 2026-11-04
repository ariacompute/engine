//! Rust SDK: open AFM-D checkpoint and run System One.

use ariacompute_core::contract::Track;
use ariacompute_core::error::{AfmError, Result};
use ariacompute_core::systemone::record_from_systemone_question;
use ariacompute_dd::DecoderScorer;
use ariacompute_de::EncoderScorer;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

pub use ariacompute_core::config::{aria_home, load_config, save_config, AriaConfig};
pub use ariacompute_core::contract::Track as EngineTrack;
pub use ariacompute_core::packing::Record;
pub use ariacompute_core::systemone::SystemOneRequest;

enum Inner {
    Encoder(Box<EncoderScorer>),
    Decoder(Box<DecoderScorer>),
}

pub struct Engine {
    inner: Inner,
    model_name: String,
}

impl Engine {
    pub fn open(checkpoint: impl AsRef<Path>, track: Track) -> Result<Self> {
        let ckpt = checkpoint.as_ref();
        let inner = match track {
            Track::Encoder => Inner::Encoder(Box::new(EncoderScorer::open(ckpt)?)),
            Track::Decoder => Inner::Decoder(Box::new(DecoderScorer::open(Some(ckpt))?)),
        };
        Ok(Self {
            inner,
            model_name: ckpt
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| track.as_str().into()),
        })
    }

    /// Open from already-built scorers — lets upper layers (SDK tests, FFI) run the
    /// success paths without shipping ModernBERT / MiniCPM weights.
    #[doc(hidden)]
    pub fn with_encoder(scorer: EncoderScorer, model_name: impl Into<String>) -> Self {
        Self {
            inner: Inner::Encoder(Box::new(scorer)),
            model_name: model_name.into(),
        }
    }

    /// See [`Engine::with_encoder`].
    #[doc(hidden)]
    pub fn with_decoder(scorer: DecoderScorer, model_name: impl Into<String>) -> Self {
        Self {
            inner: Inner::Decoder(Box::new(scorer)),
            model_name: model_name.into(),
        }
    }

    pub fn systemone(&self, request: &SystemOneRequest) -> Result<Value> {
        let mut answers = Map::new();
        for (qid, question) in &request.questions {
            let record = record_from_systemone_question(qid, &request.state, question)?;
            answers.insert(qid.clone(), self.decide_record(&record)?);
        }
        Ok(json!({
            "answers": answers,
            "model": self.model_name,
        }))
    }

    pub fn decide_record(&self, record: &Record) -> Result<Value> {
        match &self.inner {
            Inner::Encoder(s) => s.score_record(record),
            Inner::Decoder(s) => {
                let out = s.score_record(record)?;
                Ok(out.get("systemone").cloned().unwrap_or(out))
            }
        }
    }

    /// Encoder golden path without full weights.
    pub fn decide_with_logits(&self, record: &Record, logits: &[f32]) -> Result<Value> {
        match &self.inner {
            Inner::Encoder(s) => s.score_record_with_logits(record, logits),
            Inner::Decoder(_) => Err(AfmError::msg("decide_with_logits is encoder-only")),
        }
    }

    pub fn decide_with_semif_out(&self, record: &Record, out: &Value) -> Result<Value> {
        match &self.inner {
            Inner::Decoder(s) => {
                let scored = s.score_from_semif_out(record, out)?;
                Ok(scored.get("systemone").cloned().unwrap_or(scored))
            }
            Inner::Encoder(_) => Err(AfmError::msg("decide_with_semif_out is decoder-only")),
        }
    }
}

pub fn resolve_checkpoint(name: &str) -> PathBuf {
    ariacompute_core::config::resolve_checkpoint(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ariacompute_dd::{DecoderCheckpoint, LetterScorer};
    use ariacompute_de::checkpoint::EncoderCheckpoint;
    use ariacompute_de::score::EncoderForward;
    use ariacompute_de::AgentConfig;
    use serde_json::json;
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, MutexGuard};
    use tempfile::tempdir;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn lock_env() -> MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn fixture_tokenizer() -> tokenizers::Tokenizer {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../../tests/fixtures/tokenizer_wordlevel.json");
        tokenizers::Tokenizer::from_file(path).unwrap()
    }

    fn choice_record() -> Record {
        Record::from_value(&json!({
            "id": "1",
            "task": "choice",
            "instructions": "pick",
            "options": [
                {"name": "a", "description": "alpha"},
                {"name": "b", "description": "beta"}
            ],
            "state": "hello"
        }))
        .unwrap()
    }

    fn systemone_request() -> SystemOneRequest {
        serde_json::from_value(json!({
            "state": "hello",
            "questions": {
                "q1": {
                    "type": "choice",
                    "instructions": "pick",
                    "criteria": {"a": "alpha", "b": "beta"}
                }
            }
        }))
        .unwrap()
    }

    struct StubForward {
        logits: Vec<f32>,
        calls: AtomicUsize,
    }

    impl EncoderForward for StubForward {
        fn forward_logits(
            &self,
            input_ids: &[u32],
            mask_positions: &[usize],
            type_id: usize,
        ) -> Result<Vec<f32>> {
            assert!(!input_ids.is_empty());
            assert!(!mask_positions.is_empty());
            assert!(type_id <= 2);
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.logits.clone())
        }
    }

    fn stub_encoder(logits: Vec<f32>) -> EncoderScorer {
        EncoderScorer::from_parts(
            EncoderCheckpoint {
                root: PathBuf::from("/nonexistent/afm-de"),
                config: AgentConfig::default(),
                weights_path: PathBuf::from("/nonexistent/afm-de/model.safetensors"),
                tokenizer_dir: PathBuf::from("/nonexistent/afm-de"),
                encoder_dir: PathBuf::from("/nonexistent/afm-de"),
            },
            Some(fixture_tokenizer()),
            Arc::new(StubForward {
                logits,
                calls: AtomicUsize::new(0),
            }),
        )
    }

    struct StubLetters {
        ids: Vec<String>,
        probs: Vec<f32>,
    }

    impl LetterScorer for StubLetters {
        fn letter_logits(&mut self, row: &Value) -> Result<(Vec<String>, Vec<f32>)> {
            assert!(row.get("options").is_some());
            Ok((self.ids.clone(), self.probs.clone()))
        }
    }

    fn stub_decoder(ids: &[&str], probs: Vec<f32>) -> DecoderScorer {
        DecoderScorer::from_parts(
            DecoderCheckpoint::open(None::<&Path>).unwrap(),
            Box::new(StubLetters {
                ids: ids.iter().map(|s| s.to_string()).collect(),
                probs,
            }),
        )
    }

    #[test]
    fn open_without_weights_keeps_model_name() {
        let dir = tempdir().unwrap();
        let ckpt = dir.path().join("afm-de");
        std::fs::create_dir_all(&ckpt).unwrap();
        let engine = Engine::open(&ckpt, Track::Encoder).unwrap();
        let err = engine.systemone(&systemone_request()).unwrap_err();
        assert!(err.to_string().contains("encoder weights missing"));
        // model_name is the checkpoint's final path component
        assert_eq!(
            Engine::open(&ckpt, Track::Decoder).unwrap().model_name,
            "afm-de"
        );

        // name falls back to the track when the path has no final component
        let bare = Engine::open(Path::new("/"), Track::Decoder).unwrap();
        assert_eq!(
            bare.systemone(&SystemOneRequest {
                state: json!("s"),
                questions: Default::default(),
                model: None,
            })
            .unwrap()["model"],
            "decoder"
        );
    }

    #[test]
    fn decide_with_logits_is_encoder_only_and_works() {
        let engine = Engine::with_encoder(stub_encoder(vec![0.1, 3.0]), "afm-de");
        let ans = engine
            .decide_with_logits(&choice_record(), &[0.2, 2.0])
            .unwrap();
        assert_eq!(ans["choice"], "b");

        let ans = engine.systemone(&systemone_request()).unwrap();
        assert_eq!(ans["answers"]["q1"]["choice"], "b");
        assert_eq!(ans["model"], "afm-de");

        let decoder = Engine::with_decoder(stub_decoder(&["a", "b"], vec![0.1, 0.9]), "afm-dd");
        let err = decoder
            .decide_with_logits(&choice_record(), &[1.0, 0.0])
            .unwrap_err();
        assert_eq!(err.to_string(), "decide_with_logits is encoder-only");
    }

    #[test]
    fn decide_with_semif_out_is_decoder_only_and_works() {
        let engine = Engine::with_decoder(stub_decoder(&["a", "b"], vec![0.1, 0.9]), "afm-dd");
        let out = json!({"option_ids": ["a", "b"], "probabilities": [0.1, 0.9]});
        let ans = engine
            .decide_with_semif_out(&choice_record(), &out)
            .unwrap();
        assert_eq!(ans["choice"], "b");

        // systemone goes through the decoder `systemone` sub-document
        let ans = engine.systemone(&systemone_request()).unwrap();
        assert_eq!(ans["answers"]["q1"]["choice"], "b");
        assert_eq!(ans["model"], "afm-dd");

        let encoder = Engine::with_encoder(stub_encoder(vec![0.1, 3.0]), "afm-de");
        let err = encoder
            .decide_with_semif_out(&choice_record(), &out)
            .unwrap_err();
        assert_eq!(err.to_string(), "decide_with_semif_out is decoder-only");
    }

    #[test]
    fn systemone_propagates_bad_questions() {
        let engine = Engine::with_encoder(stub_encoder(vec![0.1, 3.0]), "afm-de");
        let bad: SystemOneRequest = serde_json::from_value(json!({
            "state": "s",
            "questions": {"q1": {"type": "choice"}}
        }))
        .unwrap();
        let err = engine.systemone(&bad).unwrap_err();
        assert!(err.to_string().contains("needs nonempty criteria"));

        let empty = Engine::with_encoder(stub_encoder(vec![]), "afm-de");
        let err = empty.systemone(&systemone_request()).unwrap_err();
        assert!(err.to_string().contains("logits len 0 != options 2"));
    }

    #[test]
    fn open_rejects_bad_checkpoint() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("nope");
        let err = Engine::open(&missing, Track::Encoder)
            .err()
            .expect("expected failure");
        assert!(err.to_string().contains("not a directory"));

        let err = Engine::open(&missing, Track::Decoder)
            .err()
            .expect("expected failure");
        assert!(err.to_string().contains("not a directory"));
    }

    #[test]
    #[serial_test::serial]
    fn resolve_checkpoint_prefers_existing_paths_then_models_dir() {
        let _guard = lock_env();
        let dir = tempdir().unwrap();
        std::env::set_var("ARIA_COMPUTE_HOME", dir.path());

        assert_eq!(
            resolve_checkpoint("afm-de"),
            dir.path().join("models").join("afm-de")
        );
        assert_eq!(
            resolve_checkpoint("vendor/afm-de"),
            PathBuf::from("vendor/afm-de")
        );
        assert_eq!(
            resolve_checkpoint(r"vendor\afm-de"),
            PathBuf::from(r"vendor\afm-de")
        );
        assert_eq!(
            resolve_checkpoint("/abs/afm-de"),
            PathBuf::from("/abs/afm-de")
        );

        std::env::remove_var("ARIA_COMPUTE_HOME");
    }

    #[test]
    fn decide_record_matches_track() {
        let rec: Record = Record::from_value(&json!({
            "id": "n1",
            "task": "noul",
            "instructions": "check",
            "options": [
                {"name": "false", "description": "no"},
                {"name": "true", "description": "yes"}
            ],
            "state": "s"
        }))
        .unwrap();
        let engine = Engine::with_decoder(stub_decoder(&["true"], vec![0.7]), "afm-dd");
        let ans = engine.decide_record(&rec).unwrap();
        assert!((ans["noul"].as_f64().unwrap() - 0.7).abs() < 1e-6, "{ans}");

        // option list is validated before any forward pass
        let single: Record = Record::from_value(&json!({
            "id": "x",
            "task": "choice",
            "instructions": "pick",
            "options": [{"name": "only", "description": ""}],
            "state": "s"
        }))
        .unwrap();
        let err = engine.decide_record(&single).unwrap_err();
        assert!(err.to_string().contains("need 2..16 options"));
    }
}
