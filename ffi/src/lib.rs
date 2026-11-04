//! C ABI for AFM-D System One decide.

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use ariacompute_core::config::model_cache_dir;
use ariacompute_core::contract::Track;
use ariacompute_core::systemone::{record_from_systemone_question, SystemOneRequest};
use ariacompute_dd::DecoderScorer;
use ariacompute_de::EncoderScorer;
use serde_json::{json, Map, Value};
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::{Path, PathBuf};
use std::ptr;
use std::slice;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
    static LAST_STRING: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(msg: impl Into<String>) {
    let s = CString::new(msg.into().replace('\0', "")).unwrap_or_default();
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(s));
}

fn set_string(msg: impl Into<String>) -> *const c_char {
    let s = CString::new(msg.into().replace('\0', "")).unwrap_or_default();
    LAST_STRING.with(|slot| {
        *slot.borrow_mut() = Some(s);
        slot.borrow()
            .as_ref()
            .map(|c| c.as_ptr())
            .unwrap_or(ptr::null())
    })
}

enum Inner {
    Encoder(Box<EncoderScorer>),
    Decoder(Box<DecoderScorer>),
}

pub struct AriaModel {
    inner: Inner,
}

fn cstr<'a>(p: *const c_char) -> Result<&'a str, String> {
    if p.is_null() {
        return Err("null pointer".into());
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|e| e.to_string())
}

#[no_mangle]
pub extern "C" fn aria_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|c| c.as_ptr())
            .unwrap_or(ptr::null())
    })
}

#[no_mangle]
pub extern "C" fn aria_model_cache_dir(model: *const c_char) -> *const c_char {
    match (|| -> Result<String, String> {
        let m = cstr(model)?;
        let p = model_cache_dir(m).map_err(|e| e.to_string())?;
        Ok(p.display().to_string())
    })() {
        Ok(s) => set_string(s),
        Err(e) => {
            set_error(e);
            ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn aria_is_local_path(ref_: *const c_char) -> c_int {
    match cstr(ref_) {
        Ok(s) => {
            if s.contains('/') || s.contains('\\') || Path::new(s).exists() {
                1
            } else {
                0
            }
        }
        Err(e) => {
            set_error(e);
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn aria_model_init(
    checkpoint_path: *const c_char,
    track: *const c_char,
) -> *mut AriaModel {
    match (|| -> Result<AriaModel, String> {
        let path = cstr(checkpoint_path)?;
        let track_s = cstr(track).unwrap_or("encoder");
        let track = Track::parse(track_s)?;
        let ckpt = PathBuf::from(path);
        let inner = match track {
            Track::Encoder => Inner::Encoder(Box::new(
                EncoderScorer::open(&ckpt).map_err(|e| e.to_string())?,
            )),
            Track::Decoder => Inner::Decoder(Box::new(
                DecoderScorer::open(Some(&ckpt)).map_err(|e| e.to_string())?,
            )),
        };
        Ok(AriaModel { inner })
    })() {
        Ok(m) => Box::into_raw(Box::new(m)),
        Err(e) => {
            set_error(e);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn aria_model_destroy(model: *mut AriaModel) {
    if !model.is_null() {
        unsafe {
            drop(Box::from_raw(model));
        }
    }
}

fn score_request(model: &AriaModel, body: &SystemOneRequest) -> Result<Value, String> {
    let mut answers = Map::new();
    for (qid, question) in &body.questions {
        let record = record_from_systemone_question(qid, &body.state, question)
            .map_err(|e| e.to_string())?;
        let ans = match &model.inner {
            Inner::Encoder(s) => s.score_record(&record).map_err(|e| e.to_string())?,
            Inner::Decoder(s) => {
                let out = s.score_record(&record).map_err(|e| e.to_string())?;
                out.get("systemone").cloned().unwrap_or(out)
            }
        };
        answers.insert(qid.clone(), ans);
    }
    Ok(json!({ "answers": answers }))
}

#[no_mangle]
pub extern "C" fn aria_systemone(
    model: *mut AriaModel,
    request_json: *const c_char,
    out: *mut c_char,
    out_len: usize,
) -> c_int {
    if model.is_null() || out.is_null() || out_len == 0 {
        set_error("null model/out or zero out_len");
        return -1;
    }
    let result = (|| -> Result<String, String> {
        let raw = cstr(request_json)?;
        let body: SystemOneRequest =
            serde_json::from_str(raw).map_err(|e| format!("invalid json: {e}"))?;
        let model = unsafe { &*model };
        let resp = score_request(model, &body)?;
        serde_json::to_string(&resp).map_err(|e| e.to_string())
    })();
    match result {
        Ok(s) => {
            let bytes = s.as_bytes();
            if bytes.len() + 1 > out_len {
                set_error(format!(
                    "output buffer too small (need {})",
                    bytes.len() + 1
                ));
                return -2;
            }
            unsafe {
                // c_char is u8 on some targets (e.g. aarch64 linux) and i8 on others;
                // Pointer::cast stays portable without an unnecessary `as` cast.
                let slice = slice::from_raw_parts_mut(out.cast::<u8>(), out_len);
                slice[..bytes.len()].copy_from_slice(bytes);
                slice[bytes.len()] = 0;
            }
            0
        }
        Err(e) => {
            set_error(e);
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ariacompute_core::error::Result as AriResult;
    use ariacompute_dd::{DecoderCheckpoint, LetterScorer};
    use ariacompute_de::checkpoint::EncoderCheckpoint;
    use ariacompute_de::score::EncoderForward;
    use ariacompute_de::AgentConfig;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, MutexGuard};
    use tempfile::tempdir;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn lock_env() -> MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn fixture_tokenizer() -> tokenizers::Tokenizer {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../tests/fixtures/tokenizer_wordlevel.json");
        tokenizers::Tokenizer::from_file(path).unwrap()
    }

    fn encoder_checkpoint() -> EncoderCheckpoint {
        let root = PathBuf::from("/nonexistent/afm-de");
        EncoderCheckpoint {
            root: root.clone(),
            config: AgentConfig::default(),
            weights_path: root.join("model.safetensors"),
            tokenizer_dir: root.clone(),
            encoder_dir: root,
        }
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
        ) -> AriResult<Vec<f32>> {
            assert!(!input_ids.is_empty());
            assert!(!mask_positions.is_empty());
            assert!(type_id <= 2);
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.logits.clone())
        }
    }

    /// Owned boxed pointer for a model with injected stub logits.
    fn encoder_model(logits: Vec<f32>) -> *mut AriaModel {
        let scorer = EncoderScorer::from_parts(
            encoder_checkpoint(),
            Some(fixture_tokenizer()),
            Arc::new(StubForward {
                logits,
                calls: AtomicUsize::new(0),
            }),
        );
        Box::into_raw(Box::new(AriaModel {
            inner: Inner::Encoder(Box::new(scorer)),
        }))
    }

    struct StubLetters {
        ids: Vec<String>,
        probs: Vec<f32>,
    }

    impl LetterScorer for StubLetters {
        fn letter_logits(&mut self, row: &Value) -> AriResult<(Vec<String>, Vec<f32>)> {
            assert!(row.get("options").is_some());
            Ok((self.ids.clone(), self.probs.clone()))
        }
    }

    fn decoder_model() -> *mut AriaModel {
        let scorer = DecoderScorer::from_parts(
            DecoderCheckpoint::open(None::<&Path>).unwrap(),
            Box::new(StubLetters {
                ids: vec!["a".into(), "b".into()],
                probs: vec![0.1, 0.9],
            }),
        );
        Box::into_raw(Box::new(AriaModel {
            inner: Inner::Decoder(Box::new(scorer)),
        }))
    }

    /// `{"state":…, "questions":{"q1":{"type":"choice",…}}}` as a C string.
    fn choice_request() -> CString {
        CString::new(
            json!({
                "state": "hello",
                "questions": {
                    "q1": {
                        "type": "choice",
                        "instructions": "pick",
                        "criteria": {"a": "alpha", "b": "beta"}
                    }
                }
            })
            .to_string(),
        )
        .unwrap()
    }

    fn last_error_text() -> String {
        let p = aria_last_error();
        if p.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }

    #[test]
    fn cstr_rejects_null_and_bad_utf8() {
        let err = cstr(ptr::null()).unwrap_err();
        assert_eq!(err, "null pointer");

        let bad = CString::new(vec![0x66_u8, 0xC3]).unwrap();
        let err = cstr(bad.as_ptr()).unwrap_err();
        assert!(!err.is_empty());

        let good = CString::new("hello").unwrap();
        assert_eq!(cstr(good.as_ptr()).unwrap(), "hello");
    }

    #[test]
    fn is_local_path_variants() {
        let missing = CString::new("afm-de").unwrap();
        assert_eq!(aria_is_local_path(missing.as_ptr()), 0);

        let slashed = CString::new("models/afm-de").unwrap();
        assert_eq!(aria_is_local_path(slashed.as_ptr()), 1);

        let windows = CString::new(r"models\afm-de").unwrap();
        assert_eq!(aria_is_local_path(windows.as_ptr()), 1);

        let dir = tempdir().unwrap();
        let existing = CString::new(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(aria_is_local_path(existing.as_ptr()), 1);

        assert_eq!(aria_is_local_path(ptr::null()), -1);
        assert_eq!(last_error_text(), "null pointer");
    }

    #[test]
    #[serial_test::serial]
    fn model_cache_dir_reports_errors_and_paths() {
        let _guard = lock_env();
        assert!(aria_model_cache_dir(ptr::null()).is_null());
        assert_eq!(last_error_text(), "null pointer");

        let dir = tempdir().unwrap();
        std::env::set_var("ARIA_COMPUTE_HOME", dir.path());
        let name = CString::new("afm-de").unwrap();
        let p = aria_model_cache_dir(name.as_ptr());
        assert!(!p.is_null());
        let resolved = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
        assert_eq!(
            resolved,
            dir.path().join("models/afm-de").display().to_string()
        );
        std::env::remove_var("ARIA_COMPUTE_HOME");
    }

    #[test]
    fn model_init_rejects_null_and_bad_track() {
        assert!(aria_model_init(ptr::null(), ptr::null()).is_null());
        assert_eq!(last_error_text(), "null pointer");

        let dir = tempdir().unwrap();
        let path = CString::new(dir.path().to_str().unwrap()).unwrap();
        let bad = CString::new("bogus").unwrap();
        assert!(aria_model_init(path.as_ptr(), bad.as_ptr()).is_null());
        assert!(last_error_text().contains("unknown track"));
    }

    #[test]
    fn model_init_opens_encoder_and_decoder_without_weights() {
        let dir = tempdir().unwrap();
        let path = CString::new(dir.path().to_str().unwrap()).unwrap();

        let enc = CString::new("encoder").unwrap();
        let enc_model = aria_model_init(path.as_ptr(), enc.as_ptr());
        assert!(!enc_model.is_null());
        aria_model_destroy(enc_model);

        let dec = CString::new("decoder").unwrap();
        let dec_model = aria_model_init(path.as_ptr(), dec.as_ptr());
        assert!(!dec_model.is_null());
        aria_model_destroy(dec_model);

        // a null track falls back to the encoder default
        let defaulted = aria_model_init(path.as_ptr(), ptr::null());
        assert!(!defaulted.is_null());
        aria_model_destroy(defaulted);

        aria_model_destroy(ptr::null_mut());
    }

    #[test]
    fn model_init_reports_missing_checkpoint() {
        let dir = tempdir().unwrap();
        let missing = CString::new(dir.path().join("nope").to_str().unwrap()).unwrap();
        let enc = CString::new("encoder").unwrap();
        assert!(aria_model_init(missing.as_ptr(), enc.as_ptr()).is_null());
        assert!(last_error_text().contains("not a directory"));
    }

    #[test]
    fn systemone_validates_arguments() {
        let mut buf = [0i8; 32];
        assert_eq!(
            aria_systemone(ptr::null_mut(), ptr::null(), buf.as_mut_ptr(), buf.len()),
            -1
        );
        assert_eq!(last_error_text(), "null model/out or zero out_len");

        let model = encoder_model(vec![0.1, 3.0]);
        let req = choice_request();
        assert_eq!(aria_systemone(model, req.as_ptr(), ptr::null_mut(), 0), -1);
        assert_eq!(last_error_text(), "null model/out or zero out_len");
        aria_model_destroy(model);
    }

    #[test]
    fn systemone_rejects_bad_json() {
        let model = encoder_model(vec![0.1, 3.0]);
        let bad = CString::new("{not json").unwrap();
        let mut buf = [0i8; 64];
        assert_eq!(
            aria_systemone(model, bad.as_ptr(), buf.as_mut_ptr(), buf.len()),
            -1
        );
        assert!(last_error_text().starts_with("invalid json:"));
        aria_model_destroy(model);
    }

    #[test]
    fn systemone_encoder_success_writes_json() {
        let model = encoder_model(vec![0.1, 3.0]);
        let req = choice_request();
        let mut buf = vec![0i8; 512];
        assert_eq!(
            aria_systemone(model, req.as_ptr(), buf.as_mut_ptr(), buf.len()),
            0
        );
        let text = unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let parsed: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["answers"]["q1"]["choice"], "b");
        aria_model_destroy(model);
    }

    #[test]
    fn systemone_reports_small_buffer() {
        let model = encoder_model(vec![0.1, 3.0]);
        let req = choice_request();
        let mut buf = vec![0i8; 4];
        assert_eq!(
            aria_systemone(model, req.as_ptr(), buf.as_mut_ptr(), buf.len()),
            -2
        );
        assert!(last_error_text().contains("output buffer too small"));
        aria_model_destroy(model);
    }

    #[test]
    fn systemone_decoder_unwraps_systemone_block() {
        let model = decoder_model();
        let req = choice_request();
        let mut buf = vec![0i8; 512];
        assert_eq!(
            aria_systemone(model, req.as_ptr(), buf.as_mut_ptr(), buf.len()),
            0
        );
        let text = unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let parsed: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["answers"]["q1"]["choice"], "b");
        aria_model_destroy(model);
    }

    #[test]
    fn systemone_propagates_scoring_errors() {
        // one logit for two options → logits/options mismatch inside the scorer
        let model = encoder_model(vec![1.0]);
        let req = choice_request();
        let mut buf = vec![0i8; 256];
        assert_eq!(
            aria_systemone(model, req.as_ptr(), buf.as_mut_ptr(), buf.len()),
            -1
        );
        assert!(last_error_text().contains("logits len 1 != options 2"));
        aria_model_destroy(model);
    }

    #[test]
    fn set_error_strips_interior_nul() {
        set_error("bad\0news");
        assert_eq!(last_error_text(), "badnews");
    }

    #[test]
    fn error_slot_is_per_thread_and_starts_empty() {
        std::thread::spawn(|| {
            assert!(aria_last_error().is_null());
            assert_eq!(last_error_text(), "");
        })
        .join()
        .unwrap();
    }
}
