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
