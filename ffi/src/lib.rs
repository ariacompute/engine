//! C ABI for Aria inference (chat / embeddings / ASR / tools).

#![allow(clippy::not_unsafe_ptr_arg_deref)] // C ABI: pointers are caller-owned
#![allow(clippy::too_many_arguments)]

use aria_inference::{ChatTurn, GenerateOpts, Session, SessionBuilder};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uchar, c_void};
use std::ptr;
use std::slice;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

pub struct AriaModel {
    session: Session,
}

fn set_error(msg: impl Into<String>) {
    let s = CString::new(msg.into()).unwrap_or_else(|_| CString::new("error").unwrap());
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(s));
}

fn clear_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

fn cstr_to_str<'a>(p: *const c_char) -> Result<&'a str, String> {
    if p.is_null() {
        return Err("null string".into());
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|e| e.to_string())
}

fn write_out(out: *mut c_char, out_len: usize, s: &str) -> c_int {
    if out.is_null() || out_len == 0 {
        set_error("null output buffer");
        return -1;
    }
    let bytes = s.as_bytes();
    if bytes.len() + 1 > out_len {
        set_error(format!(
            "output buffer too small: need {}, have {}",
            bytes.len() + 1,
            out_len
        ));
        return -1;
    }
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), out.cast::<u8>(), bytes.len());
        *out.add(bytes.len()) = 0;
    }
    0
}

fn parse_messages(messages_json: &str) -> Result<Vec<ChatTurn>, String> {
    let v: Value = serde_json::from_str(messages_json).map_err(|e| e.to_string())?;
    let arr = v
        .as_array()
        .ok_or_else(|| "messages must be a JSON array".to_string())?;
    let mut turns = Vec::new();
    for m in arr {
        let role = m
            .get("role")
            .and_then(|x| x.as_str())
            .unwrap_or("user")
            .to_string();
        let content = m
            .get("content")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        turns.push(ChatTurn { role, content });
    }
    Ok(turns)
}

fn parse_options(options_json: Option<&str>) -> GenerateOpts {
    let mut opts = GenerateOpts::default();
    if let Some(raw) = options_json {
        if let Ok(v) = serde_json::from_str::<Value>(raw) {
            if let Some(n) = v.get("max_tokens").and_then(|x| x.as_u64()) {
                opts.max_tokens = n as usize;
            }
            if let Some(t) = v.get("temperature").and_then(|x| x.as_f64()) {
                opts.temperature = t as f32;
            }
        }
    }
    if opts.max_tokens == 0 {
        opts.max_tokens = 16;
    }
    opts
}

fn parse_tools(tools_json: Option<&str>) -> Result<Value, String> {
    match tools_json {
        None | Some("") => Ok(json!([])),
        Some(raw) => {
            let v: Value = serde_json::from_str(raw).map_err(|e| e.to_string())?;
            if !v.is_array() && !v.is_null() {
                return Err("tools must be a JSON array or null".into());
            }
            Ok(if v.is_null() { json!([]) } else { v })
        }
    }
}

/// Opaque model handle.
pub type AriaModelHandle = *mut AriaModel;

/// Last error message (thread-local). Valid until next call on this thread.
#[no_mangle]
pub extern "C" fn aria_last_error() -> *const c_char {
    LAST_ERROR.with(|e| match e.borrow().as_ref() {
        Some(s) => s.as_ptr(),
        None => ptr::null(),
    })
}

/// Load an Aria quant bundle from `bundle_path`. Returns null on error.
#[no_mangle]
pub extern "C" fn aria_model_init(bundle_path: *const c_char) -> AriaModelHandle {
    clear_error();
    let path = match cstr_to_str(bundle_path) {
        Ok(p) => p,
        Err(e) => {
            set_error(e);
            return ptr::null_mut();
        }
    };
    match SessionBuilder::new().model(path).build() {
        Ok(session) => Box::into_raw(Box::new(AriaModel { session })),
        Err(e) => {
            set_error(e.to_string());
            ptr::null_mut()
        }
    }
}

/// Destroy a model handle. Safe on null. Double-destroy is undefined if caller reuses the pointer.
#[no_mangle]
pub extern "C" fn aria_model_destroy(model: AriaModelHandle) {
    clear_error();
    if model.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(model));
    }
}

fn complete_inner(
    model: AriaModelHandle,
    messages_json: *const c_char,
    options_json: *const c_char,
    tools_json: *const c_char,
    out: *mut c_char,
    out_len: usize,
    stream_cb: Option<unsafe extern "C" fn(*const c_char, *mut c_void)>,
    user_data: *mut c_void,
) -> c_int {
    clear_error();
    if model.is_null() {
        set_error("null model");
        return -1;
    }
    let messages = match cstr_to_str(messages_json) {
        Ok(s) => s,
        Err(e) => {
            set_error(e);
            return -1;
        }
    };
    let options = if options_json.is_null() {
        None
    } else {
        match cstr_to_str(options_json) {
            Ok(s) => Some(s),
            Err(e) => {
                set_error(e);
                return -1;
            }
        }
    };
    let tools_raw = if tools_json.is_null() {
        None
    } else {
        match cstr_to_str(tools_json) {
            Ok(s) => Some(s),
            Err(e) => {
                set_error(e);
                return -1;
            }
        }
    };

    let turns = match parse_messages(messages) {
        Ok(p) => p,
        Err(e) => {
            set_error(e);
            return -1;
        }
    };
    let tools = match parse_tools(tools_raw) {
        Ok(t) => t,
        Err(e) => {
            set_error(e);
            return -1;
        }
    };
    let opts = parse_options(options);
    let m = unsafe { &mut *model };
    let tokens = m.session.encode_chat(&turns);
    let gen = match m.session.generate(&tokens, &opts) {
        Ok(g) => g,
        Err(e) => {
            set_error(e.to_string());
            return -1;
        }
    };

    if let Some(cb) = stream_cb {
        // Stream decoded text (same as gen.text), not raw `<id>` placeholders.
        if let Ok(c) = CString::new(gen.text.as_str()) {
            unsafe { cb(c.as_ptr(), user_data) };
        }
    }

    let body = json!({
        "success": true,
        "error": null,
        "response": gen.text,
        "function_calls": json!([]),
        "segments": [],
        "cloud_handoff": false,
        "total_tokens": gen.tokens.len(),
    });
    // tools accepted for OpenAI parity; real tool routing is stage C.
    let _ = tools;
    write_out(out, out_len, &body.to_string())
}

/// Non-streaming chat completion. `tools_json` may be null.
#[no_mangle]
pub extern "C" fn aria_complete(
    model: AriaModelHandle,
    messages_json: *const c_char,
    options_json: *const c_char,
    tools_json: *const c_char,
    out: *mut c_char,
    out_len: usize,
) -> c_int {
    complete_inner(
        model,
        messages_json,
        options_json,
        tools_json,
        out,
        out_len,
        None,
        ptr::null_mut(),
    )
}

/// Streaming chat; `callback` receives each chunk as a C string.
#[no_mangle]
pub extern "C" fn aria_complete_stream(
    model: AriaModelHandle,
    messages_json: *const c_char,
    options_json: *const c_char,
    tools_json: *const c_char,
    out: *mut c_char,
    out_len: usize,
    callback: Option<unsafe extern "C" fn(*const c_char, *mut c_void)>,
    user_data: *mut c_void,
) -> c_int {
    complete_inner(
        model,
        messages_json,
        options_json,
        tools_json,
        out,
        out_len,
        callback,
        user_data,
    )
}

/// Embeddings. `input_json` is a string or `{"input":"..."}` / `{"input":[...]}`.
#[no_mangle]
pub extern "C" fn aria_embed(
    model: AriaModelHandle,
    input_json: *const c_char,
    out: *mut c_char,
    out_len: usize,
) -> c_int {
    clear_error();
    if model.is_null() {
        set_error("null model");
        return -1;
    }
    let raw = match cstr_to_str(input_json) {
        Ok(s) => s,
        Err(e) => {
            set_error(e);
            return -1;
        }
    };
    let text = match serde_json::from_str::<Value>(raw) {
        Ok(Value::String(s)) => s,
        Ok(v) => v
            .get("input")
            .and_then(|x| {
                x.as_str()
                    .map(|s| s.to_string())
                    .or_else(|| x.as_array().and_then(|a| a.first()).and_then(|x| x.as_str()).map(|s| s.to_string()))
            })
            .unwrap_or_default(),
        Err(_) => raw.to_string(),
    };
    if text.is_empty() {
        set_error("empty embedding input");
        return -1;
    }
    let m = unsafe { &*model };
    let emb = match m.session.embed_text(&text) {
        Ok(e) => e,
        Err(e) => {
            set_error(e.to_string());
            return -1;
        }
    };
    let body = json!({
        "object": "list",
        "data": [{
            "object": "embedding",
            "embedding": emb,
            "index": 0
        }]
    });
    write_out(out, out_len, &body.to_string())
}

/// Transcribe PCM16 LE bytes.
#[no_mangle]
pub extern "C" fn aria_transcribe(
    model: AriaModelHandle,
    pcm: *const c_uchar,
    pcm_len: usize,
    _options_json: *const c_char,
    out: *mut c_char,
    out_len: usize,
) -> c_int {
    clear_error();
    if model.is_null() {
        set_error("null model");
        return -1;
    }
    if pcm.is_null() || pcm_len == 0 {
        set_error("empty pcm");
        return -1;
    }
    let bytes = unsafe { slice::from_raw_parts(pcm, pcm_len) };
    let m = unsafe { &*model };
    let text = match m.session.transcribe_pcm16le(bytes) {
        Ok(t) => t,
        Err(e) => {
            set_error(e.to_string());
            return -1;
        }
    };
    let body = json!({
        "text": text,
        "segments": []
    });
    write_out(out, out_len, &body.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aria_inference::fixture::write_tiny_q4_bundle;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn out_buf() -> Vec<u8> {
        vec![0u8; 64 * 1024]
    }

    #[test]
    fn init_complete_embed_transcribe_destroy() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let path = CString::new(dir.path().to_str().unwrap()).unwrap();
        let model = aria_model_init(path.as_ptr());
        assert!(!model.is_null(), "{:?}", unsafe {
            CStr::from_ptr(aria_last_error()).to_string_lossy()
        });

        let messages = CString::new(r#"[{"role":"user","content":"hi"}]"#).unwrap();
        let options = CString::new(r#"{"max_tokens":2}"#).unwrap();
        let tools = CString::new("[]").unwrap();
        let mut buf = out_buf();
        assert_eq!(
            aria_complete(
                model,
                messages.as_ptr(),
                options.as_ptr(),
                tools.as_ptr(),
                buf.as_mut_ptr() as *mut c_char,
                buf.len(),
            ),
            0
        );
        let s = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) }
            .to_str()
            .unwrap();
        let v: Value = serde_json::from_str(s).unwrap();
        assert_eq!(v["success"], true);
        assert!(!v["response"].as_str().unwrap().is_empty());

        let input = CString::new(r#"{"input":"hello"}"#).unwrap();
        buf.fill(0);
        assert_eq!(
            aria_embed(
                model,
                input.as_ptr(),
                buf.as_mut_ptr() as *mut c_char,
                buf.len()
            ),
            0
        );

        let pcm = [0u8, 1, 2, 3, 4, 5];
        buf.fill(0);
        assert_eq!(
            aria_transcribe(
                model,
                pcm.as_ptr(),
                pcm.len(),
                ptr::null(),
                buf.as_mut_ptr() as *mut c_char,
                buf.len()
            ),
            0
        );

        aria_model_destroy(model);
    }

    #[test]
    fn init_missing_path() {
        let path = CString::new("/no/such/bundle").unwrap();
        let model = aria_model_init(path.as_ptr());
        assert!(model.is_null());
        assert!(!aria_last_error().is_null());
    }

    #[test]
    fn complete_bad_json() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let path = CString::new(dir.path().to_str().unwrap()).unwrap();
        let model = aria_model_init(path.as_ptr());
        let bad = CString::new("not-json").unwrap();
        let mut buf = out_buf();
        assert_ne!(
            aria_complete(
                model,
                bad.as_ptr(),
                ptr::null(),
                ptr::null(),
                buf.as_mut_ptr() as *mut c_char,
                buf.len()
            ),
            0
        );
        aria_model_destroy(model);
    }

    static CHUNKS: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn on_chunk(_s: *const c_char, _ud: *mut c_void) {
        CHUNKS.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn complete_stream_ok() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let path = CString::new(dir.path().to_str().unwrap()).unwrap();
        let model = aria_model_init(path.as_ptr());
        let messages = CString::new(r#"[{"role":"user","content":"hi"}]"#).unwrap();
        let options = CString::new(r#"{"max_tokens":2}"#).unwrap();
        let mut buf = out_buf();
        CHUNKS.store(0, Ordering::SeqCst);
        assert_eq!(
            aria_complete_stream(
                model,
                messages.as_ptr(),
                options.as_ptr(),
                ptr::null(),
                buf.as_mut_ptr() as *mut c_char,
                buf.len(),
                Some(on_chunk),
                ptr::null_mut(),
            ),
            0
        );
        assert!(CHUNKS.load(Ordering::SeqCst) >= 1);
        aria_model_destroy(model);
    }

    #[test]
    fn destroy_null_and_use_after_destroy() {
        aria_model_destroy(ptr::null_mut());
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let path = CString::new(dir.path().to_str().unwrap()).unwrap();
        let model = aria_model_init(path.as_ptr());
        aria_model_destroy(model);
        let messages = CString::new(r#"[{"role":"user","content":"hi"}]"#).unwrap();
        let mut buf = out_buf();
        // After destroy the pointer must not be reused by callers; we only check null path.
        assert_ne!(
            aria_complete(
                ptr::null_mut(),
                messages.as_ptr(),
                ptr::null(),
                ptr::null(),
                buf.as_mut_ptr() as *mut c_char,
                buf.len()
            ),
            0
        );
    }
}
