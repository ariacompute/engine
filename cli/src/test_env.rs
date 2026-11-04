//! Test-only helpers for process-global environment state.
//!
//! `ARIA_COMPUTE_HOME` is read by `ariacompute_core::config`, so every test that
//! points it at a throwaway directory must hold [`lock_env`] for the whole span of
//! the override. Without the lock a parallel test that clears the variable makes a
//! sibling test write into the developer's real `~/.ariacompute`.

use std::path::Path;
use std::sync::{Mutex, MutexGuard};

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Serialize access to the process-global env var used by these tests.
pub(crate) fn lock_env() -> MutexGuard<'static, ()> {
    // A panicking test poisons the mutex; the guard is still usable afterwards and
    // serializing is what matters, so recover instead of cascading failures.
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Run `f` with `ARIA_COMPUTE_HOME` pointing at a throwaway home.
pub(crate) fn with_home<T>(f: impl FnOnce(&Path) -> T) -> T {
    let _guard = lock_env();
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("ARIA_COMPUTE_HOME", dir.path());
    let out = f(dir.path());
    std::env::remove_var("ARIA_COMPUTE_HOME");
    out
}
