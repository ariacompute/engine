//! Backend health scores + fallback chain (P2).
//!
//! Each backend starts at 1.0. Success/failure/timeout events move the score;
//! a backend is `healthy` while its score is >= [`HEALTHY_THRESHOLD`]. The
//! hybrid router consults these scores to flip soft decisions onto a healthy
//! alternate backend; hard constraints are never flipped.

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

/// Score at/above which a backend is considered healthy.
pub const HEALTHY_THRESHOLD: f32 = 0.5;

const SUCCESS_DELTA: f32 = 0.05;
const FAILURE_DELTA: f32 = -0.20;
const TIMEOUT_DELTA: f32 = -0.10;

/// Routable backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Local,
    Cloud,
}

/// Execution feedback for one backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthEvent {
    Success,
    Failure,
    Timeout,
}

/// Point-in-time health view (observability endpoint).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HealthSnapshot {
    pub local: f32,
    pub cloud: f32,
}

#[derive(Debug)]
struct HealthState {
    local: f32,
    cloud: f32,
}

/// Thread-safe health tracker (same Arc<Mutex> pattern as OutcomeStore).
#[derive(Debug, Clone)]
pub struct HealthTracker {
    inner: Arc<Mutex<HealthState>>,
}

impl Default for HealthTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthTracker {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HealthState {
                local: 1.0,
                cloud: 1.0,
            })),
        }
    }

    /// Apply one execution event; score clamped to `[0, 1]`.
    pub fn record(&self, backend: BackendKind, event: HealthEvent) {
        let delta = match event {
            HealthEvent::Success => SUCCESS_DELTA,
            HealthEvent::Failure => FAILURE_DELTA,
            HealthEvent::Timeout => TIMEOUT_DELTA,
        };
        let mut guard = self.inner.lock().expect("health tracker poisoned");
        let slot = match backend {
            BackendKind::Local => &mut guard.local,
            BackendKind::Cloud => &mut guard.cloud,
        };
        *slot = (*slot + delta).clamp(0.0, 1.0);
    }

    pub fn score(&self, backend: BackendKind) -> f32 {
        let guard = self.inner.lock().expect("health tracker poisoned");
        match backend {
            BackendKind::Local => guard.local,
            BackendKind::Cloud => guard.cloud,
        }
    }

    pub fn healthy(&self, backend: BackendKind) -> bool {
        self.score(backend) >= HEALTHY_THRESHOLD
    }

    pub fn snapshot(&self) -> HealthSnapshot {
        let guard = self.inner.lock().expect("health tracker poisoned");
        HealthSnapshot {
            local: guard.local,
            cloud: guard.cloud,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_tracker_is_fully_healthy() {
        let h = HealthTracker::new();
        assert!(h.healthy(BackendKind::Local));
        assert!(h.healthy(BackendKind::Cloud));
        let snap = h.snapshot();
        assert!((snap.local - 1.0).abs() < 1e-6);
        assert!((snap.cloud - 1.0).abs() < 1e-6);
    }

    #[test]
    fn success_caps_at_one() {
        let h = HealthTracker::new();
        for _ in 0..10 {
            h.record(BackendKind::Local, HealthEvent::Success);
        }
        assert!((h.score(BackendKind::Local) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn failures_drop_below_threshold_and_bottom_out() {
        let h = HealthTracker::new();
        // 3 failures: 1.0 - 0.6 = 0.4 < 0.5 → unhealthy.
        for _ in 0..3 {
            h.record(BackendKind::Cloud, HealthEvent::Failure);
        }
        assert!(!h.healthy(BackendKind::Cloud));
        assert!((h.score(BackendKind::Cloud) - 0.4).abs() < 1e-6);
        // 10 more failures bottom out at 0.0, never negative.
        for _ in 0..10 {
            h.record(BackendKind::Cloud, HealthEvent::Failure);
        }
        assert!((h.score(BackendKind::Cloud)).abs() < 1e-6);
        // Local backend untouched.
        assert!(h.healthy(BackendKind::Local));
    }

    #[test]
    fn timeouts_weigh_less_than_failures() {
        let h = HealthTracker::new();
        // 3 timeouts: 1.0 - 0.3 = 0.7 ≥ 0.5 → still healthy.
        for _ in 0..3 {
            h.record(BackendKind::Cloud, HealthEvent::Timeout);
        }
        assert!(h.healthy(BackendKind::Cloud));
        assert!((h.score(BackendKind::Cloud) - 0.7).abs() < 1e-6);
        // 3 more: 0.4 → unhealthy.
        for _ in 0..3 {
            h.record(BackendKind::Cloud, HealthEvent::Timeout);
        }
        assert!(!h.healthy(BackendKind::Cloud));
    }

    #[test]
    fn recovery_via_successes() {
        let h = HealthTracker::new();
        for _ in 0..4 {
            h.record(BackendKind::Local, HealthEvent::Failure);
        }
        assert!(!h.healthy(BackendKind::Local));
        // 0.2 + 7*0.05 = 0.55 → healthy again.
        for _ in 0..7 {
            h.record(BackendKind::Local, HealthEvent::Success);
        }
        assert!(h.healthy(BackendKind::Local));
    }

    #[test]
    fn snapshot_reflects_both_backends() {
        let h = HealthTracker::new();
        h.record(BackendKind::Local, HealthEvent::Failure);
        h.record(BackendKind::Cloud, HealthEvent::Timeout);
        let snap = h.snapshot();
        assert!((snap.local - 0.8).abs() < 1e-6);
        assert!((snap.cloud - 0.9).abs() < 1e-6);
        let v = serde_json::to_value(snap).unwrap();
        assert!(v["local"].is_number());
        assert!(v["cloud"].is_number());
    }
}
