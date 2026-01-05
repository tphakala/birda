//! Inference watchdog timer for detecting GPU hangs.
//!
//! Provides a watchdog that kills the process if inference takes too long,
//! which typically indicates GPU memory exhaustion causing the system to hang.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

/// Start a watchdog timer that kills the process if inference exceeds the deadline.
///
/// Returns a guard that must be dropped when inference completes successfully.
/// If the guard is not dropped before the timeout, the process exits with an error.
///
/// # Arguments
/// * `timeout` - Maximum time allowed for the operation
/// * `batch_size` - Batch size being processed (for error message)
///
/// # Returns
/// A `WatchdogGuard` that cancels the watchdog when dropped.
pub fn start_inference_watchdog(timeout: Duration, batch_size: usize) -> WatchdogGuard {
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancelled_clone = Arc::clone(&cancelled);
    let timeout_secs = timeout.as_secs();

    thread::spawn(move || {
        thread::sleep(timeout);

        if !cancelled_clone.load(Ordering::SeqCst) {
            // Watchdog fired - inference didn't complete in time
            eprintln!();
            eprintln!("═══════════════════════════════════════════════════════════════");
            eprintln!("FATAL: Inference timeout after {timeout_secs}s (batch size: {batch_size})");
            eprintln!("═══════════════════════════════════════════════════════════════");
            eprintln!();
            eprintln!("The GPU inference operation did not complete within the expected time.");
            eprintln!("This usually indicates GPU memory exhaustion causing the system to hang.");
            eprintln!();
            eprintln!("Recommendations:");
            eprintln!("  1. Reduce batch size with -b flag");
            eprintln!("  2. Use CPU inference: birda --cpu <input>");
            eprintln!("  3. Close other GPU applications and try again");
            eprintln!();
            eprintln!("Terminating process to prevent system lockup.");
            std::process::exit(1);
        }
    });

    WatchdogGuard { cancelled }
}

/// Guard that cancels the watchdog timer when dropped.
///
/// Create this with [`start_inference_watchdog`] before inference,
/// and let it drop naturally after inference completes.
pub struct WatchdogGuard {
    cancelled: Arc<AtomicBool>,
}

impl Drop for WatchdogGuard {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watchdog_cancelled_when_dropped() {
        // Create watchdog with 1 second timeout
        let guard = start_inference_watchdog(Duration::from_secs(1), 32);

        // Drop immediately - should cancel the watchdog
        drop(guard);

        // Sleep past the timeout - process should NOT exit
        thread::sleep(Duration::from_millis(1500));

        // If we get here, the test passed (process didn't exit)
    }

    #[test]
    fn test_watchdog_guard_is_send() {
        // Ensure WatchdogGuard can be sent across threads
        fn assert_send<T: Send>() {}
        assert_send::<WatchdogGuard>();
    }
}
