//! GPU utilities for inference safety.
//!
//! This module provides the inference watchdog timer that kills the process
//! if inference takes too long, indicating likely GPU memory exhaustion.

mod watchdog;

pub use watchdog::{WatchdogGuard, start_inference_watchdog};
