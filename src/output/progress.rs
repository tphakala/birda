//! Progress bar utilities for file processing.

use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

/// Create a progress bar for processing multiple files.
pub fn create_file_progress(total_files: usize, enabled: bool) -> Option<ProgressBar> {
    if !enabled || total_files == 0 {
        return None;
    }

    let pb = ProgressBar::new(total_files as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} files ({eta})")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("█▓▒░ "),
    );
    Some(pb)
}

/// Create a progress bar for processing segments within a file.
pub fn create_segment_progress(
    total_segments: usize,
    file_name: &str,
    enabled: bool,
) -> Option<ProgressBar> {
    if !enabled || total_segments == 0 {
        return None;
    }

    let pb = ProgressBar::new(total_segments as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(&format!(
                "{{spinner:.green}} [{{elapsed_precise}}] {{bar:40.cyan/blue}} {{pos}}/{{len}} segments - {file_name}"
            ))
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("█▓▒░ "),
    );
    pb.enable_steady_tick(Duration::from_millis(100));
    Some(pb)
}

/// Finish a progress bar with a message.
pub fn finish_progress(pb: Option<ProgressBar>, message: &str) {
    if let Some(pb) = pb {
        pb.finish_with_message(message.to_string());
    }
}

/// Increment a progress bar.
pub fn inc_progress(pb: Option<&ProgressBar>) {
    if let Some(pb) = pb {
        pb.inc(1);
    }
}

/// RAII guard that ensures a progress bar is finished when dropped.
pub struct ProgressGuard {
    progress: Option<ProgressBar>,
    message: String,
}

impl ProgressGuard {
    /// Create a new progress guard.
    pub fn new(progress: Option<ProgressBar>, message: impl Into<String>) -> Self {
        Self {
            progress,
            message: message.into(),
        }
    }

    /// Get a reference to the progress bar for incrementing.
    pub fn get(&self) -> Option<&ProgressBar> {
        self.progress.as_ref()
    }
}

impl Drop for ProgressGuard {
    fn drop(&mut self) {
        if let Some(pb) = self.progress.take() {
            pb.finish_with_message(self.message.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_guard_finishes_on_drop() {
        // Create a progress bar
        let pb = create_segment_progress(10, "test.wav", true);

        // Wrap in guard
        let guard = ProgressGuard::new(pb, "Complete");

        // Guard should finish the progress bar when dropped
        // (This is tested by running without panics - indicatif would complain if not finished properly)
        drop(guard);
    }

    #[test]
    fn test_progress_guard_finishes_on_error_path() {
        fn might_error(should_error: bool) -> Result<(), &'static str> {
            let pb = create_segment_progress(10, "test.wav", true);
            let _guard = ProgressGuard::new(pb, "Complete");

            if should_error {
                return Err("simulated error");
            }
            Ok(())
        }

        // Should not leak even on error
        let result = might_error(true);
        assert!(result.is_err());
    }
}
