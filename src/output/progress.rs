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

    // Sanitize filename to prevent template injection (remove curly braces)
    let safe_name = file_name.replace(['{', '}'], "");

    let pb = ProgressBar::new(total_segments as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(&format!(
                "{{spinner:.green}} [{{elapsed_precise}}] {{bar:40.cyan/blue}} {{pos}}/{{len}} segments - {safe_name}"
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

/// Format seconds as HH:MM:SS.
pub fn format_duration(secs: f32) -> String {
    const SECONDS_PER_HOUR: u32 = 3600;
    const SECONDS_PER_MINUTE: u32 = 60;

    debug_assert!(secs >= 0.0, "Audio duration should not be negative: {secs}");

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let total_secs = secs.abs() as u32;
    let hours = total_secs / SECONDS_PER_HOUR;
    let mins = (total_secs % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE;
    let secs_remainder = total_secs % SECONDS_PER_MINUTE;
    format!("{hours:02}:{mins:02}:{secs_remainder:02}")
}

/// RAII guard that ensures a progress bar is finished when dropped.
pub struct ProgressGuard {
    progress: Option<ProgressBar>,
}

impl ProgressGuard {
    /// Create a new progress guard.
    pub fn new(progress: Option<ProgressBar>) -> Self {
        Self { progress }
    }

    /// Get a reference to the progress bar for incrementing.
    pub fn get(&self) -> Option<&ProgressBar> {
        self.progress.as_ref()
    }
}

impl Drop for ProgressGuard {
    fn drop(&mut self) {
        if let Some(pb) = self.progress.take() {
            // Set to 100% completion
            if let Some(len) = pb.length() {
                pb.set_position(len);
                // Give the progress bar time to render the final state
                // before we clear it. This ensures users see 100% completion.
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            // Now finish and clear to avoid duplication
            pb.finish_and_clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(0.0), "00:00:00");
        assert_eq!(format_duration(59.0), "00:00:59");
        assert_eq!(format_duration(60.0), "00:01:00");
        assert_eq!(format_duration(3661.0), "01:01:01");
        assert_eq!(format_duration(44589.0), "12:23:09");
        assert_eq!(format_duration(86399.0), "23:59:59");
        assert_eq!(format_duration(86400.0), "24:00:00");
    }

    #[test]
    fn test_progress_guard_finishes_on_drop() {
        // Create a progress bar
        let pb = create_segment_progress(10, "test.wav", true);

        // Wrap in guard
        let guard = ProgressGuard::new(pb);

        // Guard should finish the progress bar when dropped
        // (This is tested by running without panics - indicatif would complain if not finished properly)
        drop(guard);
    }

    #[test]
    fn test_progress_guard_finishes_on_error_path() {
        fn might_error(should_error: bool) -> Result<(), &'static str> {
            let pb = create_segment_progress(10, "test.wav", true);
            let _guard = ProgressGuard::new(pb);

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
