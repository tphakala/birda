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
