//! Progress reporting infrastructure for CLI output.
//!
//! This module provides a trait for progress reporting and implementations
//! for different output modes (human-readable, JSON, NDJSON).

use crate::config::OutputMode;
use crate::output::json_envelope::{
    BatchProgress, CancelReason, CancelledPayload, ErrorPayload, ErrorSeverity, EventType,
    FileCompletedPayload, FileErrorInfo, FileProgress, FileStartedPayload, FileStatus,
    JsonEnvelope, PipelineCompletedPayload, PipelineStartedPayload, PipelineStatus,
    ProgressPayload,
};
use std::io::{self, Write};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

/// Trait for reporting progress during pipeline operations.
///
/// Implementations can output to different formats (human, JSON, NDJSON).
pub trait ProgressReporter: Send + Sync {
    /// Report pipeline start.
    fn pipeline_started(&self, total_files: usize, model: &str, min_confidence: f32);

    /// Report file processing start.
    fn file_started(
        &self,
        file: &Path,
        index: usize,
        estimated_segments: usize,
        duration_seconds: Option<f64>,
    );

    /// Report progress update.
    fn progress(&self, batch: Option<&BatchProgress>, file: Option<&FileProgress>);

    /// Report file completion (success).
    fn file_completed_success(&self, file: &Path, detections: usize, duration_ms: u64);

    /// Report file completion (failure).
    fn file_completed_failure(&self, file: &Path, error_code: &str, error_message: &str);

    /// Report file skipped.
    fn file_skipped(&self, file: &Path, reason: FileStatus);

    /// Report pipeline completion.
    fn pipeline_completed(&self, summary: &PipelineSummary);

    /// Report an error.
    fn error(&self, code: &str, severity: ErrorSeverity, message: &str, suggestion: Option<&str>);

    /// Report cancellation.
    fn cancelled(&self, reason: CancelReason, files_completed: usize, files_total: usize);

    /// Report detection results.
    fn detections(&self, file: &Path, detections: &[crate::output::Detection]);
}

/// Summary of pipeline execution.
#[derive(Debug, Clone)]
pub struct PipelineSummary {
    /// Files successfully processed.
    pub files_processed: usize,
    /// Files that failed.
    pub files_failed: usize,
    /// Files skipped.
    pub files_skipped: usize,
    /// Total detections.
    pub total_detections: usize,
    /// Total segments processed.
    pub total_segments: usize,
    /// Total duration in milliseconds.
    pub duration_ms: u64,
    /// Realtime processing factor.
    pub realtime_factor: f64,
}

/// Progress throttler to limit update frequency.
pub struct ProgressThrottler {
    /// Last reported percentage (0-100).
    last_percent: AtomicU8,
    /// Last update time.
    last_update: Mutex<Instant>,
    /// Minimum time between updates in milliseconds.
    min_interval_ms: u64,
    /// Minimum percentage change between updates.
    min_percent_change: u8,
}

impl ProgressThrottler {
    /// Create a new throttler with default settings (10%, 500ms).
    pub fn new() -> Self {
        Self {
            last_percent: AtomicU8::new(0),
            last_update: Mutex::new(Instant::now()),
            min_interval_ms: 500,
            min_percent_change: 10,
        }
    }

    /// Check if an update should be emitted.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn should_emit(&self, current_percent: f32) -> bool {
        // Clamp to 0-100 range before converting
        // Use floor() so 100% only shows when truly complete
        let clamped = current_percent.floor().clamp(0.0, 100.0);
        let current = clamped as u8;
        let last = self.last_percent.load(Ordering::Relaxed);

        // Always emit at 0% and 100%
        if current == 0 || current >= 100 {
            self.last_percent.store(current, Ordering::Relaxed);
            if let Ok(mut last_update) = self.last_update.lock() {
                *last_update = Instant::now();
            }
            return true;
        }

        // Check percentage threshold
        let percent_changed = current.saturating_sub(last) >= self.min_percent_change;

        // Check time threshold
        let time_elapsed = self
            .last_update
            .lock()
            .map(|last| last.elapsed().as_millis() >= u128::from(self.min_interval_ms))
            .unwrap_or(true);

        if percent_changed || time_elapsed {
            self.last_percent.store(current, Ordering::Relaxed);
            if let Ok(mut last_update) = self.last_update.lock() {
                *last_update = Instant::now();
            }
            true
        } else {
            false
        }
    }

    /// Reset the throttler for a new file.
    pub fn reset(&self) {
        self.last_percent.store(0, Ordering::Relaxed);
        if let Ok(mut last_update) = self.last_update.lock() {
            *last_update = Instant::now();
        }
    }
}

impl Default for ProgressThrottler {
    fn default() -> Self {
        Self::new()
    }
}

/// JSON/NDJSON progress reporter implementation.
pub struct JsonProgressReporter {
    /// Output mode (Json or Ndjson).
    mode: OutputMode,
    /// Progress throttler.
    throttler: ProgressThrottler,
    /// Writer for output (typically stdout).
    writer: Mutex<Box<dyn Write + Send>>,
    /// Buffer for JSON mode (collect all events).
    json_buffer: Mutex<Vec<String>>,
}

impl JsonProgressReporter {
    /// Create a new JSON progress reporter.
    pub fn new(mode: OutputMode) -> Self {
        Self {
            mode,
            throttler: ProgressThrottler::new(),
            writer: Mutex::new(Box::new(io::stdout())),
            json_buffer: Mutex::new(Vec::new()),
        }
    }

    /// Create a reporter with a custom writer (for testing).
    #[cfg(test)]
    pub fn with_writer<W: Write + Send + 'static>(mode: OutputMode, writer: W) -> Self {
        Self {
            mode,
            throttler: ProgressThrottler::new(),
            writer: Mutex::new(Box::new(writer)),
            json_buffer: Mutex::new(Vec::new()),
        }
    }

    /// Emit an event as JSON.
    fn emit<T: serde::Serialize>(&self, event: EventType, payload: T) {
        let envelope = JsonEnvelope::new(event, payload);
        if let Ok(json) = serde_json::to_string(&envelope) {
            match self.mode {
                OutputMode::Ndjson => {
                    // Write directly to stdout
                    if let Ok(mut writer) = self.writer.lock() {
                        if let Err(e) = writeln!(writer, "{json}") {
                            // Log first error only to avoid spam on broken pipe
                            use std::sync::atomic::{AtomicBool, Ordering};
                            static STDOUT_ERROR_LOGGED: AtomicBool = AtomicBool::new(false);
                            if !STDOUT_ERROR_LOGGED.swap(true, Ordering::Relaxed) {
                                eprintln!(
                                    "birda: warning: failed to write to stdout: {e} (subsequent errors suppressed)"
                                );
                            }
                        }
                        // Flush errors are less critical - silent ignore is OK
                        let _ = writer.flush();
                    }
                }
                OutputMode::Json => {
                    // Buffer for final output
                    if let Ok(mut buffer) = self.json_buffer.lock() {
                        buffer.push(json);
                    }
                }
                OutputMode::Human => {
                    // Should not be used with this reporter
                }
            }
        }
    }

    /// Flush buffered JSON output (for Json mode).
    pub fn flush(&self) {
        if self.mode == OutputMode::Json
            && let Ok(buffer) = self.json_buffer.lock()
            && let Ok(mut writer) = self.writer.lock()
        {
            // Output as JSON array
            let _ = writeln!(writer, "[");
            for (i, json) in buffer.iter().enumerate() {
                if i > 0 {
                    let _ = writeln!(writer, ",");
                }
                let _ = write!(writer, "  {json}");
            }
            let _ = writeln!(writer);
            let _ = writeln!(writer, "]");
            let _ = writer.flush();
        }
    }
}

impl ProgressReporter for JsonProgressReporter {
    fn pipeline_started(&self, total_files: usize, model: &str, min_confidence: f32) {
        self.emit(
            EventType::PipelineStarted,
            PipelineStartedPayload {
                total_files,
                model: model.to_string(),
                min_confidence,
            },
        );
    }

    fn file_started(
        &self,
        file: &Path,
        index: usize,
        estimated_segments: usize,
        duration_seconds: Option<f64>,
    ) {
        self.throttler.reset();
        self.emit(
            EventType::FileStarted,
            FileStartedPayload {
                file: file.to_path_buf(),
                index,
                estimated_segments,
                duration_seconds,
            },
        );
    }

    fn progress(&self, batch: Option<&BatchProgress>, file: Option<&FileProgress>) {
        // Check throttling based on file progress
        let should_emit = file.is_none_or(|f| self.throttler.should_emit(f.percent));

        if should_emit {
            self.emit(
                EventType::Progress,
                ProgressPayload {
                    batch: batch.cloned(),
                    file: file.cloned(),
                    download: None,
                },
            );
        }
    }

    fn file_completed_success(&self, file: &Path, detections: usize, duration_ms: u64) {
        self.emit(
            EventType::FileCompleted,
            FileCompletedPayload {
                file: file.to_path_buf(),
                status: FileStatus::Processed,
                detections: Some(detections),
                duration_ms: Some(duration_ms),
                error: None,
            },
        );
    }

    fn file_completed_failure(&self, file: &Path, error_code: &str, error_message: &str) {
        self.emit(
            EventType::FileCompleted,
            FileCompletedPayload {
                file: file.to_path_buf(),
                status: FileStatus::Failed,
                detections: None,
                duration_ms: None,
                error: Some(FileErrorInfo {
                    code: error_code.to_string(),
                    message: error_message.to_string(),
                }),
            },
        );
    }

    fn file_skipped(&self, file: &Path, reason: FileStatus) {
        self.emit(
            EventType::FileCompleted,
            FileCompletedPayload {
                file: file.to_path_buf(),
                status: reason,
                detections: None,
                duration_ms: None,
                error: None,
            },
        );
    }

    fn pipeline_completed(&self, summary: &PipelineSummary) {
        let status = if summary.files_failed == 0 {
            PipelineStatus::Success
        } else if summary.files_processed > 0 {
            PipelineStatus::PartialSuccess
        } else {
            PipelineStatus::Failed
        };

        self.emit(
            EventType::PipelineCompleted,
            PipelineCompletedPayload {
                status,
                files_processed: summary.files_processed,
                files_failed: summary.files_failed,
                files_skipped: summary.files_skipped,
                total_detections: summary.total_detections,
                total_segments: summary.total_segments,
                duration_ms: summary.duration_ms,
                realtime_factor: summary.realtime_factor,
            },
        );

        // Flush buffer for Json mode
        self.flush();
    }

    fn error(&self, code: &str, severity: ErrorSeverity, message: &str, suggestion: Option<&str>) {
        self.emit(
            EventType::Error,
            ErrorPayload {
                code: code.to_string(),
                severity,
                message: message.to_string(),
                suggestion: suggestion.map(ToString::to_string),
            },
        );
    }

    fn cancelled(&self, reason: CancelReason, files_completed: usize, files_total: usize) {
        self.emit(
            EventType::Cancelled,
            CancelledPayload {
                reason,
                files_completed,
                files_total,
            },
        );

        // Flush buffer for Json mode
        self.flush();
    }

    fn detections(&self, file: &Path, detections: &[crate::output::Detection]) {
        use crate::output::{DetectionInfo, DetectionsPayload};

        let detection_infos: Vec<DetectionInfo> = detections
            .iter()
            .map(|d| DetectionInfo {
                species: format!("{}_{}", d.scientific_name, d.common_name),
                common_name: d.common_name.clone(),
                scientific_name: d.scientific_name.clone(),
                confidence: d.confidence,
                start_time: d.start_time,
                end_time: d.end_time,
            })
            .collect();

        self.emit(
            EventType::Detections,
            DetectionsPayload {
                file: file.to_path_buf(),
                detections: detection_infos,
            },
        );
    }
}

/// Null reporter that does nothing (for human mode or disabled progress).
///
/// Human mode uses the existing `indicatif` progress bars in the pipeline
/// rather than this trait, so this is intentionally a no-op.
pub struct NullReporter;

impl ProgressReporter for NullReporter {
    fn pipeline_started(&self, _total_files: usize, _model: &str, _min_confidence: f32) {}
    fn file_started(
        &self,
        _file: &Path,
        _index: usize,
        _estimated_segments: usize,
        _duration_seconds: Option<f64>,
    ) {
    }
    fn progress(&self, _batch: Option<&BatchProgress>, _file: Option<&FileProgress>) {}
    fn file_completed_success(&self, _file: &Path, _detections: usize, _duration_ms: u64) {}
    fn file_completed_failure(&self, _file: &Path, _error_code: &str, _error_message: &str) {}
    fn file_skipped(&self, _file: &Path, _reason: FileStatus) {}
    fn pipeline_completed(&self, _summary: &PipelineSummary) {}
    fn error(
        &self,
        _code: &str,
        _severity: ErrorSeverity,
        _message: &str,
        _suggestion: Option<&str>,
    ) {
    }
    fn cancelled(&self, _reason: CancelReason, _files_completed: usize, _files_total: usize) {}
    fn detections(&self, _file: &Path, _detections: &[crate::output::Detection]) {}
}

/// Create a reporter based on output mode.
pub fn create_reporter(mode: OutputMode) -> Box<dyn ProgressReporter> {
    match mode {
        OutputMode::Human => Box::new(NullReporter),
        OutputMode::Json | OutputMode::Ndjson => Box::new(JsonProgressReporter::new(mode)),
    }
}

/// Emit a JSON result event to stdout.
///
/// This is used by command handlers to output structured results when
/// running in JSON or NDJSON output mode.
pub fn emit_json_result<T: serde::Serialize>(payload: &T) {
    use crate::output::json_envelope::{EventType, JsonEnvelope};

    let envelope = JsonEnvelope::new(EventType::Result, payload);
    match serde_json::to_string(&envelope) {
        Ok(json) => println!("{json}"),
        Err(e) => {
            // Log to stderr so it doesn't corrupt JSON output stream
            eprintln!("error: failed to serialize JSON result: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_throttler_emits_at_boundaries() {
        let throttler = ProgressThrottler::new();

        // Should always emit at 0%
        assert!(throttler.should_emit(0.0));

        // Should not emit small changes
        assert!(!throttler.should_emit(5.0));

        // Should emit at 10% threshold
        assert!(throttler.should_emit(10.0));

        // Should always emit at 100%
        assert!(throttler.should_emit(100.0));
    }

    #[test]
    fn test_throttler_reset() {
        let throttler = ProgressThrottler::new();

        throttler.should_emit(50.0);
        assert!(!throttler.should_emit(55.0));

        throttler.reset();
        assert!(throttler.should_emit(0.0));
    }

    #[test]
    fn test_json_reporter_ndjson_mode() {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let writer = TestWriter {
            buffer: buffer.clone(),
        };

        let reporter = JsonProgressReporter::with_writer(OutputMode::Ndjson, writer);
        reporter.pipeline_started(5, "test-model", 0.1);

        let output = buffer.lock().expect("lock");
        let output_str = String::from_utf8_lossy(&output);
        assert!(output_str.contains("\"event\":\"pipeline_started\""));
        assert!(output_str.contains("\"total_files\":5"));
    }

    #[test]
    fn test_null_reporter_does_nothing() {
        let reporter = NullReporter;
        reporter.pipeline_started(10, "model", 0.1);
        reporter.file_started(Path::new("test.wav"), 0, 100, Some(60.0));
        reporter.file_completed_success(Path::new("test.wav"), 5, 1000);
        // No assertions - just verifying it doesn't panic
    }

    #[test]
    fn test_json_reporter_emits_detections() {
        use crate::output::Detection;
        use std::path::PathBuf;

        let buffer = Arc::new(Mutex::new(Vec::new()));
        let writer = TestWriter {
            buffer: buffer.clone(),
        };

        let reporter = JsonProgressReporter::with_writer(OutputMode::Ndjson, writer);

        let detections = vec![Detection {
            file_path: PathBuf::from("test.wav"),
            start_time: 0.0,
            end_time: 3.0,
            scientific_name: "Parus major".to_string(),
            common_name: "Great Tit".to_string(),
            confidence: 0.95,
            metadata: Default::default(),
        }];

        reporter.detections(Path::new("test.wav"), &detections);

        let output = buffer.lock().expect("lock");
        let output_str = String::from_utf8_lossy(&output);
        assert!(output_str.contains("\"event\":\"detections\""));
        assert!(output_str.contains("\"Great Tit\""));
        assert!(output_str.contains("\"confidence\":0.95"));
    }

    #[test]
    fn test_reporter_handles_write_errors() {
        use std::io;

        struct FailingWriter;
        impl Write for FailingWriter {
            fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "pipe closed"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let reporter = JsonProgressReporter::with_writer(OutputMode::Ndjson, FailingWriter);

        // Should not panic when write fails
        reporter.pipeline_started(1, "test", 0.1);
        // Test passes if no panic occurs
    }

    /// Test writer that captures output.
    struct TestWriter {
        buffer: Arc<Mutex<Vec<u8>>>,
    }

    impl Write for TestWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.buffer.lock().expect("lock").extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
}
