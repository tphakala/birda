//! JSON envelope types for CLI output.
//!
//! This module provides structured JSON output for command-line operations,
//! enabling birda to be used as a backend service for web frontends.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Current spec version for JSON envelope.
pub const SPEC_VERSION: &str = "1.0";

/// JSON envelope wrapping all CLI output events.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(deserialize = "T: serde::de::DeserializeOwned"))]
pub struct JsonEnvelope<T> {
    /// API specification version.
    pub spec_version: String,
    /// Event timestamp.
    pub timestamp: DateTime<Utc>,
    /// Event type.
    pub event: EventType,
    /// Event-specific payload.
    pub payload: T,
}

impl<T: Serialize> JsonEnvelope<T> {
    /// Create a new envelope with the current timestamp.
    pub fn new(event: EventType, payload: T) -> Self {
        Self {
            spec_version: SPEC_VERSION.to_string(),
            timestamp: Utc::now(),
            event,
            payload,
        }
    }
}

/// Event types for JSON output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    /// Analysis pipeline starting.
    PipelineStarted,
    /// Starting to process a file.
    FileStarted,
    /// Periodic progress update.
    Progress,
    /// File processing finished.
    FileCompleted,
    /// All files processed.
    PipelineCompleted,
    /// Final result.
    Result,
    /// Error occurred.
    Error,
    /// Operation cancelled.
    Cancelled,
}

/// Result type discriminator for result payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultType {
    /// Audio analysis results.
    Analysis,
    /// Model list.
    ModelList,
    /// Model information.
    ModelInfo,
    /// Clip extraction results.
    ClipExtraction,
    /// Species list generation.
    SpeciesList,
    /// Configuration display.
    Config,
    /// Available execution providers.
    Providers,
    /// Version information.
    Version,
}

/// Error severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorSeverity {
    /// Fatal error - pipeline cannot continue.
    Fatal,
    /// Warning - operation continues but with issues.
    Warning,
}

/// Progress information for a batch of files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProgress {
    /// Current file index (1-based).
    pub current: usize,
    /// Total number of files.
    pub total: usize,
    /// Progress percentage (0-100).
    pub percent: f32,
}

/// Progress information for a single file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileProgress {
    /// File path.
    pub path: PathBuf,
    /// Segments processed.
    pub segments_done: usize,
    /// Total segments.
    pub segments_total: usize,
    /// Progress percentage (0-100).
    pub percent: f32,
}

/// Error payload for error events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPayload {
    /// Error code (`snake_case` identifier).
    pub code: String,
    /// Error severity.
    pub severity: ErrorSeverity,
    /// Human-readable error message.
    pub message: String,
    /// Suggested action to resolve the error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
}

/// File processing status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileStatus {
    /// File was processed successfully.
    Processed,
    /// File processing failed.
    Failed,
    /// File was skipped (output exists).
    Skipped,
    /// File was skipped due to being locked.
    Locked,
}

// ============================================================================
// Pipeline Event Payloads
// ============================================================================

/// Payload for `pipeline_started` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStartedPayload {
    /// Total number of files to process.
    pub total_files: usize,
    /// Model being used.
    pub model: String,
    /// Minimum confidence threshold.
    pub min_confidence: f32,
}

/// Payload for `file_started` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileStartedPayload {
    /// File path.
    pub file: PathBuf,
    /// File index (0-based).
    pub index: usize,
    /// Estimated number of segments.
    pub estimated_segments: usize,
    /// Estimated duration in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
}

/// Payload for progress event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressPayload {
    /// Batch progress (if processing multiple files).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch: Option<BatchProgress>,
    /// Current file progress.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<FileProgress>,
    /// Download progress (for model downloads).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub download: Option<DownloadProgress>,
}

/// Download progress information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    /// Operation type.
    pub operation: String,
    /// Model being downloaded.
    pub model: String,
    /// File being downloaded.
    pub file: String,
    /// Bytes downloaded.
    pub downloaded_bytes: u64,
    /// Total bytes.
    pub total_bytes: u64,
    /// Progress percentage.
    pub percent: f32,
}

/// Payload for `file_completed` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileCompletedPayload {
    /// File path.
    pub file: PathBuf,
    /// Processing status.
    pub status: FileStatus,
    /// Number of detections (if processed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detections: Option<usize>,
    /// Processing duration in milliseconds (if processed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// Error details (if failed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<FileErrorInfo>,
}

/// Error information for a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileErrorInfo {
    /// Error code.
    pub code: String,
    /// Error message.
    pub message: String,
}

/// Payload for `pipeline_completed` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineCompletedPayload {
    /// Overall status.
    pub status: PipelineStatus,
    /// Files successfully processed.
    pub files_processed: usize,
    /// Files that failed.
    pub files_failed: usize,
    /// Files skipped.
    pub files_skipped: usize,
    /// Total detections across all files.
    pub total_detections: usize,
    /// Total segments processed.
    pub total_segments: usize,
    /// Total duration in milliseconds.
    pub duration_ms: u64,
    /// Realtime processing factor.
    pub realtime_factor: f64,
}

/// Pipeline completion status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineStatus {
    /// All files processed successfully.
    Success,
    /// Some files failed.
    PartialSuccess,
    /// Pipeline failed completely.
    Failed,
}

/// Payload for cancelled event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelledPayload {
    /// Cancellation reason.
    pub reason: CancelReason,
    /// Files completed before cancellation.
    pub files_completed: usize,
    /// Total files that were planned.
    pub files_total: usize,
}

/// Reason for cancellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancelReason {
    /// User interrupted (SIGINT).
    UserInterrupt,
    /// Timeout.
    Timeout,
}

// ============================================================================
// Result Payloads for Commands
// ============================================================================

/// Payload for model list result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListPayload {
    /// Result type discriminator.
    pub result_type: ResultType,
    /// List of configured models.
    pub models: Vec<ModelEntry>,
}

/// A single model entry in the list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Model identifier/name.
    pub id: String,
    /// Model type (e.g., `birdnet-v24`).
    pub model_type: String,
    /// Whether this is the default model.
    pub is_default: bool,
    /// Path to the model file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    /// Path to the labels file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels_path: Option<PathBuf>,
    /// Whether a meta model is configured.
    pub has_meta_model: bool,
}

/// Payload for model info result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfoPayload {
    /// Result type discriminator.
    pub result_type: ResultType,
    /// Model details.
    pub model: ModelDetails,
}

/// Detailed model information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDetails {
    /// Model identifier/name.
    pub id: String,
    /// Model type.
    pub model_type: String,
    /// Path to the model file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    /// Path to the labels file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels_path: Option<PathBuf>,
    /// Path to the meta model file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta_model_path: Option<PathBuf>,
    /// Source (configured or registry).
    pub source: String,
}

/// Payload for providers result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvidersPayload {
    /// Result type discriminator.
    pub result_type: ResultType,
    /// Available execution providers.
    pub providers: Vec<ProviderInfo>,
}

/// Information about an execution provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderInfo {
    /// Provider identifier.
    pub id: String,
    /// Provider display name.
    pub name: String,
    /// Description of the provider.
    pub description: String,
}

/// Payload for config show result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigPayload {
    /// Result type discriminator.
    pub result_type: ResultType,
    /// Path to the config file.
    pub config_path: PathBuf,
    /// The configuration contents (as JSON value for flexibility).
    pub config: serde_json::Value,
}

/// Payload for species list result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesListPayload {
    /// Result type discriminator.
    pub result_type: ResultType,
    /// Location latitude.
    pub lat: f64,
    /// Location longitude.
    pub lon: f64,
    /// Week number used.
    pub week: u32,
    /// Threshold used.
    pub threshold: f32,
    /// Number of species.
    pub species_count: usize,
    /// Output file path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_file: Option<PathBuf>,
    /// List of species.
    pub species: Vec<SpeciesEntry>,
}

/// A single species entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesEntry {
    /// Scientific name.
    pub scientific_name: String,
    /// Common name.
    pub common_name: String,
    /// Occurrence frequency/probability.
    pub frequency: f32,
}

/// Payload for version result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionPayload {
    /// Result type discriminator.
    pub result_type: ResultType,
    /// Application version.
    pub version: String,
}

/// Payload for clip extraction result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipExtractionPayload {
    /// Result type discriminator.
    pub result_type: ResultType,
    /// Output directory.
    pub output_dir: PathBuf,
    /// Total clips extracted.
    pub total_clips: usize,
    /// Total detection files processed.
    pub total_files: usize,
    /// List of extracted clips.
    pub clips: Vec<ClipExtractionEntry>,
}

/// A single extracted clip entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipExtractionEntry {
    /// Source audio file.
    pub source_audio: PathBuf,
    /// Scientific name of detected species.
    pub scientific_name: String,
    /// Maximum confidence score.
    pub confidence: f32,
    /// Start time in seconds.
    pub start_time: f64,
    /// End time in seconds.
    pub end_time: f64,
    /// Output clip file path.
    pub output_file: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_envelope_serialization() {
        let payload = PipelineStartedPayload {
            total_files: 10,
            model: "birdnet-v24".to_string(),
            min_confidence: 0.1,
        };
        let envelope = JsonEnvelope::new(EventType::PipelineStarted, payload);

        let json = serde_json::to_string(&envelope).expect("serialize");
        assert!(json.contains("\"spec_version\":\"1.0\""));
        assert!(json.contains("\"event\":\"pipeline_started\""));
        assert!(json.contains("\"total_files\":10"));
    }

    #[test]
    fn test_event_type_serialization() {
        assert_eq!(
            serde_json::to_string(&EventType::PipelineStarted).expect("serialize"),
            "\"pipeline_started\""
        );
        assert_eq!(
            serde_json::to_string(&EventType::FileCompleted).expect("serialize"),
            "\"file_completed\""
        );
    }

    #[test]
    fn test_error_severity_serialization() {
        assert_eq!(
            serde_json::to_string(&ErrorSeverity::Fatal).expect("serialize"),
            "\"fatal\""
        );
        assert_eq!(
            serde_json::to_string(&ErrorSeverity::Warning).expect("serialize"),
            "\"warning\""
        );
    }

    #[test]
    fn test_result_type_serialization() {
        assert_eq!(
            serde_json::to_string(&ResultType::Analysis).expect("serialize"),
            "\"analysis\""
        );
        assert_eq!(
            serde_json::to_string(&ResultType::ModelList).expect("serialize"),
            "\"model_list\""
        );
    }

    #[test]
    fn test_progress_payload_skips_none() {
        let payload = ProgressPayload {
            batch: Some(BatchProgress {
                current: 1,
                total: 10,
                percent: 10.0,
            }),
            file: None,
            download: None,
        };

        let json = serde_json::to_string(&payload).expect("serialize");
        assert!(json.contains("\"batch\""));
        assert!(!json.contains("\"file\""));
        assert!(!json.contains("\"download\""));
    }
}
