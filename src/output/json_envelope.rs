//! JSON envelope types for CLI output.
//!
//! This module provides structured JSON output for command-line operations,
//! enabling birda to be used as a backend service for web frontends.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Current spec version for JSON envelope.
pub const SPEC_VERSION: &str = "1.1";

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
    /// Detection results for a file.
    Detections,
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
    /// Available models from registry.
    AvailableModels,
    /// Model validation check results.
    ModelCheck,
    /// Configuration file path.
    ConfigPath,
    /// Model removed from configuration.
    ModelRemoved,
    /// Model installed.
    ModelInstalled,
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
    /// Execution provider information.
    pub execution_provider: ExecutionProviderInfo,
    /// Range filter information, if active.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range_filter: Option<RangeFilterInfo>,
}

/// Range filter status information for GUI display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeFilterInfo {
    /// `BirdNET` Geomodel version used, e.g. "3.0.2".
    pub geomodel_version: String,
    /// Number of species predicted present at the query location and date.
    pub species_in_range: usize,
    /// Total classifier species.
    pub total_species: usize,
    /// Classifier species that have a geomodel entry.
    pub mapped_species: usize,
    /// Classifier species with no geomodel entry, so no range data.
    pub unmatched_species: usize,
    /// What happens to unmatched species: "keep" or "drop".
    pub unmatched_policy: String,
    /// Minimum occurrence score for a species to count as present.
    pub threshold: f32,
}

/// Execution provider information for GUI display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionProviderInfo {
    /// What the user requested ("auto", "gpu", "tensorrt", etc).
    pub requested: String,
    /// What execution provider is actually being used ("`TensorRT`", "CUDA", "CPU", etc).
    pub actual: String,
    /// Reason for fallback if we didn't use requested provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
}

impl From<crate::inference::ExecutionProviderStatus> for ExecutionProviderInfo {
    fn from(status: crate::inference::ExecutionProviderStatus) -> Self {
        Self {
            requested: status.requested,
            actual: status.actual,
            fallback_reason: status.fallback_reason,
        }
    }
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
// Detections Event Payload
// ============================================================================

/// Payload for `detections` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionsPayload {
    /// Path to source file.
    pub file: PathBuf,
    /// All detections found in the file.
    pub detections: Vec<DetectionInfo>,
    /// BSG post-processing metadata (if BSG model used).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bsg: Option<BsgMetadata>,
}

/// BSG (Bird Sounds Global) Finland post-processing metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BsgMetadata {
    /// Whether calibration was applied (always true for BSG models).
    pub calibration_applied: bool,
    /// Whether SDM (Species Distribution Model) was applied.
    pub sdm_applied: bool,
    /// Latitude used for SDM (if SDM applied).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latitude: Option<f32>,
    /// Longitude used for SDM (if SDM applied).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub longitude: Option<f32>,
    /// Day of year used for SDM (if SDM applied).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub day_of_year: Option<u32>,
}

/// Information about a single detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionInfo {
    /// Full species label (e.g., `"Parus major_Great Tit"`).
    /// Format: `"{scientific_name}_{common_name}"`.
    /// Provided for convenience and `BirdNET` format compatibility.
    pub species: String,
    /// Common name.
    pub common_name: String,
    /// Scientific name.
    pub scientific_name: String,
    /// Confidence score (0.0-1.0).
    pub confidence: f32,
    /// Start time in seconds.
    pub start_time: f32,
    /// End time in seconds.
    pub end_time: f32,
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

/// Payload for available models list result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableModelsPayload {
    /// Result type discriminator.
    pub result_type: ResultType,
    /// List of available models from registry.
    pub models: Vec<AvailableModelEntry>,
    /// The shared range filter asset, which is not one of `models`.
    ///
    /// Kept in its own field rather than folded into `models` because it is not
    /// selectable with `-m`: a consumer building a model picker from `models`
    /// would otherwise offer an entry that fails on use. This also keeps the
    /// change additive, so existing consumers of `models` are unaffected.
    ///
    /// Named `available_range_filter` rather than `range_filter` because
    /// `DetectionsPayload` already has a `range_filter` field carrying a
    /// completely different shape (`RangeFilterInfo`: per-run coverage counts).
    /// Distinct names keep the two readable side by side in a schema, a doc
    /// table or a generated client, where one name over two disjoint shapes
    /// invites the wrong one being reached for. This is a naming-clarity call,
    /// not a fix for a known consumer: TypeScript scopes field names to their
    /// interface, so nothing in birda-gui collides today.
    ///
    /// `None` when the registry predates the geomodel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub available_range_filter: Option<AvailableRangeFilterEntry>,
}

/// The shared range filter asset, as offered by `birda models list-available`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableRangeFilterEntry {
    /// The id to pass to `birda models install` and `birda models info`.
    ///
    /// This is the install handle ("geomodel"), not the registry asset id
    /// ("birdnet-geomodel-v3"), because it is the string a user can type.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Asset version.
    pub version: String,
    /// Organization/author.
    pub vendor: String,
    /// License type (SPDX identifier).
    pub license: String,
    /// Whether commercial use is allowed.
    pub commercial_use: bool,
    /// Whether derivatives must be shared under the same license.
    pub share_alike: bool,
    /// Number of species the model scores.
    pub species_count: usize,
    /// Combined download size of the model and labels files, in bytes.
    ///
    /// `None` unless both files declare a size and their sum fits in `u64`.
    /// Both files are required for the range filter to work, so a partial total
    /// would read as the whole download and understate it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
}

/// A single available model from the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableModelEntry {
    /// Model identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Short description.
    pub description: String,
    /// Organization/author.
    pub vendor: String,
    /// Model version.
    pub version: String,
    /// Model type identifier.
    pub model_type: String,
    /// Whether this model is recommended.
    pub recommended: bool,
    /// License type (SPDX identifier).
    pub license: String,
    /// Whether commercial use is allowed.
    pub commercial_use: bool,
}

/// Payload for model check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCheckPayload {
    /// Result type discriminator.
    pub result_type: ResultType,
    /// Check results per model.
    pub models: Vec<ModelCheckEntry>,
    /// Shared range filter status.
    pub geomodel: GeomodelInfo,
    /// Leftover partial-download files (`<name>.<pid>.part`) in the models
    /// directory, from a download interrupted before it finished. Reported so a
    /// directory that looks clean but is holding an abandoned multi-hundred-MB
    /// download is visible; birda never deletes these automatically.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub leftover_downloads: Vec<PathBuf>,
}

/// Status of the shared `BirdNET` Geomodel range filter.
///
/// Reported once rather than per model, since every classifier shares it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeomodelInfo {
    /// Geomodel version, e.g. "3.0.2".
    pub version: String,
    /// Whether both geomodel files are present on disk.
    pub installed: bool,
    /// Number of species the geomodel scores.
    pub species_count: usize,
    /// Path to the geomodel ONNX file, when installed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<PathBuf>,
    /// Path to the geomodel labels file, when installed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels_path: Option<PathBuf>,
    /// Files from earlier birda versions that are no longer used and can be
    /// deleted, such as a leftover `birdnet-v24-meta.onnx`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub obsolete_files: Vec<PathBuf>,
}

/// Validation result for a single model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCheckEntry {
    /// Model name/identifier.
    pub id: String,
    /// Whether validation passed.
    pub valid: bool,
    /// Error message if validation failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Payload for config path result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigPathPayload {
    /// Result type discriminator.
    pub result_type: ResultType,
    /// Path to the configuration file.
    pub config_path: PathBuf,
    /// Whether the configuration file exists.
    pub exists: bool,
}

/// Payload for model removed result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRemovedPayload {
    /// Result type discriminator.
    pub result_type: ResultType,
    /// ID/name of the removed model.
    pub id: String,
    /// Whether a purge (file deletion) was requested.
    pub purge_requested: bool,
    /// New default model (if promoted), or absent if no promotion occurred.
    pub new_default: Option<String>,
}

/// Payload for model installed result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInstalledPayload {
    /// Result type discriminator.
    pub result_type: ResultType,
    /// ID/name of the installed model.
    pub id: String,
    /// Whether the model was set as default.
    pub set_as_default: bool,
    /// Path to the installed model file.
    pub model_path: PathBuf,
    /// Path to the installed labels file.
    pub labels_path: PathBuf,
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
            execution_provider: ExecutionProviderInfo {
                requested: "auto".to_string(),
                actual: "CPU".to_string(),
                fallback_reason: None,
            },
            range_filter: None,
        };
        let envelope = JsonEnvelope::new(EventType::PipelineStarted, payload);

        let json = serde_json::to_string(&envelope).expect("serialize");
        assert!(json.contains("\"spec_version\":\"1.1\""));
        assert!(json.contains("\"event\":\"pipeline_started\""));
        assert!(json.contains("\"total_files\":10"));
    }

    #[test]
    fn test_execution_provider_info_serialization() {
        let info = ExecutionProviderInfo {
            requested: "auto".to_string(),
            actual: "CUDA".to_string(),
            fallback_reason: Some("TensorRT libraries not found".to_string()),
        };

        let json = serde_json::to_string(&info).expect("serialize");
        let actual: serde_json::Value = serde_json::from_str(&json).expect("deserialize");

        let expected = serde_json::json!({
            "requested": "auto",
            "actual": "CUDA",
            "fallback_reason": "TensorRT libraries not found"
        });

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_execution_provider_info_without_fallback() {
        let info = ExecutionProviderInfo {
            requested: "cuda".to_string(),
            actual: "CUDA".to_string(),
            fallback_reason: None,
        };

        let json = serde_json::to_string(&info).expect("serialize");
        let actual: serde_json::Value = serde_json::from_str(&json).expect("deserialize");

        let expected = serde_json::json!({
            "requested": "cuda",
            "actual": "CUDA"
        });

        // Should not have fallback_reason key
        assert_eq!(actual, expected);
        assert!(actual.get("fallback_reason").is_none());
    }

    #[test]
    fn test_pipeline_started_with_execution_provider() {
        let payload = PipelineStartedPayload {
            total_files: 10,
            model: "birdnet-v24".to_string(),
            min_confidence: 0.1,
            execution_provider: ExecutionProviderInfo {
                requested: "auto".to_string(),
                actual: "TensorRT".to_string(),
                fallback_reason: None,
            },
            range_filter: None,
        };

        let envelope = JsonEnvelope::new(EventType::PipelineStarted, payload);
        let json = serde_json::to_string(&envelope).expect("serialize");

        assert!(json.contains("\"execution_provider\""));
        assert!(json.contains("\"requested\":\"auto\""));
        assert!(json.contains("\"actual\":\"TensorRT\""));
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

    #[test]
    fn test_new_result_type_serialization() {
        assert_eq!(
            serde_json::to_string(&ResultType::AvailableModels).expect("serialize"),
            "\"available_models\""
        );
        assert_eq!(
            serde_json::to_string(&ResultType::ModelCheck).expect("serialize"),
            "\"model_check\""
        );
        assert_eq!(
            serde_json::to_string(&ResultType::ConfigPath).expect("serialize"),
            "\"config_path\""
        );
        assert_eq!(
            serde_json::to_string(&ResultType::ModelRemoved).expect("serialize"),
            "\"model_removed\""
        );
        assert_eq!(
            serde_json::to_string(&ResultType::ModelInstalled).expect("serialize"),
            "\"model_installed\""
        );
    }

    #[test]
    fn test_available_models_payload() {
        let payload = AvailableModelsPayload {
            result_type: ResultType::AvailableModels,
            models: vec![AvailableModelEntry {
                id: "birdnet-v24".to_string(),
                name: "BirdNET v2.4".to_string(),
                description: "Bird sound recognition".to_string(),
                vendor: "Cornell".to_string(),
                version: "2.4".to_string(),
                model_type: "birdnet-v24".to_string(),
                recommended: true,
                license: "CC-BY-NC-SA-4.0".to_string(),
                commercial_use: false,
            }],
            available_range_filter: None,
        };
        let json = serde_json::to_string(&payload).expect("serialize");
        let actual: serde_json::Value = serde_json::from_str(&json).expect("deserialize");
        let expected = serde_json::json!({
            "result_type": "available_models",
            "models": [{
                "id": "birdnet-v24",
                "name": "BirdNET v2.4",
                "description": "Bird sound recognition",
                "vendor": "Cornell",
                "version": "2.4",
                "model_type": "birdnet-v24",
                "recommended": true,
                "license": "CC-BY-NC-SA-4.0",
                "commercial_use": false
            }]
        });
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_model_check_payload() {
        let payload = ModelCheckPayload {
            result_type: ResultType::ModelCheck,
            models: vec![
                ModelCheckEntry {
                    id: "birdnet".to_string(),
                    valid: true,
                    error: None,
                },
                ModelCheckEntry {
                    id: "broken".to_string(),
                    valid: false,
                    error: Some("model file not found".to_string()),
                },
            ],
            geomodel: GeomodelInfo {
                version: "3.0.2".to_string(),
                installed: true,
                species_count: 12012,
                model_path: Some(PathBuf::from("/models/birdnet-geomodel-v3.0.2.onnx")),
                labels_path: Some(PathBuf::from("/models/birdnet-geomodel-v3.0.2-labels.txt")),
                obsolete_files: Vec::new(),
            },
            leftover_downloads: vec![PathBuf::from("/models/birdnet-v30.onnx.4242.part")],
        };
        let json = serde_json::to_string(&payload).expect("serialize");
        let actual: serde_json::Value = serde_json::from_str(&json).expect("deserialize");
        let expected = serde_json::json!({
            "result_type": "model_check",
            "geomodel": {
                "version": "3.0.2",
                "installed": true,
                "species_count": 12012,
                "model_path": "/models/birdnet-geomodel-v3.0.2.onnx",
                "labels_path": "/models/birdnet-geomodel-v3.0.2-labels.txt"
            },
            "models": [
                {
                    "id": "birdnet",
                    "valid": true
                },
                {
                    "id": "broken",
                    "valid": false,
                    "error": "model file not found"
                }
            ],
            "leftover_downloads": ["/models/birdnet-v30.onnx.4242.part"]
        });
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_model_check_payload_omits_empty_leftover_downloads() {
        // skip_serializing_if keeps the key out of the JSON when there are no
        // leftovers, so the common case is byte-identical to before the field.
        let payload = ModelCheckPayload {
            result_type: ResultType::ModelCheck,
            models: Vec::new(),
            geomodel: GeomodelInfo {
                version: "3.0.2".to_string(),
                installed: false,
                species_count: 0,
                model_path: None,
                labels_path: None,
                obsolete_files: Vec::new(),
            },
            leftover_downloads: Vec::new(),
        };
        let json = serde_json::to_string(&payload).expect("serialize");
        assert!(!json.contains("leftover_downloads"));
    }

    #[test]
    fn test_config_path_payload() {
        let payload = ConfigPathPayload {
            result_type: ResultType::ConfigPath,
            config_path: PathBuf::from("/home/user/.config/birda/config.toml"),
            exists: true,
        };
        let json = serde_json::to_string(&payload).expect("serialize");
        let actual: serde_json::Value = serde_json::from_str(&json).expect("deserialize");
        let expected = serde_json::json!({
            "result_type": "config_path",
            "config_path": "/home/user/.config/birda/config.toml",
            "exists": true
        });
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_detections_event_serialization() {
        assert_eq!(
            serde_json::to_string(&EventType::Detections).expect("serialize"),
            "\"detections\""
        );
    }

    #[test]
    fn test_detection_info_serialization() {
        let info = DetectionInfo {
            species: "Parus major_Great Tit".to_string(),
            common_name: "Great Tit".to_string(),
            scientific_name: "Parus major".to_string(),
            confidence: 0.95,
            start_time: 0.0,
            end_time: 3.0,
        };
        let json = serde_json::to_string(&info).expect("serialize");
        let actual: serde_json::Value = serde_json::from_str(&json).expect("deserialize");
        let expected = serde_json::json!({
            "species": "Parus major_Great Tit",
            "common_name": "Great Tit",
            "scientific_name": "Parus major",
            "confidence": 0.95,
            "start_time": 0.0,
            "end_time": 3.0
        });
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_detections_payload_serialization() {
        let payload = DetectionsPayload {
            file: PathBuf::from("audio.wav"),
            detections: vec![DetectionInfo {
                species: "Parus major_Great Tit".to_string(),
                common_name: "Great Tit".to_string(),
                scientific_name: "Parus major".to_string(),
                confidence: 0.95,
                start_time: 0.0,
                end_time: 3.0,
            }],
            bsg: None,
        };
        let json = serde_json::to_string(&payload).expect("serialize");
        assert!(json.contains("\"file\":\"audio.wav\""));
        assert!(json.contains("\"detections\""));
        assert!(json.contains("\"Great Tit\""));
    }

    #[test]
    fn test_detections_payload_with_bsg_metadata() {
        let payload = DetectionsPayload {
            file: PathBuf::from("audio.wav"),
            detections: vec![DetectionInfo {
                species: "Parus major_Great Tit".to_string(),
                common_name: "Great Tit".to_string(),
                scientific_name: "Parus major".to_string(),
                confidence: 0.95,
                start_time: 0.0,
                end_time: 3.0,
            }],
            bsg: Some(BsgMetadata {
                calibration_applied: true,
                sdm_applied: true,
                latitude: Some(60.1699),
                longitude: Some(24.9384),
                day_of_year: Some(150),
            }),
        };
        let json = serde_json::to_string(&payload).expect("serialize");
        let actual: serde_json::Value = serde_json::from_str(&json).expect("deserialize");

        assert!(actual["bsg"]["calibration_applied"].as_bool().unwrap());
        assert!(actual["bsg"]["sdm_applied"].as_bool().unwrap());
        assert_eq!(actual["bsg"]["latitude"].as_f64().unwrap(), 60.1699);
        assert_eq!(actual["bsg"]["longitude"].as_f64().unwrap(), 24.9384);
        assert_eq!(actual["bsg"]["day_of_year"].as_u64().unwrap(), 150);
    }

    #[test]
    fn test_model_removed_payload_with_promotion() {
        let payload = ModelRemovedPayload {
            result_type: ResultType::ModelRemoved,
            id: "birdnet-v24".to_string(),
            purge_requested: true,
            new_default: Some("perch-v1".to_string()),
        };
        let json = serde_json::to_string(&payload).expect("serialize");
        let actual: serde_json::Value = serde_json::from_str(&json).expect("deserialize");
        let expected = serde_json::json!({
            "result_type": "model_removed",
            "id": "birdnet-v24",
            "purge_requested": true,
            "new_default": "perch-v1"
        });
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_model_removed_payload_null_new_default() {
        let payload = ModelRemovedPayload {
            result_type: ResultType::ModelRemoved,
            id: "birdnet-v24".to_string(),
            purge_requested: false,
            new_default: None,
        };
        let json = serde_json::to_string(&payload).expect("serialize");
        let actual: serde_json::Value = serde_json::from_str(&json).expect("deserialize");
        // new_default must be present as null (not omitted)
        assert!(actual.get("new_default").is_some());
        assert!(actual["new_default"].is_null());
    }

    #[test]
    fn test_range_filter_info_serializes_geomodel_fields() {
        let info = RangeFilterInfo {
            geomodel_version: "3.0.2".to_string(),
            species_in_range: 341,
            total_species: 6522,
            mapped_species: 6217,
            unmatched_species: 305,
            unmatched_policy: "keep".to_string(),
            threshold: 0.01,
        };

        let json = serde_json::to_string(&info).expect("serialize");
        let actual: serde_json::Value = serde_json::from_str(&json).expect("deserialize");

        let expected = serde_json::json!({
            "geomodel_version": "3.0.2",
            "species_in_range": 341,
            "total_species": 6522,
            "mapped_species": 6217,
            "unmatched_species": 305,
            "unmatched_policy": "keep",
            "threshold": 0.01
        });

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_range_filter_info_drops_the_cross_model_fields() {
        // The cross-model fallback is gone: every classifier uses the same
        // geomodel, so these fields must not reappear in the envelope.
        let info = RangeFilterInfo {
            geomodel_version: "3.0.2".to_string(),
            species_in_range: 341,
            total_species: 14795,
            mapped_species: 11145,
            unmatched_species: 3650,
            unmatched_policy: "drop".to_string(),
            threshold: 0.01,
        };

        let json = serde_json::to_string(&info).expect("serialize");
        let actual: serde_json::Value = serde_json::from_str(&json).expect("deserialize");

        assert!(actual.get("cross_model").is_none());
        assert!(actual.get("meta_model_source").is_none());
    }

    #[test]
    fn test_spec_version_is_bumped_for_the_geomodel_envelope() {
        // 1.1 is the version in which `DetectionsPayload.range_filter` became
        // `RangeFilterInfo` (geomodel coverage counts), replacing the
        // cross-model fields. This is a change-detector on that constant, not
        // evidence that any later change was accompanied by a bump: it can only
        // fire when someone edits the literal.
        //
        // Purely additive optional fields do NOT move it. That is why
        // `AvailableModelsPayload::available_range_filter` was added under 1.1:
        // it carries `#[serde(default, skip_serializing_if)]`, so old consumers
        // deserialize unchanged and the emitted bytes are identical when the
        // registry has no range filter.
        assert_eq!(
            SPEC_VERSION, "1.1",
            "changing this constant is a wire-contract decision; update the \
             reasoning above along with it"
        );
    }

    #[test]
    fn test_model_installed_payload() {
        let payload = ModelInstalledPayload {
            result_type: ResultType::ModelInstalled,
            id: "birdnet-v24".to_string(),
            set_as_default: true,
            model_path: PathBuf::from("/models/birdnet.onnx"),
            labels_path: PathBuf::from("/models/labels.txt"),
        };
        let json = serde_json::to_string(&payload).expect("serialize");
        let actual: serde_json::Value = serde_json::from_str(&json).expect("deserialize");
        let expected = serde_json::json!({
            "result_type": "model_installed",
            "id": "birdnet-v24",
            "set_as_default": true,
            "model_path": "/models/birdnet.onnx",
            "labels_path": "/models/labels.txt"
        });
        assert_eq!(actual, expected);
    }
}
