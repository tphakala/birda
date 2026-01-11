//! Output format writers.

mod audacity;
mod csv;
mod json;
pub mod json_envelope;
mod kaleidoscope;
pub mod progress;
mod raven;
mod reporter;
mod types;
mod writer;

pub use audacity::AudacityWriter;
pub use csv::CsvWriter;
pub use json::JsonResultWriter;
pub use json_envelope::{
    BatchProgress, CancelReason, CancelledPayload, ClipExtractionEntry, ClipExtractionPayload,
    ConfigPayload, DownloadProgress, ErrorPayload, ErrorSeverity, EventType, FileCompletedPayload,
    FileErrorInfo, FileProgress, FileStartedPayload, FileStatus, JsonEnvelope, ModelDetails,
    ModelEntry, ModelInfoPayload, ModelListPayload, PipelineCompletedPayload,
    PipelineStartedPayload, PipelineStatus, ProgressPayload, ProviderInfo, ProvidersPayload,
    ResultType, SPEC_VERSION, SpeciesEntry, SpeciesListPayload, VersionPayload,
};
pub use kaleidoscope::KaleidoscopeWriter;
pub use raven::RavenWriter;
pub use reporter::{
    JsonProgressReporter, NullReporter, PipelineSummary, ProgressReporter, ProgressThrottler,
    create_reporter, emit_json_result,
};
pub use types::{Detection, DetectionMetadata};
pub use writer::OutputWriter;
