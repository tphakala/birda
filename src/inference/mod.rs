//! Inference module for bird species detection.

mod classifier;
mod cuda_detection;
pub mod geomodel;
pub mod geomodel_filter;
mod library_detection;
mod provider;
pub mod range_filter;
mod runtime;
mod tensorrt_detection;

pub use birdnet_onnx::{BatchInferenceContext, InferenceOptions};
pub use classifier::{BirdClassifier, ExecutionProviderStatus};
pub use cuda_detection::{get_cuda_library_patterns, is_cuda_available};
pub use geomodel::{GeomodelScores, SpeciesMapping, scientific_name};
pub use geomodel_filter::{FilterSettings, filter_predictions};
pub use provider::{ProviderMetadata, provider_metadata};
pub use runtime::ensure_runtime_available;
pub use tensorrt_detection::{get_tensorrt_library_name, is_tensorrt_available};

use std::path::PathBuf;

/// Configuration for range filtering at runtime.
///
/// Every classifier uses the same `BirdNET` Geomodel v3.0.2, so there is no
/// per-model variation here beyond the query parameters.
#[derive(Debug, Clone)]
pub struct RangeFilterConfig {
    /// Path to the geomodel ONNX file.
    pub geomodel_path: PathBuf,
    /// Path to the geomodel labels file.
    ///
    /// The filter is built from these labels, not the classifier's: the
    /// geomodel has 12,012 outputs and `birdnet_onnx` validates that the label
    /// count matches. Scores are projected into the classifier's label space
    /// afterwards.
    pub geomodel_labels_path: PathBuf,
    /// Filtering threshold.
    pub threshold: f32,
    /// Latitude.
    pub latitude: f64,
    /// Longitude.
    pub longitude: f64,
    /// Month (1-12).
    pub month: u32,
    /// Day (1-31).
    pub day: u32,
    /// Enable re-ranking.
    pub rerank: bool,
    /// What to do with classifier species that have no geomodel entry.
    pub unmatched: crate::config::UnmatchedPolicy,
}
