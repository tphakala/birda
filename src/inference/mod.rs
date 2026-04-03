//! Inference module for bird species detection.

mod classifier;
mod cuda_detection;
mod library_detection;
mod provider;
pub mod range_filter;
mod runtime;
mod tensorrt_detection;

pub use birdnet_onnx::{BatchInferenceContext, InferenceOptions};
pub use classifier::{BirdClassifier, ExecutionProviderStatus};
pub use cuda_detection::{get_cuda_library_patterns, is_cuda_available};
pub use provider::{ProviderMetadata, provider_metadata};
pub use runtime::ensure_runtime_available;
pub use tensorrt_detection::{get_tensorrt_library_name, is_tensorrt_available};

use std::path::PathBuf;

/// Configuration for range filtering at runtime.
#[derive(Debug, Clone)]
pub struct RangeFilterConfig {
    /// Path to meta model file.
    pub meta_model_path: PathBuf,
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
    /// When using a meta model from a different model, this holds the path
    /// to that model's labels file. The classifier uses these labels to build
    /// the range filter (for correct output-size validation) and then remaps
    /// the resulting location scores to the classifier's own label format.
    pub cross_model_labels: Option<PathBuf>,
    /// Name of the model that provided the meta model (for logging/reporting).
    /// e.g., "birdnet-v24" when using `BirdNET`'s meta model for perch-v2.
    pub meta_model_source: Option<String>,
}
