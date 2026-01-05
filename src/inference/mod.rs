//! Inference module for bird species detection.

mod classifier;
pub mod range_filter;

pub use birdnet_onnx::InferenceOptions;
pub use classifier::BirdClassifier;

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
}
