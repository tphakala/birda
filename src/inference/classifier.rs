//! Inference classifier wrapper around birdnet-onnx.

use crate::config::{InferenceDevice, ModelConfig as BirdaModelConfig};
use crate::error::{Error, Result};
use birdnet_onnx::{
    Classifier, ClassifierBuilder, ExecutionProviderInfo, PredictionResult,
    available_execution_providers,
};
use std::collections::HashSet;
use tracing::{debug, info, warn};

/// Wrapper around birdnet-onnx Classifier with birda configuration.
pub struct BirdClassifier {
    inner: Classifier,
    range_filter: Option<crate::inference::range_filter::RangeFilter>,
    range_filter_config: Option<crate::inference::RangeFilterConfig>,
    /// Optional species list for filtering (from file).
    /// None if no species list file provided or if using dynamic range filtering.
    species_list: Option<HashSet<String>>,
}

impl BirdClassifier {
    /// Build a classifier from birda model configuration.
    pub fn from_config(
        model_config: &BirdaModelConfig,
        device: InferenceDevice,
        min_confidence: f32,
        top_k: usize,
        range_filter_config: Option<crate::inference::RangeFilterConfig>,
        species_list: Option<HashSet<String>>,
    ) -> Result<Self> {
        // Check available execution providers at compile-time
        let available_providers = available_execution_providers();
        debug!(
            "Available execution providers: {}",
            available_providers
                .iter()
                .map(|p| format!("{p:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        );

        let tensorrt_available = available_providers.contains(&ExecutionProviderInfo::TensorRt);
        let cuda_available = available_providers.contains(&ExecutionProviderInfo::Cuda);

        let builder = ClassifierBuilder::new()
            .model_path(model_config.path.to_string_lossy().to_string())
            .labels_path(model_config.labels.to_string_lossy().to_string())
            .top_k(top_k)
            .min_confidence(min_confidence);

        // Add execution provider based on device setting and determine actual device used
        let (builder, actual_device_msg) = match device {
            InferenceDevice::Auto => {
                if tensorrt_available {
                    info!("Auto mode: TensorRT available, registering");
                    debug!("Registering TensorRT execution provider");
                    (
                        builder.with_tensorrt(),
                        "Auto (TensorRT → CPU fallback)".to_string(),
                    )
                } else if cuda_available {
                    info!("Auto mode: CUDA available, registering");
                    debug!("Registering CUDA execution provider");
                    (
                        builder.with_cuda(),
                        "Auto (CUDA → CPU fallback)".to_string(),
                    )
                } else {
                    info!("Auto mode: No GPU providers available, using CPU");
                    (builder, "Auto (CPU only)".to_string())
                }
            }
            InferenceDevice::Gpu => {
                info!("Requested device: GPU");
                if tensorrt_available {
                    debug!("Registering TensorRT execution provider");
                    (
                        builder.with_tensorrt(),
                        "GPU (TensorRT → CPU fallback)".to_string(),
                    )
                } else if cuda_available {
                    debug!("Registering CUDA execution provider");
                    (builder.with_cuda(), "GPU (CUDA → CPU fallback)".to_string())
                } else {
                    warn!("GPU requested but no GPU providers available, will use CPU");
                    (builder, "GPU requested (using CPU)".to_string())
                }
            }
            InferenceDevice::TensorRT => {
                info!("Requested device: TensorRT (strict mode)");
                if !tensorrt_available {
                    return Err(Error::TensorRTUnavailable);
                }
                debug!("Registering TensorRT execution provider (no fallback)");
                (builder.with_tensorrt(), "TensorRT only".to_string())
            }
            InferenceDevice::Cuda => {
                info!("Requested device: CUDA (strict mode)");
                if !cuda_available {
                    return Err(Error::CudaUnavailable);
                }
                debug!("Registering CUDA execution provider (no fallback)");
                (builder.with_cuda(), "CUDA only".to_string())
            }
            InferenceDevice::Cpu => {
                info!("Requested device: CPU");
                (builder, "CPU".to_string())
            }
        };

        let inner = builder.build().map_err(|e| Error::ClassifierBuild {
            reason: e.to_string(),
        })?;

        // Get the requested provider from the classifier
        let requested_provider = inner.requested_provider();
        debug!(
            "Classifier reports requested provider: {:?}",
            requested_provider
        );

        info!(
            "Loaded model: {:?}, sample_rate: {}, segment_duration: {}s, device: {}",
            inner.config().model_type,
            inner.config().sample_rate,
            inner.config().segment_duration,
            actual_device_msg
        );

        // Build optional range filter
        let range_filter = if let Some(ref rf_config) = range_filter_config {
            use crate::inference::range_filter::RangeFilter;
            Some(RangeFilter::from_config(
                &rf_config.meta_model_path,
                inner.labels(),
                rf_config.threshold,
            )?)
        } else {
            None
        };

        Ok(Self {
            inner,
            range_filter,
            range_filter_config,
            species_list,
        })
    }

    /// Get the model configuration.
    pub fn config(&self) -> &birdnet_onnx::ModelConfig {
        self.inner.config()
    }

    /// Get the expected sample rate for this model.
    pub fn sample_rate(&self) -> u32 {
        self.inner.config().sample_rate
    }

    /// Get the expected segment duration in seconds.
    pub fn segment_duration(&self) -> f32 {
        self.inner.config().segment_duration
    }

    /// Get the expected sample count per segment.
    pub fn sample_count(&self) -> usize {
        self.inner.config().sample_count
    }

    /// Run inference on a single audio segment.
    pub fn predict(&self, segment: &[f32]) -> Result<PredictionResult> {
        self.inner.predict(segment).map_err(|e| Error::Inference {
            reason: e.to_string(),
        })
    }

    /// Run inference on a batch of audio segments.
    pub fn predict_batch(&self, segments: &[&[f32]]) -> Result<Vec<PredictionResult>> {
        self.inner
            .predict_batch(segments)
            .map_err(|e| Error::Inference {
                reason: e.to_string(),
            })
    }

    /// Get the optional range filter.
    pub fn range_filter(&self) -> Option<&crate::inference::range_filter::RangeFilter> {
        self.range_filter.as_ref()
    }

    /// Apply range filtering to predictions if configured.
    ///
    /// Returns filtered predictions. If range filtering is not enabled, returns predictions unchanged.
    pub fn apply_range_filter(
        &self,
        mut predictions: Vec<PredictionResult>,
    ) -> Result<Vec<PredictionResult>> {
        if let (Some(range_filter), Some(rf_config)) = (
            self.range_filter.as_ref(),
            self.range_filter_config.as_ref(),
        ) {
            use tracing::debug;

            // Get location scores once for all predictions
            let location_scores = range_filter.predict(
                rf_config.latitude,
                rf_config.longitude,
                rf_config.month,
                rf_config.day,
            )?;

            debug!(
                "Range filter: applying to {} prediction results",
                predictions.len()
            );

            // Apply filtering to each prediction result
            for result in &mut predictions {
                let before_count = result.predictions.len();

                result.predictions = range_filter.filter_predictions(
                    &result.predictions,
                    &location_scores,
                    rf_config.rerank,
                );

                let after_count = result.predictions.len();
                if before_count != after_count {
                    debug!(
                        "Range filter: {} predictions before, {} after (filtered {})",
                        before_count,
                        after_count,
                        before_count - after_count
                    );
                }
            }
        } else if let Some(ref species_list) = self.species_list {
            use tracing::debug;

            debug!(
                "Species list filter: applying to {} prediction results",
                predictions.len()
            );

            // Apply species list filtering to each prediction result
            for result in &mut predictions {
                let before_count = result.predictions.len();

                result
                    .predictions
                    .retain(|p| species_list.contains(&p.species));

                let after_count = result.predictions.len();
                if before_count != after_count {
                    debug!(
                        "Species list filter: {} predictions before, {} after (filtered {})",
                        before_count,
                        after_count,
                        before_count - after_count
                    );
                }
            }
        }

        Ok(predictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::unwrap_used)]
    fn test_filter_predictions_with_species_list() {
        use birdnet_onnx::Prediction;

        let predictions = vec![
            Prediction {
                species: "Parus major_Great Tit".to_string(),
                confidence: 0.95,
                index: 0,
            },
            Prediction {
                species: "Turdus merula_Blackbird".to_string(),
                confidence: 0.85,
                index: 1,
            },
            Prediction {
                species: "Cyanistes caeruleus_Blue Tit".to_string(),
                confidence: 0.75,
                index: 2,
            },
        ];

        let species_list: HashSet<String> = vec![
            "Parus major_Great Tit".to_string(),
            "Cyanistes caeruleus_Blue Tit".to_string(),
        ]
        .into_iter()
        .collect();

        // Filter using the species list (now O(1) lookup)
        let filtered: Vec<Prediction> = predictions
            .iter()
            .filter(|p| species_list.contains(&p.species))
            .cloned()
            .collect();

        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().any(|p| p.species.contains("Parus major")));
        assert!(filtered.iter().any(|p| p.species.contains("Cyanistes")));
        assert!(!filtered.iter().any(|p| p.species.contains("Turdus")));
    }
}
