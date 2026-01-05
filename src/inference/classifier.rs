//! Inference classifier wrapper around birdnet-onnx.

use crate::config::{InferenceDevice, ModelConfig as BirdaModelConfig};
use crate::error::{Error, Result};
use birdnet_onnx::{
    Classifier, ClassifierBuilder, ExecutionProviderInfo, InferenceOptions, PredictionResult,
    available_execution_providers, ort_execution_providers,
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

        let builder = ClassifierBuilder::new()
            .model_path(model_config.path.to_string_lossy().to_string())
            .labels_path(model_config.labels.to_string_lossy().to_string())
            .top_k(top_k)
            .min_confidence(min_confidence);

        // Select and configure execution provider based on device setting
        // GPU provider priority order (shared by Auto and --gpu modes)
        //
        // This list includes general-purpose GPU acceleration providers.
        // Excluded from auto-selection:
        // - oneDNN: Intel CPU optimizer (not GPU acceleration)
        // - QNN: Qualcomm-specific hardware (mobile/edge devices only)
        // - ACL/ArmNN: ARM-specific devices only
        //
        // These specialized providers are available via explicit flags
        // (--onednn, --qnn, --acl, --armnn) for users with specific hardware.
        let gpu_priority = [
            (ExecutionProviderInfo::TensorRt, "TensorRT"),
            (ExecutionProviderInfo::Cuda, "CUDA"),
            (ExecutionProviderInfo::DirectMl, "DirectML"),
            (ExecutionProviderInfo::CoreMl, "CoreML"),
            (ExecutionProviderInfo::Rocm, "ROCm"),
            (ExecutionProviderInfo::OpenVino, "OpenVINO"),
        ];

        let (builder, actual_device_msg) = match device {
            InferenceDevice::Cpu => {
                info!("Requested device: CPU");
                (builder, "CPU")
            }
            InferenceDevice::Auto => {
                // Auto mode: try GPU providers in priority order, silent CPU fallback
                if let Some(&(provider_info, name)) = gpu_priority
                    .iter()
                    .find(|(p, _)| available_providers.contains(p))
                {
                    info!("Auto mode: {} available, attempting GPU", name);
                    let builder = add_execution_provider(builder, provider_info);
                    (builder, name)
                } else {
                    info!("Auto mode: No GPU providers available, using CPU");
                    (builder, "Auto (CPU)")
                }
            }
            InferenceDevice::Gpu => {
                // Best-effort GPU: try providers in priority order, warn if CPU fallback
                if let Some(&(provider_info, name)) = gpu_priority
                    .iter()
                    .find(|(p, _)| available_providers.contains(p))
                {
                    info!("--gpu: Selected {} provider", name);
                    let builder = add_execution_provider(builder, provider_info);
                    (builder, name)
                } else {
                    warn!("--gpu requested but no GPU providers available, using CPU");
                    (builder, "GPU (fallback to CPU)")
                }
            }
            // Explicit providers use the helper function
            InferenceDevice::Cuda => configure_explicit_provider(
                builder,
                &available_providers,
                ExecutionProviderInfo::Cuda,
                "CUDA",
            )?,
            InferenceDevice::TensorRt => configure_explicit_provider(
                builder,
                &available_providers,
                ExecutionProviderInfo::TensorRt,
                "TensorRT",
            )?,
            InferenceDevice::DirectMl => configure_explicit_provider(
                builder,
                &available_providers,
                ExecutionProviderInfo::DirectMl,
                "DirectML",
            )?,
            InferenceDevice::CoreMl => configure_explicit_provider(
                builder,
                &available_providers,
                ExecutionProviderInfo::CoreMl,
                "CoreML",
            )?,
            InferenceDevice::Rocm => configure_explicit_provider(
                builder,
                &available_providers,
                ExecutionProviderInfo::Rocm,
                "ROCm",
            )?,
            InferenceDevice::OpenVino => configure_explicit_provider(
                builder,
                &available_providers,
                ExecutionProviderInfo::OpenVino,
                "OpenVINO",
            )?,
            InferenceDevice::OneDnn => configure_explicit_provider(
                builder,
                &available_providers,
                ExecutionProviderInfo::OneDnn,
                "oneDNN",
            )?,
            InferenceDevice::Qnn => configure_explicit_provider(
                builder,
                &available_providers,
                ExecutionProviderInfo::Qnn,
                "QNN",
            )?,
            InferenceDevice::Acl => configure_explicit_provider(
                builder,
                &available_providers,
                ExecutionProviderInfo::Acl,
                "ACL",
            )?,
            InferenceDevice::ArmNn => configure_explicit_provider(
                builder,
                &available_providers,
                ExecutionProviderInfo::ArmNn,
                "ArmNN",
            )?,
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
    pub fn predict(&self, segment: &[f32], options: &InferenceOptions) -> Result<PredictionResult> {
        self.inner
            .predict(segment, options)
            .map_err(|e| Error::Inference {
                reason: e.to_string(),
            })
    }

    /// Run inference on a batch of audio segments.
    pub fn predict_batch(
        &self,
        segments: &[&[f32]],
        options: &InferenceOptions,
    ) -> Result<Vec<PredictionResult>> {
        self.inner
            .predict_batch(segments, options)
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

/// Configure an explicit execution provider (fail if unavailable).
fn configure_explicit_provider(
    builder: ClassifierBuilder,
    available_providers: &[ExecutionProviderInfo],
    provider_info: ExecutionProviderInfo,
    provider_name: &'static str,
) -> Result<(ClassifierBuilder, &'static str)> {
    if !available_providers.contains(&provider_info) {
        return Err(provider_unavailable_error(
            provider_name,
            available_providers,
        ));
    }
    info!("Requested device: {provider_name}");
    let builder = add_execution_provider(builder, provider_info);
    Ok((builder, provider_name))
}

/// Helper function to add execution provider to builder based on provider type.
fn add_execution_provider(
    builder: ClassifierBuilder,
    provider_info: ExecutionProviderInfo,
) -> ClassifierBuilder {
    use ort_execution_providers::{
        ACLExecutionProvider, ArmNNExecutionProvider, CUDAExecutionProvider,
        CoreMLExecutionProvider, DirectMLExecutionProvider, OneDNNExecutionProvider,
        OpenVINOExecutionProvider, QNNExecutionProvider, ROCmExecutionProvider,
    };

    match provider_info {
        ExecutionProviderInfo::Cuda => builder.execution_provider(CUDAExecutionProvider::default()),
        ExecutionProviderInfo::TensorRt => {
            // Use optimized TensorRT configuration (enables FP16, engine caching, timing cache)
            builder.with_tensorrt()
        }
        ExecutionProviderInfo::DirectMl => {
            builder.execution_provider(DirectMLExecutionProvider::default())
        }
        ExecutionProviderInfo::CoreMl => {
            builder.execution_provider(CoreMLExecutionProvider::default())
        }
        ExecutionProviderInfo::Rocm => builder.execution_provider(ROCmExecutionProvider::default()),
        ExecutionProviderInfo::OpenVino => {
            builder.execution_provider(OpenVINOExecutionProvider::default())
        }
        ExecutionProviderInfo::OneDnn => {
            builder.execution_provider(OneDNNExecutionProvider::default())
        }
        ExecutionProviderInfo::Qnn => builder.execution_provider(QNNExecutionProvider::default()),
        ExecutionProviderInfo::Acl => builder.execution_provider(ACLExecutionProvider::default()),
        ExecutionProviderInfo::ArmNn => {
            builder.execution_provider(ArmNNExecutionProvider::default())
        }
        ExecutionProviderInfo::Cpu => builder, // CPU doesn't need explicit provider
    }
}

/// Create a descriptive error for unavailable execution provider.
fn provider_unavailable_error(provider_name: &str, available: &[ExecutionProviderInfo]) -> Error {
    use std::fmt::Write;

    let mut message = format!("{provider_name} provider not available\n\n");
    message.push_str("Available providers:\n");

    for provider in available {
        let name = match provider {
            ExecutionProviderInfo::Cpu => "CPU",
            ExecutionProviderInfo::Cuda => "CUDA",
            ExecutionProviderInfo::TensorRt => "TensorRT",
            ExecutionProviderInfo::DirectMl => "DirectML",
            ExecutionProviderInfo::CoreMl => "CoreML",
            ExecutionProviderInfo::Rocm => "ROCm",
            ExecutionProviderInfo::OpenVino => "OpenVINO",
            ExecutionProviderInfo::OneDnn => "oneDNN",
            ExecutionProviderInfo::Qnn => "QNN",
            ExecutionProviderInfo::Acl => "ACL",
            ExecutionProviderInfo::ArmNn => "ArmNN",
        };
        let _ = writeln!(message, "  ✓ {name}");
    }

    message.push_str("\nTry one of:\n");
    message.push_str("  birda --cpu <input>     (use CPU)\n");
    message.push_str("  birda --gpu <input>     (auto-select best GPU)\n");
    message.push_str("  birda <input>           (auto mode with fallback)\n");

    Error::ClassifierBuild { reason: message }
}
