//! Inference classifier wrapper around birdnet-onnx.

use crate::config::{
    InferenceDevice, ModelConfig as BirdaModelConfig, ModelType, tensorrt_cache_dir,
};
use crate::error::{Error, Result};
use birdnet_onnx::{
    BatchInferenceContext, BsgPostProcessor, Classifier, ClassifierBuilder, ExecutionProviderInfo,
    InferenceOptions, LocationScore, PredictionResult, TensorRTConfig,
    available_execution_providers, ort_execution_providers,
};
use std::collections::HashSet;
use std::path::PathBuf;
use tracing::{debug, error, info, warn};

use super::get_tensorrt_library_name;

/// Tracks execution provider selection and fallback status.
#[derive(Debug, Clone)]
pub struct ExecutionProviderStatus {
    /// What the user requested ("auto", "gpu", "tensorrt", etc).
    pub requested: String,
    /// What execution provider is actually being used ("`TensorRT`", "CUDA", "CPU", etc).
    pub actual: String,
    /// Reason for fallback if we didn't use requested provider.
    pub fallback_reason: Option<String>,
}

/// Range filtering data that is always kept together.
///
/// Invariant: All three fields are present together. This struct encapsulates
/// the range filter, its configuration, and pre-computed location scores.
struct RangeFilterData {
    /// The range filter instance.
    filter: crate::inference::range_filter::RangeFilter,
    /// Range filter configuration parameters.
    config: crate::inference::RangeFilterConfig,
    /// Pre-computed location scores (computed once at initialization).
    /// Avoids recomputing scores on every batch (significant performance optimization).
    scores: Vec<LocationScore>,
}

/// Wrapper around birdnet-onnx Classifier with birda configuration.
pub struct BirdClassifier {
    inner: Classifier,
    /// Range filtering data (filter, config, and cached scores).
    /// All three components are present together or None.
    range_filter_data: Option<RangeFilterData>,
    /// Optional species list for filtering (from file).
    /// None if no species list file provided or if using dynamic range filtering.
    species_list: Option<HashSet<String>>,
    /// Whether `TensorRT` is being used (for warmup messaging).
    uses_tensorrt: bool,
    /// BSG post-processor (for BSG models only).
    bsg_processor: Option<BsgPostProcessor>,
    /// Execution provider status (requested, actual, fallback reason).
    ep_status: ExecutionProviderStatus,
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
        // - CoreML: Excluded on macOS due to poor ONNX Runtime support (use --coreml to force)
        //
        // These specialized providers are available via explicit flags
        // (--onednn, --qnn, --acl, --armnn, --coreml) for users with specific hardware.
        #[allow(unused_mut)]
        let mut gpu_priority = vec![
            (ExecutionProviderInfo::TensorRt, "TensorRT"),
            (ExecutionProviderInfo::Cuda, "CUDA"),
            (ExecutionProviderInfo::DirectMl, "DirectML"),
            (ExecutionProviderInfo::Rocm, "ROCm"),
            (ExecutionProviderInfo::OpenVino, "OpenVINO"),
        ];

        // Include CoreML in auto-selection only on non-macOS platforms
        // (macOS users can still use --coreml explicitly if needed)
        // Insert at position 3 to preserve original priority order (between DirectML and ROCm)
        #[cfg(not(target_os = "macos"))]
        gpu_priority.insert(3, (ExecutionProviderInfo::CoreMl, "CoreML"));

        let (builder, actual_device_msg, ep_status) = match device {
            InferenceDevice::Cpu => {
                info!("Requested device: CPU");
                (
                    builder,
                    "CPU",
                    ExecutionProviderStatus {
                        requested: "cpu".to_string(),
                        actual: "CPU".to_string(),
                        fallback_reason: None,
                    },
                )
            }
            InferenceDevice::Auto => {
                // Auto mode: try GPU providers in priority order, silent CPU fallback

                // Filter TensorRT if libraries not available
                let mut available_gpu_priority = gpu_priority.clone();
                if let Some(pos) = available_gpu_priority
                    .iter()
                    .position(|(p, _)| *p == ExecutionProviderInfo::TensorRt)
                    && !crate::inference::is_tensorrt_available()
                {
                    debug!(
                        "Auto mode: TensorRT in priority list but libraries not found, skipping"
                    );
                    available_gpu_priority.remove(pos);
                }

                if let Some(&(provider_info, name)) = available_gpu_priority
                    .iter()
                    .find(|(p, _)| available_providers.contains(p))
                {
                    info!("Auto mode: {} available, attempting GPU", name);
                    let builder = add_execution_provider(builder, provider_info);
                    (
                        builder,
                        name,
                        ExecutionProviderStatus {
                            requested: "auto".to_string(),
                            actual: name.to_string(),
                            fallback_reason: None,
                        },
                    )
                } else {
                    info!("Auto mode: No GPU providers available, using CPU");
                    (
                        builder,
                        "Auto (CPU)",
                        ExecutionProviderStatus {
                            requested: "auto".to_string(),
                            actual: "CPU".to_string(),
                            fallback_reason: Some("No GPU providers available".to_string()),
                        },
                    )
                }
            }
            InferenceDevice::Gpu => {
                // Best-effort GPU: try providers in priority order, warn if CPU fallback

                // Filter TensorRT if libraries not available
                let mut available_gpu_priority = gpu_priority.clone();
                let mut tensorrt_fallback = None;

                if let Some(pos) = available_gpu_priority
                    .iter()
                    .position(|(p, _)| *p == ExecutionProviderInfo::TensorRt)
                    && !crate::inference::is_tensorrt_available()
                {
                    warn!(
                        "TensorRT libraries not found ({})",
                        get_tensorrt_library_name()
                    );
                    warn!("TensorRT requires NVIDIA TensorRT 10.x runtime libraries");
                    warn!("Install from: https://developer.nvidia.com/tensorrt");
                    tensorrt_fallback = Some(format!(
                        "TensorRT libraries not found ({} missing)",
                        get_tensorrt_library_name()
                    ));
                    available_gpu_priority.remove(pos);
                }

                if let Some(&(provider_info, name)) = available_gpu_priority
                    .iter()
                    .find(|(p, _)| available_providers.contains(p))
                {
                    info!("--gpu: Selected {} provider", name);
                    let builder = add_execution_provider(builder, provider_info);
                    let fallback = tensorrt_fallback.inspect(|_| warn!("Falling back to {}", name));
                    (
                        builder,
                        name,
                        ExecutionProviderStatus {
                            requested: "gpu".to_string(),
                            actual: name.to_string(),
                            fallback_reason: fallback,
                        },
                    )
                } else {
                    warn!("--gpu requested but no GPU providers available, using CPU");
                    (
                        builder,
                        "GPU (fallback to CPU)",
                        ExecutionProviderStatus {
                            requested: "gpu".to_string(),
                            actual: "CPU".to_string(),
                            fallback_reason: Some("No GPU providers available".to_string()),
                        },
                    )
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
            InferenceDevice::Xnnpack => configure_explicit_provider(
                builder,
                &available_providers,
                ExecutionProviderInfo::Xnnpack,
                "XNNPACK",
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
            model_config.model_type,
            inner.config().sample_rate,
            inner.config().segment_duration,
            actual_device_msg
        );

        // Build optional range filter and compute location scores
        // Note: Range filter is not compatible with BSG models (different species set)
        let range_filter_data = if model_config.model_type == ModelType::BsgFinland {
            // BSG models use SDM for geographic/seasonal filtering, skip range filter
            None
        } else if let Some(rf_config) = range_filter_config {
            use crate::inference::range_filter::RangeFilter;

            let filter = RangeFilter::from_config(
                &rf_config.meta_model_path,
                inner.labels(),
                rf_config.threshold,
            )?;

            // Compute location scores once during initialization
            // If this fails, we want to fail classifier construction (not silently disable)
            let scores = filter.predict(
                rf_config.latitude,
                rf_config.longitude,
                rf_config.month,
                rf_config.day,
            )?;

            debug!(
                "Range filter: computed {} location scores for lat={:.4}, lon={:.4}, month={}, day={}",
                scores.len(),
                rf_config.latitude,
                rf_config.longitude,
                rf_config.month,
                rf_config.day
            );

            Some(RangeFilterData {
                filter,
                config: rf_config,
                scores,
            })
        } else {
            None
        };

        // Check if TensorRT is being used (for warmup messaging)
        let uses_tensorrt = requested_provider == birdnet_onnx::ExecutionProviderInfo::TensorRt;

        // Build BSG post-processor if this is a BSG model
        let bsg_processor = if model_config.model_type == ModelType::BsgFinland {
            // Calibration is required for BSG models
            let calibration =
                model_config
                    .bsg_calibration
                    .as_ref()
                    .ok_or_else(|| Error::BsgConfig {
                        message: "BSG model requires calibration file".to_string(),
                    })?;

            let mut builder = BsgPostProcessor::builder()
                .labels_path(model_config.labels.to_string_lossy().to_string())
                .calibration_path(calibration.to_string_lossy().to_string());

            // Add optional SDM files
            if let Some(migration) = &model_config.bsg_migration {
                builder = builder.migration_path(migration.to_string_lossy().to_string());
            }
            if let Some(maps) = &model_config.bsg_distribution_maps {
                builder = builder.distribution_maps_path(maps.to_string_lossy().to_string());
            }

            Some(builder.build().map_err(|e| match e {
                birdnet_onnx::Error::BsgCalibrationLoad(msg) => Error::BsgCalibration(msg),
                birdnet_onnx::Error::BsgMapsLoad(msg) => Error::BsgDistributionMaps(msg),
                other => Error::BsgConfig {
                    message: other.to_string(),
                },
            })?)
        } else {
            None
        };

        Ok(Self {
            inner,
            range_filter_data,
            species_list,
            uses_tensorrt,
            bsg_processor,
            ep_status,
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

    /// Check if `TensorRT` is being used.
    pub fn uses_tensorrt(&self) -> bool {
        self.uses_tensorrt
    }

    /// Get execution provider status (requested, actual, fallback reason).
    pub fn execution_provider_status(&self) -> &ExecutionProviderStatus {
        &self.ep_status
    }

    /// Perform a warm-up inference to initialize GPU resources.
    ///
    /// This method runs inference with the specified batch size to trigger any
    /// deferred initialization (such as `TensorRT` engine compilation). This should
    /// be called before the main processing loop to ensure that the inference
    /// watchdog doesn't kill the process during engine compilation.
    ///
    /// `TensorRT` builds separate optimized engines for each batch size, so the
    /// warmup must use the same batch size as the actual inference runs.
    ///
    /// `TensorRT` engine compilation can take several minutes on first run, but
    /// the compiled engine is cached for subsequent runs.
    pub fn warmup(&self, batch_size: usize) -> Result<()> {
        let sample_count = self.inner.config().sample_count;
        let dummy_segment = vec![0.0f32; sample_count];
        let options = InferenceOptions::default();

        if batch_size <= 1 {
            // Single inference warmup
            self.inner
                .predict(&dummy_segment, &options)
                .map_err(|e| Error::Inference {
                    reason: format!("warmup inference failed: {e}"),
                })?;
        } else {
            // Batch inference warmup - TensorRT needs to build engine for this batch size
            let segments = vec![dummy_segment.as_slice(); batch_size];
            self.inner
                .predict_batch(&segments, &options)
                .map_err(|e| Error::Inference {
                    reason: format!("warmup batch inference failed: {e}"),
                })?;
        }

        Ok(())
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

    /// Check if this classifier has BSG post-processing enabled.
    pub fn has_bsg_processor(&self) -> bool {
        self.bsg_processor.is_some()
    }

    /// Apply BSG post-processing to a prediction result.
    ///
    /// For BSG models, applies per-species calibration (always) and optionally
    /// Species Distribution Model (SDM) adjustment if location and date are provided.
    ///
    /// For non-BSG models, returns the result unchanged.
    ///
    /// # Arguments
    ///
    /// * `result` - Prediction result from classifier
    /// * `lat` - Optional latitude for SDM adjustment
    /// * `lon` - Optional longitude for SDM adjustment
    /// * `day_of_year` - Optional day of year (1-366) for SDM adjustment
    pub fn apply_bsg_postprocessing(
        &self,
        result: PredictionResult,
        lat: Option<f32>,
        lon: Option<f32>,
        day_of_year: Option<u32>,
    ) -> Result<PredictionResult> {
        let Some(bsg) = &self.bsg_processor else {
            return Ok(result); // Not a BSG model
        };

        if let (Some(lat), Some(lon), Some(day)) = (lat, lon, day_of_year) {
            // Apply calibration + SDM
            bsg.process(&result, lat, lon, day).map_err(|e| match e {
                birdnet_onnx::Error::BsgProcessing(msg) => Error::BsgConfig { message: msg },
                birdnet_onnx::Error::InvalidDayOfYear { day_of_year } => Error::BsgConfig {
                    message: format!("invalid day of year: {day_of_year} (must be 1-366)"),
                },
                other => Error::Inference {
                    reason: other.to_string(),
                },
            })
        } else {
            // Apply calibration only
            bsg.calibrate(&result).map_err(|e| Error::Inference {
                reason: e.to_string(),
            })
        }
    }

    /// Create a batch inference context for efficient repeated batch inference.
    ///
    /// Pre-allocates GPU memory for the specified batch size. Use this when processing
    /// many batches of audio segments to avoid memory growth issues on GPU.
    ///
    /// # Arguments
    ///
    /// * `max_batch_size` - Maximum number of segments per batch
    ///
    /// # Supported Models
    ///
    /// Currently supports `BirdNET` v2.4 and v3.0 only. Returns an error for `PerchV2`.
    pub fn create_batch_context(&self, max_batch_size: usize) -> Result<BatchInferenceContext> {
        self.inner
            .create_batch_context(max_batch_size)
            .map_err(|e| Error::Inference {
                reason: format!("failed to create batch context: {e}"),
            })
    }

    /// Run inference on a batch of audio segments using a pre-allocated context.
    ///
    /// This method reuses GPU memory from the context, preventing memory growth
    /// across repeated batch inference calls.
    pub fn predict_batch_with_context(
        &self,
        context: &mut BatchInferenceContext,
        segments: &[&[f32]],
        options: &InferenceOptions,
    ) -> Result<Vec<PredictionResult>> {
        self.inner
            .predict_batch_with_context(context, segments, options)
            .map_err(|e| Error::Inference {
                reason: e.to_string(),
            })
    }

    /// Get the optional range filter.
    pub fn range_filter(&self) -> Option<&crate::inference::range_filter::RangeFilter> {
        self.range_filter_data.as_ref().map(|data| &data.filter)
    }

    /// Apply range filtering to predictions if configured.
    ///
    /// Returns filtered predictions. If range filtering is not enabled, returns predictions unchanged.
    pub fn apply_range_filter(
        &self,
        mut predictions: Vec<PredictionResult>,
    ) -> Result<Vec<PredictionResult>> {
        if let Some(rf_data) = &self.range_filter_data {
            use tracing::debug;

            debug!(
                "Range filter: applying to {} prediction results",
                predictions.len()
            );

            // Apply filtering to each prediction result
            for result in &mut predictions {
                let before_count = result.predictions.len();

                result.predictions = rf_data.filter.filter_predictions(
                    &result.predictions,
                    &rf_data.scores,
                    rf_data.config.rerank,
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

    #[test]
    fn test_execution_provider_status_creation() {
        let status = ExecutionProviderStatus {
            requested: "auto".to_string(),
            actual: "CUDA".to_string(),
            fallback_reason: Some("TensorRT libraries not found".to_string()),
        };

        assert_eq!(status.requested, "auto");
        assert_eq!(status.actual, "CUDA");
        assert!(status.fallback_reason.is_some());
    }

    #[test]
    fn test_execution_provider_status_no_fallback() {
        let status = ExecutionProviderStatus {
            requested: "cuda".to_string(),
            actual: "CUDA".to_string(),
            fallback_reason: None,
        };

        assert_eq!(status.requested, "cuda");
        assert_eq!(status.actual, "CUDA");
        assert!(status.fallback_reason.is_none());
    }
}

/// Configure an explicit execution provider (fail if unavailable).
fn configure_explicit_provider(
    builder: ClassifierBuilder,
    available_providers: &[ExecutionProviderInfo],
    provider_info: ExecutionProviderInfo,
    provider_name: &'static str,
) -> Result<(ClassifierBuilder, &'static str, ExecutionProviderStatus)> {
    if !available_providers.contains(&provider_info) {
        return Err(provider_unavailable_error(
            provider_name,
            available_providers,
        ));
    }

    // Check TensorRT libraries if this is TensorRT
    if provider_info == ExecutionProviderInfo::TensorRt
        && !crate::inference::is_tensorrt_available()
    {
        warn!(
            "TensorRT libraries not found ({})",
            get_tensorrt_library_name()
        );
        warn!("TensorRT requires NVIDIA TensorRT 10.x runtime libraries");
        warn!("Install from: https://developer.nvidia.com/tensorrt");

        return Err(Error::ClassifierBuild {
            reason: format!(
                "TensorRT libraries not found ({} missing in library path). \
                 Install TensorRT 10.x runtime libraries from https://developer.nvidia.com/tensorrt",
                get_tensorrt_library_name()
            ),
        });
    }

    info!("Requested device: {provider_name}");
    let builder = add_execution_provider(builder, provider_info);
    let ep_status = ExecutionProviderStatus {
        requested: provider_name.to_lowercase(),
        actual: provider_name.to_string(),
        fallback_reason: None,
    };
    Ok((builder, provider_name, ep_status))
}

/// Setup `TensorRT` cache directory, returning the path if successful.
///
/// This function handles all the filesystem operations needed for `TensorRT` caching:
/// - Determines the platform-specific cache directory
/// - Validates the path is valid UTF-8 (required by `TensorRT` C++ backend)
/// - Creates the directory if it doesn't exist
///
/// Returns `None` if any step fails, with appropriate warning logs.
fn setup_tensorrt_cache() -> Option<PathBuf> {
    let cache_dir = match tensorrt_cache_dir() {
        Ok(dir) => dir,
        Err(e) => {
            warn!("Could not determine TensorRT cache directory: {}", e);
            return None;
        }
    };

    // Validate path is valid UTF-8 (required by TensorRT C++ backend)
    if cache_dir.to_str().is_none() {
        error!(
            "TensorRT cache path contains non-UTF-8 characters: {}, using default",
            cache_dir.display()
        );
        error!("TensorRT engines will be rebuilt on every run (significant performance impact)");
        return None;
    }

    // Create directory if it doesn't exist
    if let Err(e) = std::fs::create_dir_all(&cache_dir) {
        error!(
            "Failed to create TensorRT cache directory {}: {}, using default",
            cache_dir.display(),
            e
        );
        error!("TensorRT engines will be rebuilt on every run (minutes vs seconds)");
        return None;
    }

    debug!("TensorRT cache directory: {}", cache_dir.display());
    Some(cache_dir)
}

/// Helper function to add execution provider to builder based on provider type.
fn add_execution_provider(
    builder: ClassifierBuilder,
    provider_info: ExecutionProviderInfo,
) -> ClassifierBuilder {
    use ort_execution_providers::{
        ACLExecutionProvider, ArmNNExecutionProvider, CoreMLExecutionProvider,
        DirectMLExecutionProvider, OneDNNExecutionProvider, OpenVINOExecutionProvider,
        QNNExecutionProvider, ROCmExecutionProvider,
    };

    match provider_info {
        ExecutionProviderInfo::Cuda => {
            // Use with_cuda() for safe memory defaults (SameAsRequested arena strategy)
            builder.with_cuda()
        }
        ExecutionProviderInfo::TensorRt => {
            // Use optimized TensorRT configuration with app-specific cache directory
            let config = setup_tensorrt_cache().map_or_else(TensorRTConfig::new, |cache_dir| {
                // UTF-8 validated in setup_tensorrt_cache; panic if invariant violated
                #[allow(clippy::expect_used)]
                let cache_path = cache_dir
                    .to_str()
                    .expect("UTF-8 validated in setup_tensorrt_cache");
                TensorRTConfig::new()
                    .with_engine_cache_path(cache_path)
                    .with_timing_cache_path(cache_path)
            });
            builder.with_tensorrt_config(config)
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
        ExecutionProviderInfo::Xnnpack => builder.with_xnnpack(),
        // CPU is handled by not calling this function at all (default builder behavior).
        // Unknown/future providers fall back to CPU with a warning.
        _ => {
            warn!(
                "Unknown execution provider {:?}, using CPU fallback",
                provider_info
            );
            builder
        }
    }
}

/// Create a descriptive error for unavailable execution provider.
fn provider_unavailable_error(provider_name: &str, available: &[ExecutionProviderInfo]) -> Error {
    use std::fmt::Write;

    let mut message = format!("{provider_name} provider not available\n\n");
    message.push_str("Available providers:\n");

    for provider in available {
        let _ = writeln!(message, "  ✓ {}", super::provider_metadata(*provider).name);
    }

    message.push_str("\nTry one of:\n");
    message.push_str("  birda --cpu <input>     (use CPU)\n");
    message.push_str("  birda --gpu <input>     (auto-select best GPU)\n");
    message.push_str("  birda <input>           (auto mode with fallback)\n");

    Error::ClassifierBuild { reason: message }
}
