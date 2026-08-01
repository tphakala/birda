//! Inference classifier wrapper around birdnet-onnx.

use crate::config::{
    InferenceDevice, ModelConfig as BirdaModelConfig, ModelType, tensorrt_cache_dir,
};
use crate::error::{Error, Result};
use crate::inference::geomodel::{GeomodelScores, SpeciesMapping};
use crate::inference::geomodel_filter::{FilterSettings, filter_predictions};
use birdnet_onnx::{
    BatchInferenceContext, BsgPostProcessor, Classifier, ClassifierBuilder, ExecutionProviderInfo,
    InferenceOptions, PredictionResult, TensorRTConfig, available_execution_providers,
    ort_execution_providers,
};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Mutex;
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

/// Range filtering data, computed once at initialization.
///
/// The `birdnet_onnx::RangeFilter` itself is not retained: it is queried once
/// for the configured location and date, and its scores are projected into the
/// classifier's label space, after which the ONNX session has no further use.
struct RangeFilterData {
    /// Geomodel scores, keyed by the classifier's own labels.
    scores: GeomodelScores,
    /// Threshold, unmatched-species policy, and rerank flag.
    settings: FilterSettings,
    /// Coverage counts, for reporting.
    summary: MappingSummary,
}

/// How much of the classifier's label set the geomodel covers.
#[derive(Debug, Clone, Copy)]
pub struct MappingSummary {
    /// Classifier species that have a geomodel entry.
    pub mapped: usize,
    /// Classifier species with no geomodel entry.
    pub unmatched: usize,
    /// Total classifier species.
    pub total: usize,
    /// Mapped species scoring at or above the threshold at this location.
    pub in_range: usize,
}

impl MappingSummary {
    /// Summarize a mapping together with the projected scores.
    #[must_use]
    pub fn new(mapping: &SpeciesMapping, scores: &GeomodelScores, threshold: f32) -> Self {
        Self {
            mapped: mapping.mapped_count(),
            unmatched: mapping.unmatched_count(),
            total: mapping.total_classifier_species(),
            in_range: scores.in_range_count(threshold),
        }
    }
}

/// Validate that a geomodel labels file matches the model's output size.
///
/// A mismatch means the labels and the ONNX file came from different versions,
/// which `birdnet_onnx` would otherwise report as a bare label-count error.
fn validate_geomodel_labels(labels: &[String], expected: usize) -> Result<()> {
    if labels.len() == expected {
        return Ok(());
    }

    Err(Error::GeomodelLabelCount {
        expected,
        actual: labels.len(),
    })
}

/// Read a geomodel labels file, one `Scientific name_Common name` per line.
fn read_geomodel_labels(path: &std::path::Path) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path).map_err(|e| Error::LabelLoad {
        path: path.display().to_string(),
        reason: e.to_string(),
    })?;

    let labels: Vec<String> = content
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();

    if labels.is_empty() {
        return Err(Error::LabelLoad {
            path: path.display().to_string(),
            reason: "file contains no labels".to_string(),
        });
    }

    Ok(labels)
}

/// Query the geomodel once and project its scores onto the classifier's labels.
///
/// The filter is built from the geomodel's own labels rather than the
/// classifier's, because `birdnet_onnx` validates that the label count matches
/// the model's output size and no classifier has the geomodel's 12,012 classes.
/// It is queried with a zero threshold so every class comes back; thresholding
/// and the unmatched-species policy are applied later, in birda.
fn build_range_filter_data(
    rf_config: &crate::inference::RangeFilterConfig,
    classifier_labels: &[String],
) -> Result<RangeFilterData> {
    use crate::inference::range_filter::RangeFilter;

    let geomodel_labels = read_geomodel_labels(&rf_config.geomodel_labels_path)?;
    validate_geomodel_labels(
        &geomodel_labels,
        crate::constants::range_filter::GEOMODEL_SPECIES_COUNT,
    )?;

    let filter = RangeFilter::from_config(
        &rf_config.geomodel_path,
        &geomodel_labels,
        crate::constants::range_filter::GEOMODEL_QUERY_THRESHOLD,
    )?;

    let raw_scores = filter.predict(
        rf_config.latitude,
        rf_config.longitude,
        rf_config.month,
        rf_config.day,
    )?;

    let mapping = SpeciesMapping::build(&geomodel_labels, classifier_labels);
    let scores = GeomodelScores::project(&raw_scores, &mapping);
    let summary = MappingSummary::new(&mapping, &scores, rf_config.threshold);

    let policy_word = match rf_config.unmatched {
        crate::config::UnmatchedPolicy::Keep => "kept",
        crate::config::UnmatchedPolicy::Drop => "dropped",
    };
    info!(
        "BirdNET Geomodel v3.0.2 range filter: {} of {} classifier species mapped \
         ({} unmatched, {}); {} species in range at {:.4}, {:.4} month {} day {}",
        summary.mapped,
        summary.total,
        summary.unmatched,
        policy_word,
        summary.in_range,
        rf_config.latitude,
        rf_config.longitude,
        rf_config.month,
        rf_config.day
    );

    if rf_config.rerank && summary.unmatched > 0 {
        warn!(
            "--rerank is enabled: {} species with no BirdNET Geomodel v3.0.2 entry will be \
             excluded, because reranking has no occurrence probability to weight them by",
            summary.unmatched
        );
    }

    if summary.mapped == 0 {
        warn!(
            "No classifier species have BirdNET Geomodel v3.0.2 coverage; \
             range filtering will have no effect beyond the unmatched-species policy"
        );
    }

    Ok(RangeFilterData {
        scores,
        settings: FilterSettings {
            threshold: rf_config.threshold,
            unmatched: rf_config.unmatched,
            rerank: rf_config.rerank,
        },
        summary,
    })
}

/// Wrapper around birdnet-onnx Classifier with birda configuration.
pub struct BirdClassifier {
    inner: Classifier,
    /// Projected geomodel scores, filter settings and coverage counts.
    /// None when range filtering is not active for this run.
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
    /// Batch sizes this classifier has already been warmed up for.
    ///
    /// Kept here rather than in the pipeline because it is a property of this
    /// classifier's session, and a run processes files of differing lengths
    /// that each pick their own batch size.
    warmed: WarmupRegistry,
}

/// Which batch sizes a classifier has already been warmed up for.
///
/// Execution providers compile a graph per input shape, and the first
/// inference at a shape is the one that pays for it. Under `OpenVINO` that first
/// inference does not merely cost time: for `Perch` v2 it returns output
/// tensors that were never filled, so the batch is silently scored as noise.
/// Warming a shape absorbs that first run, leaving every batch the caller
/// actually cares about at least the second of its shape.
#[derive(Debug, Default)]
struct WarmupRegistry {
    sizes: Mutex<HashSet<usize>>,
}

impl WarmupRegistry {
    /// Whether `batch_size` has already been warmed.
    ///
    /// A poisoned registry means another thread panicked mid-warmup. The set
    /// is still readable and the worst case is a redundant warmup, so this
    /// recovers rather than turning it into a failed run.
    fn is_warm(&self, batch_size: usize) -> bool {
        self.sizes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(&batch_size)
    }

    /// Record `batch_size` as warmed.
    fn mark_warm(&self, batch_size: usize) {
        self.sizes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(batch_size);
    }
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

        let ProviderSelection {
            builder,
            device_name: actual_device_msg,
            status: ep_status,
        } = select_execution_provider(builder, device, &available_providers)?;

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

        // Build the range filter and project its scores into this classifier's
        // label space. Which models range filter at all is decided once, by
        // config::range_filter::supports_range_filter, and a caller that passes
        // a config here has already been through it. Re-deriving the rule here
        // is what let an earlier copy drift and omit the bat-mode exclusion.
        let range_filter_data = if let Some(rf_config) = range_filter_config {
            Some(build_range_filter_data(&rf_config, inner.labels())?)
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
            warmed: WarmupRegistry::default(),
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

    /// Get range filter info for reporting (geomodel coverage at this location).
    pub fn range_filter_info(&self) -> Option<crate::output::RangeFilterInfo> {
        self.range_filter_data
            .as_ref()
            .map(|data| crate::output::RangeFilterInfo {
                geomodel_version: crate::constants::range_filter::GEOMODEL_VERSION.to_string(),
                species_in_range: data.summary.in_range,
                total_species: data.summary.total,
                mapped_species: data.summary.mapped,
                unmatched_species: data.summary.unmatched,
                unmatched_policy: data.settings.unmatched.to_string(),
                threshold: data.settings.threshold,
            })
    }

    /// Warm up for `batch_size` unless that shape has already been warmed.
    ///
    /// Every batch size a run submits has to go through this before it carries
    /// real audio. Warming one shape does nothing for another: providers key
    /// their compiled graph on the input shape, so each distinct batch size
    /// pays its own first-inference cost, and under `OpenVINO` that first
    /// inference returns unpopulated outputs rather than merely being slow.
    ///
    /// Repeat calls for a shape already warmed return without running
    /// inference, so this is cheap to call once per file.
    pub fn ensure_warm(&self, batch_size: usize) -> Result<()> {
        if self.warmed.is_warm(batch_size) {
            return Ok(());
        }

        self.warmup(batch_size)?;

        // Recorded only after the warmup succeeds. A failed warmup that was
        // recorded anyway would let the next caller skip straight to real
        // audio on a shape that was never warmed.
        self.warmed.mark_warm(batch_size);

        Ok(())
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
    ///
    /// Prefer [`Self::ensure_warm`], which skips shapes already warmed.
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
                // The bounds are read from `constants::day_of_year`, the pair
                // `cli::validators::parse_day_of_year` and `config::validate`
                // enforce (#340). This guard stays because the error it renders
                // comes from birdnet-onnx, so it is a library boundary rather
                // than a duplicate of the input checks.
                birdnet_onnx::Error::InvalidDayOfYear { day_of_year } => Error::BsgConfig {
                    message: format!(
                        "invalid day of year: {day_of_year} (must be {}-{})",
                        crate::constants::day_of_year::MIN,
                        crate::constants::day_of_year::MAX
                    ),
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

                result.predictions =
                    filter_predictions(&result.predictions, &rf_data.scores, rf_data.settings);

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

/// Holds the result of execution provider selection.
struct ProviderSelection {
    /// Builder with the chosen provider configured.
    builder: ClassifierBuilder,
    /// Human-readable name of the selected device (e.g. "CUDA", "CPU", "Auto (CPU)").
    device_name: &'static str,
    /// Status record for reporting requested vs. actual provider.
    status: ExecutionProviderStatus,
}

/// Select and configure the execution provider based on the requested device.
///
/// Handles the full priority logic for Auto and Gpu modes, library availability
/// checks, and delegates explicit provider arms to `configure_explicit_provider`.
fn select_execution_provider(
    builder: ClassifierBuilder,
    device: InferenceDevice,
    available_providers: &[ExecutionProviderInfo],
) -> Result<ProviderSelection> {
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

    let (builder, device_name, status) = match device {
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
                debug!("Auto mode: TensorRT in priority list but libraries not found, skipping");
                available_gpu_priority.remove(pos);
            }

            // Filter CUDA if libraries not available
            if let Some(pos) = available_gpu_priority
                .iter()
                .position(|(p, _)| *p == ExecutionProviderInfo::Cuda)
                && !crate::inference::is_cuda_available()
            {
                debug!("Auto mode: CUDA in priority list but libraries not found, skipping");
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
            let mut cuda_fallback = None;

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

            // Filter CUDA if libraries not available
            if let Some(pos) = available_gpu_priority
                .iter()
                .position(|(p, _)| *p == ExecutionProviderInfo::Cuda)
                && !crate::inference::is_cuda_available()
            {
                warn!("CUDA runtime libraries not found");
                warn!(
                    "Looking for: {}",
                    crate::inference::get_cuda_library_patterns().join(", ")
                );
                warn!("CUDA requires NVIDIA CUDA runtime libraries");
                warn!("Install from: https://developer.nvidia.com/cuda-downloads");
                cuda_fallback = Some("CUDA runtime libraries not found".to_string());
                available_gpu_priority.remove(pos);
            }

            if let Some(&(provider_info, name)) = available_gpu_priority
                .iter()
                .find(|(p, _)| available_providers.contains(p))
            {
                info!("--gpu: Selected {} provider", name);
                let builder = add_execution_provider(builder, provider_info);

                // Combine fallback reasons
                let fallback = match (tensorrt_fallback, cuda_fallback) {
                    (Some(tr), Some(cu)) => {
                        warn!("Falling back to {}", name);
                        Some(format!("{tr}; {cu}"))
                    }
                    (Some(tr), None) => {
                        warn!("Falling back to {}", name);
                        Some(tr)
                    }
                    (None, Some(cu)) => {
                        warn!("Falling back to {}", name);
                        Some(cu)
                    }
                    (None, None) => None,
                };

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
            available_providers,
            ExecutionProviderInfo::Cuda,
            "CUDA",
        )?,
        InferenceDevice::TensorRt => configure_explicit_provider(
            builder,
            available_providers,
            ExecutionProviderInfo::TensorRt,
            "TensorRT",
        )?,
        InferenceDevice::DirectMl => configure_explicit_provider(
            builder,
            available_providers,
            ExecutionProviderInfo::DirectMl,
            "DirectML",
        )?,
        InferenceDevice::CoreMl => configure_explicit_provider(
            builder,
            available_providers,
            ExecutionProviderInfo::CoreMl,
            "CoreML",
        )?,
        InferenceDevice::Rocm => configure_explicit_provider(
            builder,
            available_providers,
            ExecutionProviderInfo::Rocm,
            "ROCm",
        )?,
        InferenceDevice::OpenVino => configure_explicit_provider(
            builder,
            available_providers,
            ExecutionProviderInfo::OpenVino,
            "OpenVINO",
        )?,
        InferenceDevice::OneDnn => configure_explicit_provider(
            builder,
            available_providers,
            ExecutionProviderInfo::OneDnn,
            "oneDNN",
        )?,
        InferenceDevice::Qnn => configure_explicit_provider(
            builder,
            available_providers,
            ExecutionProviderInfo::Qnn,
            "QNN",
        )?,
        InferenceDevice::Acl => configure_explicit_provider(
            builder,
            available_providers,
            ExecutionProviderInfo::Acl,
            "ACL",
        )?,
        InferenceDevice::ArmNn => configure_explicit_provider(
            builder,
            available_providers,
            ExecutionProviderInfo::ArmNn,
            "ArmNN",
        )?,
        InferenceDevice::Xnnpack => configure_explicit_provider(
            builder,
            available_providers,
            ExecutionProviderInfo::Xnnpack,
            "XNNPACK",
        )?,
    };

    Ok(ProviderSelection {
        builder,
        device_name,
        status,
    })
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

    // Check CUDA libraries if this is CUDA
    if provider_info == ExecutionProviderInfo::Cuda && !crate::inference::is_cuda_available() {
        warn!("CUDA runtime libraries not found");
        warn!(
            "Looking for: {}",
            crate::inference::get_cuda_library_patterns().join(", ")
        );
        warn!("CUDA requires NVIDIA CUDA runtime libraries");
        warn!("Install from: https://developer.nvidia.com/cuda-downloads");

        return Err(Error::ClassifierBuild {
            reason: format!(
                "CUDA runtime libraries not found (looking for {} in library path). \
                 Install NVIDIA CUDA runtime libraries from https://developer.nvidia.com/cuda-downloads",
                crate::inference::get_cuda_library_patterns().join(", ")
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
    #[allow(deprecated)]
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
        #[allow(deprecated)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_registry_starts_cold_for_every_size() {
        let registry = WarmupRegistry::default();

        assert!(!registry.is_warm(1));
        assert!(!registry.is_warm(8));
        assert!(!registry.is_warm(16));
    }

    #[test]
    fn warmup_registry_reports_a_marked_size_as_warm() {
        let registry = WarmupRegistry::default();
        registry.mark_warm(8);

        assert!(registry.is_warm(8));
    }

    #[test]
    fn warmup_registry_keeps_sizes_independent() {
        // The bug this guards against: warming the configured batch size and
        // treating the whole classifier as warm, while a short file submits a
        // smaller batch that was never warmed. Each size stands alone.
        let registry = WarmupRegistry::default();
        registry.mark_warm(16);

        assert!(registry.is_warm(16));
        assert!(!registry.is_warm(8), "warming 16 must not vouch for 8");
        assert!(!registry.is_warm(1), "warming 16 must not vouch for 1");
    }

    #[test]
    fn warmup_registry_marking_twice_is_idempotent() {
        let registry = WarmupRegistry::default();
        registry.mark_warm(4);
        registry.mark_warm(4);

        assert!(registry.is_warm(4));
    }

    #[test]
    fn warmup_registry_survives_a_poisoned_lock() {
        // Recovery matters because a panic in one warmup would otherwise turn
        // every later `ensure_warm` into a failure, including for sizes that
        // were warmed successfully before the panic.
        let registry = WarmupRegistry::default();
        registry.mark_warm(8);

        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = registry.sizes.lock();
            {
                panic!("poison the registry");
            }
        }));
        assert!(poisoned.is_err(), "the test must actually poison the lock");
        assert!(registry.sizes.is_poisoned());

        assert!(registry.is_warm(8), "a poisoned registry still reads");
        registry.mark_warm(2);
        assert!(registry.is_warm(2), "a poisoned registry still records");
    }

    #[test]
    fn test_filter_predictions_with_species_list() {
        use birdnet_onnx::Prediction;

        let predictions = [
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
