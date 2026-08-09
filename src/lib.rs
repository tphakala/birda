//! Birda - Bird species detection CLI tool.
//!
//! This crate provides audio analysis capabilities using `BirdNET` and Perch models.

#![warn(missing_docs)]
// Test code is held to different rules than shipping code, and these four lints
// encode the difference.
//
// `unwrap`, `expect` and `panic` are how a test reports failure: a panic in a
// test is the assertion firing, not an unhandled error path, and rewriting them
// into propagated `Result`s would hide which assertion failed. The crate-level
// deny still applies to everything birda actually ships, because `cfg(test)` is
// false in that build.
//
// `float_cmp` is a false positive throughout this suite. Every exact float
// assertion here is on a value that is passed through rather than computed: a
// literal parsed from a file, a coordinate round-tripped through JSON, or a
// clip boundary clamped to a whole number. Exact equality is the assertion the
// test wants, and an epsilon would stop it catching a boundary that is off by a
// hair. Where a value IS computed, these tests already compare against an
// epsilon by hand.
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::float_cmp
    )
)]

pub mod audio;
pub mod cli;
pub mod clipper;
pub mod config;
pub mod constants;
pub mod error;
/// Maintenance tool that regenerates `registry.json` from the vendored manifests.
#[cfg(feature = "gen-registry")]
pub mod gen_registry;
pub mod gpu;
pub mod inference;
pub mod locking;
pub mod output;
pub mod pipeline;
pub mod registry;
pub mod update;
pub mod utils;

use clap::Parser;
use cli::{AnalyzeArgs, Cli, Command};
use config::{
    BatConfig, Config, InferenceDevice, ModelConfig, ModelType, OutputFormat, OutputMode,
    config_file_path, load_default_config, range_filter::build_range_filter_config,
};
use constants::DEFAULT_TOP_K;
use inference::BirdClassifier;
use output::{
    ConfigPathPayload, ConfigPayload, FileStatus, ModelCheckEntry, ModelCheckPayload, ModelDetails,
    ModelEntry, ModelInfoPayload, ModelInstalledPayload, ModelListPayload, ModelRemovedPayload,
    PipelineSummary, ProgressReporter, ProviderInfo, ProvidersPayload, ResultType, create_reporter,
    emit_json_result,
};
use pipeline::{
    ProcessCheck, ProcessingConfig, collect_input_files, output_dir_for, process_file,
    should_process,
};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{error, info, warn};

pub use error::{Error, Result};

/// Model name used for ad-hoc (CLI-specified) models.
const ADHOC_MODEL_NAME: &str = "<ad-hoc>";

/// Threshold in seconds to distinguish `TensorRT` engine build from cache load.
/// Warmup taking >= this long indicates `TensorRT` compiled a new engine.
const WARMUP_BUILD_THRESHOLD_SECS: u64 = 2;

/// Resolve model configuration using priority-based logic.
///
/// # Priority Order
///
/// 1. **Explicit Named Model** (`-m <name>` provided): Load from config, apply overrides
/// 2. **Explicit Ad-hoc Model** (both `--model-type` AND `--model-path` provided): Build from CLI args
/// 3. **Implicit Default Model** (`defaults.model` set, no explicit model): Load default
/// 4. **Incomplete Ad-hoc** (`--model-path` but no `--model-type`): Error
/// 5. **No Model** (nothing specified): Error
fn resolve_model_config(args: &AnalyzeArgs, config: &Config) -> Result<(ModelConfig, String)> {
    // Priority 1: Explicit named model via -m
    if let Some(ref name) = args.model {
        let mut model_config = config::get_model(config, name)?.clone();

        // Warn if --model-type is also provided (will be ignored)
        if args.model_type.is_some() {
            warn!("--model-type is ignored when -m is provided (using model type from config)");
        }

        // Apply CLI overrides
        apply_model_overrides(&mut model_config, args);
        return Ok((model_config, name.clone()));
    }

    // Priority 2: Explicit ad-hoc model (requires both --model-type AND --model-path)
    if let (Some(model_type), Some(path)) = (args.model_type, &args.model_path) {
        let labels = args
            .labels_path
            .clone()
            .ok_or_else(|| Error::ConfigValidation {
                message: "--labels-path required when using --model-path with --model-type".into(),
            })?;

        let model_config = ModelConfig {
            registry_id: None,
            installed_version: None,
            installed_build: None,
            region: None,
            variant: None,
            path: path.clone(),
            labels,
            model_type,
            meta_model: None,
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
        };

        return Ok((model_config, ADHOC_MODEL_NAME.to_string()));
    }

    // Priority 3: Implicit default model from config
    if let Some(ref name) = config.defaults.model {
        let mut model_config = config::get_model(config, name)?.clone();

        // Warn if --model-type is also provided (will be ignored)
        if args.model_type.is_some() {
            warn!(
                "--model-type is ignored when using default model '{}' (provide both --model-path and --model-type to use ad-hoc mode)",
                name
            );
        }

        // Apply CLI overrides (allows patching default model)
        apply_model_overrides(&mut model_config, args);
        return Ok((model_config, name.clone()));
    }

    // Priority 4: Incomplete ad-hoc (has --model-path but no --model-type)
    if args.model_path.is_some() {
        return Err(Error::ConfigValidation {
            message: "--model-type required when using --model-path without -m".into(),
        });
    }

    // Priority 5: Nothing specified
    Err(Error::ConfigValidation {
        message: "no model specified (use -m, set defaults.model in config, or provide --model-path with --labels-path and --model-type)".into(),
    })
}

/// Resolve the shared geomodel for range filtering, if it is wanted at all.
///
/// Returns `None` when range filtering is not requested, or when the geomodel
/// is unavailable for any reason. The absent `Result` is deliberate: range
/// filtering turns on implicitly whenever coordinates are configured, so a
/// hard error would break every automated pipeline that upgrades to this
/// version before installing the geomodel. Making the signature infallible
/// means a future edit cannot reintroduce that failure mode by adding a `?`.
fn resolve_range_filter_geomodel(
    args: &AnalyzeArgs,
    config: &Config,
    model_config: &ModelConfig,
    output_mode: OutputMode,
) -> Option<registry::InstalledRangeFilter> {
    // Ask whether range filtering can apply at all before touching the
    // registry. Coordinates alone are not enough: a run also needs a time
    // parameter, and BSG and bat mode never range filter. Gating on
    // coordinates only would prompt for and download 15 MB that
    // build_range_filter_config then discards.
    if !config::range_filter::wants_range_filter(args, config, model_config.model_type) {
        return None;
    }

    // load_registry writes ~/.config/birda/registry.json, both to bootstrap it and
    // to perform the v3 to v4 rewrite this release triggers, so it can fail on an
    // unresolvable config dir. That must not abort analysis either: it is the
    // upgrade path itself.
    //
    // The read-only case no longer reaches here at all: `persist_registry` now
    // warns rather than propagating, because a cache it could not save is no reason
    // to fail a command. This branch still covers a config directory that cannot be
    // resolved and a bundled registry that will not parse.
    let registry = match registry::load_registry() {
        Ok(registry) => registry,
        Err(e) => {
            warn!("Range filtering disabled: {e}");
            return None;
        }
    };

    // Every failure here degrades to unfiltered analysis. Returning Err would
    // abort the whole run on a transient network blip, a stale cached registry,
    // or a checksum failure, which is exactly the pipeline break this function
    // exists to prevent. Only `birda species`, where the species list IS the
    // output, treats an unavailable geomodel as fatal.
    let resolution =
        match config::resolve_geomodel(args.geomodel_request(), config, &registry, output_mode) {
            Ok(resolution) => resolution,
            Err(e) => {
                warn!("Range filtering disabled: {e}");
                return None;
            }
        };

    match resolution {
        config::GeomodelResolution::Ready(geomodel) => Some(geomodel),
        config::GeomodelResolution::Unavailable(reason) => {
            warn!("Range filtering disabled: {reason}");
            None
        }
    }
}

/// Apply CLI overrides to a model configuration.
fn apply_model_overrides(model_config: &mut ModelConfig, args: &AnalyzeArgs) {
    if let Some(ref path) = args.model_path {
        model_config.path.clone_from(path);
    }
    if let Some(ref labels) = args.labels_path {
        model_config.labels.clone_from(labels);
    }
    if args.meta_model_path.is_some() {
        warn!(
            "--meta-model-path is deprecated and ignored; range filtering uses the \
             BirdNET Geomodel v3.0.2. Use --geomodel-path and --geomodel-labels-path \
             to point at a specific copy."
        );
    }
}

/// Determine optimal default batch size based on model type and execution provider.
///
/// Uses actual execution provider from classifier (not requested device) to handle
/// Auto/Gpu fallback scenarios correctly.
///
/// # Arguments
///
/// * `model_type` - Type of model being used
/// * `ep_status` - Execution provider status from classifier
///
/// # Returns
///
/// Recommended batch size optimized for the model and provider combination
fn determine_default_batch_size(
    model_type: ModelType,
    ep_status: &inference::ExecutionProviderStatus,
) -> usize {
    use ModelType::{BirdnetV24, BirdnetV30, BsgFinland, PerchV2};
    use constants::batch_size;

    // Match on model type and actual provider (not requested)
    match (model_type, ep_status.actual.as_str()) {
        // CPU - all models (explicit CPU, auto fallback, or GPU fallback)
        (_, "CPU") => batch_size::CPU,

        // CUDA with BirdNET 2.4 or BSG Finland
        (BirdnetV24 | BsgFinland, "CUDA") => batch_size::CUDA_BIRDNET_V24,

        // CUDA with BirdNET 3.0 or Perch v2
        (BirdnetV30 | PerchV2, "CUDA") => batch_size::CUDA_BIRDNET_V30,

        // TensorRT with any model
        (_, "TensorRT") => batch_size::TENSORRT,

        // Other GPU providers (DirectML, ROCm, OpenVINO, CoreML, etc.)
        _ => {
            tracing::debug!(
                "Using conservative batch size {} for model {:?} with provider {}",
                batch_size::OTHER_GPU,
                model_type,
                ep_status.actual
            );
            batch_size::OTHER_GPU
        }
    }
}

/// Parameters for file processing.
#[allow(clippy::struct_excessive_bools)]
struct ProcessingParams<'a> {
    formats: &'a [OutputFormat],
    output_dir: Option<&'a Path>,
    min_confidence: f32,
    overlap: f32,
    batch_size: usize,
    csv_columns: &'a [String],
    csv_bom: bool,
    model_name: &'a str,
    range_filter_params: Option<(f64, f64, u8)>,
    force: bool,
    fail_fast: bool,
    progress_enabled: bool,
    stdout_mode: bool,
    /// Dual output mode: progress events to stdout, detections to files.
    /// When true, reporter receives progress events but not detection events.
    dual_output_mode: bool,
    /// BSG SDM parameters: (latitude, longitude, `day_of_year`)
    /// `day_of_year` is None for auto-detection from file timestamp
    bsg_params: Option<(f64, f64, Option<u32>)>,
    /// Optional custom classifier for two-stage inference (bat detection).
    custom_classifier: Option<&'a birdnet_onnx::CustomClassifier>,
    /// Reclaim a per-file lock older than this before processing, if set.
    /// Backs `--stale-lock-timeout`; `None` leaves every lock in place.
    stale_lock_timeout: Option<std::time::Duration>,
}

/// Statistics from processing all files.
#[derive(Debug, Default)]
struct ProcessingStats {
    processed: usize,
    skipped: usize,
    errors: usize,
    total_detections: usize,
    total_segments: usize,
    total_audio_duration: f64,
}

/// Main entry point for birda CLI.
pub fn run() -> Result<()> {
    let cli = Cli::parse();

    validate_analyze_args_preflight(&cli.inputs, &cli.analyze)?;

    // Initialize logging
    init_logging(cli.analyze.verbose, cli.analyze.quiet);

    // Install Ctrl+C handler to clean up lock files on interrupt
    if let Err(e) = ctrlc::set_handler(|| {
        locking::cleanup_all_locks();
        locking::cleanup_all_config_locks();
        std::process::exit(130); // 128 + SIGINT(2)
    }) {
        warn!("Failed to install Ctrl+C handler: {e}");
    }

    // Load configuration
    let config = load_default_config()?;

    // `load_config_file` only parses TOML and reports deprecated keys, so until
    // this call every rule in `config::validate` guarded `config set` alone.
    // That is the path where clap's value parsers have usually already checked
    // the value, and it left the hand-edited file, where a wrong value is most
    // likely, entirely unchecked.
    if command_requires_valid_config(cli.command.as_ref(), cli.inputs.is_empty()) {
        config::validate_config(&config)?;
    }

    // Determine output mode (CLI flag takes precedence over config)
    // Auto-enable NDJSON mode for stdout
    let output_mode = if cli.analyze.stdout {
        OutputMode::Ndjson
    } else {
        cli.output_mode.unwrap_or(config.output.default_format)
    };

    // Create reporter based on output mode
    let reporter: Arc<dyn ProgressReporter> = Arc::from(create_reporter(output_mode));

    // Initialize ONNX Runtime only for commands that will touch it. This keeps
    // non-inference commands like `clip` working without a runtime install.
    if command_requires_runtime(cli.command.as_ref(), cli.inputs.is_empty()) {
        inference::ensure_runtime_available()?;
    }

    // Handle subcommands.
    //
    // The geomodel request is read from the flattened root args rather than
    // from the subcommand, because the three flags it carries are `global`.
    // A subcommand that needs the geomodel therefore sees the user's flags
    // whichever side of the subcommand name they were written on.
    if let Some(command) = cli.command {
        let geomodel_request = cli.analyze.geomodel_request();
        return handle_command(command, &config, output_mode, &reporter, geomodel_request);
    }

    // Default: analyze files
    // Show help if no inputs provided
    if cli.inputs.is_empty() {
        cli::help::print_smart_help(&config);
        std::process::exit(0);
    }

    // Run analysis
    analyze_files(&cli.inputs, &cli.analyze, &config, output_mode, &reporter)
}

/// Whether a command must run against a configuration that passes validation.
///
/// Only the commands that turn `config.toml` values into a result are gated,
/// because those are the ones an invalid value corrupts. Everything else either
/// reads no configuration at all or exists to inspect and repair the thing that
/// is broken, and gating those would leave hand-editing as the only way out of
/// a state that hand-editing caused.
///
/// Enumerated rather than defaulted through a wildcard, deliberately and for
/// the same reason as [`command_requires_runtime`] below: a new `Command` must
/// not inherit an answer nobody chose. Adding a variant is a compile error here
/// until someone decides which side it belongs on.
///
/// The exemption widens what can be read, never what can be persisted: every
/// writer still validates inside [`config::save_config`].
fn command_requires_valid_config(command: Option<&Command>, has_no_inputs: bool) -> bool {
    match command {
        // Analysis is the only path that turns the whole of `[defaults]` into a
        // result, and it is the path the reported bug ran down. With no inputs
        // this falls through to help, which is the least useful moment to
        // refuse.
        None => !has_no_inputs,

        // Every subcommand is exempt, and each for a stated reason.
        //
        // `config` and `models` are the repair surfaces: one edits the file,
        // the other installs the model a dangling `defaults.model` names.
        // Gating them would leave hand-editing as the only way out of a state
        // that hand-editing caused, and `models install` is itself a repair.
        //
        // `providers`, `clip` and `update` receive no `Config` at all;
        // `handle_providers_command`, `clipper::command::execute` and
        // `handle_update_command` each take only an output mode. Refusing to
        // run the updater over a config it never reads would block the route to
        // a build that fixes the config handling.
        //
        // `species` takes `--lat` and `--lon` as required arguments, so it
        // never reads the coordinate defaults. Failing it on a stale
        // `defaults.latitude` would reject a run that could not have used it.
        Some(
            Command::Config { .. }
            | Command::Models { .. }
            | Command::Species { .. }
            | Command::Providers
            | Command::Clip(_)
            | Command::Update { .. },
        ) => false,
    }
}

fn command_requires_runtime(command: Option<&Command>, has_no_inputs: bool) -> bool {
    match command {
        Some(
            Command::Config { .. }
            | Command::Models { .. }
            | Command::Clip(_)
            | Command::Update { .. },
        ) => false,
        Some(Command::Providers | Command::Species { .. }) => true,
        None => !has_no_inputs,
    }
}

fn validate_analyze_args_preflight(inputs: &[PathBuf], args: &AnalyzeArgs) -> Result<()> {
    if args.stdout {
        // Must have exactly one input file
        if inputs.len() != 1 {
            return Err(Error::ConfigValidation {
                message: "--stdout requires exactly one input file".to_string(),
            });
        }
    }

    Ok(())
}

/// Resolve inference device from CLI flags or config default.
fn resolve_device(args: &AnalyzeArgs, config: &Config) -> InferenceDevice {
    [
        (args.gpu, InferenceDevice::Gpu),
        (args.cpu, InferenceDevice::Cpu),
        (args.cuda, InferenceDevice::Cuda),
        (args.tensorrt, InferenceDevice::TensorRt),
        (args.directml, InferenceDevice::DirectMl),
        (args.coreml, InferenceDevice::CoreMl),
        (args.rocm, InferenceDevice::Rocm),
        (args.openvino, InferenceDevice::OpenVino),
        (args.onednn, InferenceDevice::OneDnn),
        (args.qnn, InferenceDevice::Qnn),
        (args.acl, InferenceDevice::Acl),
        (args.armnn, InferenceDevice::ArmNn),
        (args.xnnpack, InferenceDevice::Xnnpack),
    ]
    .into_iter()
    .find(|(flag, _)| *flag)
    .map_or(config.inference.device, |(_, device)| device)
}

/// Load species list from file if no range filter is active.
///
/// Priority: range filter (dynamic) > species list file (static) > no filtering.
/// When range filter is active, species filtering comes from the range filter,
/// so we return `None` here.
fn resolve_species_filter(
    args: &AnalyzeArgs,
    config: &Config,
    has_range_filter: bool,
) -> Result<Option<HashSet<String>>> {
    if has_range_filter {
        // Dynamic filtering wins, but say so when the user explicitly asked for
        // a list. Range filtering used to require a configured meta model, so
        // a user without one kept their --slist; it now activates on
        // coordinates plus a time parameter alone and would silently override
        // the list they passed.
        if args.slist.is_some() {
            warn!(
                "Ignoring --slist: range filtering takes precedence when coordinates \
                 and a date are given. Drop --lat/--lon to use the species list instead."
            );
        }
        return Ok(None);
    }

    if let Some(slist_path) = args
        .slist
        .as_ref()
        .or(config.defaults.species_list_file.as_ref())
    {
        info!("Loading species list: {}", slist_path.display());
        let species = utils::species_list::read_species_list(slist_path)?
            .into_iter()
            .collect::<HashSet<_>>();
        info!(
            "Species list filter enabled: {} species loaded",
            species.len()
        );
        return Ok(Some(species));
    }

    Ok(None)
}

/// Validate that the model and labels files exist.
fn validate_model_files(model_config: &ModelConfig) -> Result<()> {
    if !model_config.path.exists() {
        return Err(Error::ModelFileNotFound {
            path: model_config.path.clone(),
        });
    }
    if !model_config.labels.exists() {
        return Err(Error::LabelsFileNotFound {
            path: model_config.labels.clone(),
        });
    }
    Ok(())
}

/// Warm up the classifier, with special `TensorRT` spinner handling.
///
/// `TensorRT` compiles/loads its engine during the first inference, which can
/// take several minutes on first run. This warmup triggers that initialization
/// before the main processing loop starts.
fn warmup_classifier(classifier: &BirdClassifier, batch_size: usize) -> Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};
    use std::time::{Duration, Instant};

    if !classifier.uses_tensorrt() {
        return classifier.ensure_warm(batch_size);
    }

    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner()),
    );
    spinner.set_message(format!(
        "TensorRT: Initializing engine for batch size {batch_size} (may take several minutes on first run)..."
    ));
    spinner.enable_steady_tick(Duration::from_millis(100));

    let warmup_start = Instant::now();
    let result = classifier.ensure_warm(batch_size);
    let warmup_duration = warmup_start.elapsed();

    spinner.finish_and_clear();
    result?;

    if warmup_duration.as_secs() >= WARMUP_BUILD_THRESHOLD_SECS {
        info!(
            "TensorRT: Engine built in {:.1}s (cached for future runs)",
            warmup_duration.as_secs_f64()
        );
    } else {
        info!(
            "TensorRT: Engine loaded from cache ({:.0}ms)",
            warmup_duration.as_secs_f64() * 1000.0
        );
    }

    Ok(())
}

/// Report final summary statistics.
///
/// Logs processing summary and performance metrics, then reports to the reporter.
fn report_summary(
    stats: &ProcessingStats,
    total_start: std::time::Instant,
    fail_fast: bool,
    reporter: &Arc<dyn ProgressReporter>,
) {
    use crate::output::progress;

    let elapsed = total_start.elapsed();
    let total_duration = elapsed.as_secs_f64();
    #[allow(clippy::cast_possible_truncation)]
    let duration_ms = elapsed.as_millis() as u64;

    info!(
        "Complete: {} processed, {} skipped, {} errors, {} total detections in {:.2}s",
        stats.processed, stats.skipped, stats.errors, stats.total_detections, total_duration
    );

    #[allow(clippy::cast_precision_loss)]
    let realtime_factor = if total_duration > 0.0 {
        stats.total_audio_duration / total_duration
    } else {
        0.0
    };

    if stats.processed > 0 {
        #[allow(clippy::cast_precision_loss)]
        let avg_segments_per_sec = if total_duration > 0.0 {
            stats.total_segments as f64 / total_duration
        } else {
            0.0
        };
        info!(
            "Performance: {:.1} segments/sec overall, {:.1}x realtime ({} total audio)",
            avg_segments_per_sec,
            realtime_factor,
            progress::format_duration(stats.total_audio_duration)
        );
    }

    if stats.errors > 0 && !fail_fast {
        warn!("{} file(s) had errors", stats.errors);
    }

    reporter.pipeline_completed(&PipelineSummary {
        files_processed: stats.processed,
        files_failed: stats.errors,
        files_skipped: stats.skipped,
        total_detections: stats.total_detections,
        total_segments: stats.total_segments,
        duration_ms,
        realtime_factor,
    });
}

/// Reclaim a per-file lock older than `timeout`, if a timeout is set.
///
/// A peer that crashed leaves its lock behind: the Ctrl+C handler calls
/// `process::exit` and runs no `Drop`. Without reclamation every later run skips
/// that file as locked, forever. `--stale-lock-timeout` opts in; `None` honours
/// every lock. Racing a live peer is safe: `FileLock::acquire` still creates with
/// `O_EXCL`, so a peer that re-locks in the gap wins and this file takes the
/// graceful `FileLocked` skip instead.
fn reclaim_stale_lock(input: &Path, output_dir: &Path, timeout: Option<std::time::Duration>) {
    let Some(max_age) = timeout else { return };
    if !crate::locking::FileLock::is_stale(input, output_dir, max_age) {
        return;
    }
    match crate::locking::FileLock::remove_stale(input, output_dir) {
        Ok(()) => info!("Removed stale lock: {}", input.display()),
        Err(e) => warn!("Could not remove stale lock for {}: {e}", input.display()),
    }
}

/// Process all files, updating stats in place.
///
/// On fail-fast error, returns `Err` immediately but `stats` contains partial results.
/// The caller is responsible for all summary reporting (success or failure).
fn process_all_files(
    files: &[PathBuf],
    classifier: &BirdClassifier,
    params: &ProcessingParams<'_>,
    reporter: &Arc<dyn ProgressReporter>,
    stats: &mut ProcessingStats,
) -> Result<()> {
    use crate::output::progress;

    let file_progress = progress::create_file_progress(files.len(), params.progress_enabled);

    for (index, file) in files.iter().enumerate() {
        let file_output_dir = output_dir_for(file, params.output_dir);

        // Reclaim a stale lock before the skip-locked check below (see the fn doc).
        reclaim_stale_lock(file, &file_output_dir, params.stale_lock_timeout);

        // Check if should process
        match should_process(
            file,
            &file_output_dir,
            params.formats,
            params.force,
            params.stdout_mode,
        ) {
            ProcessCheck::SkipExists => {
                info!("Skipping (output exists): {}", file.display());
                reporter.file_skipped(file, FileStatus::Skipped);
                stats.skipped += 1;
                progress::inc_progress(file_progress.as_ref());
                continue;
            }
            ProcessCheck::SkipLocked => {
                info!("Skipping (locked): {}", file.display());
                reporter.file_skipped(file, FileStatus::Locked);
                stats.skipped += 1;
                progress::inc_progress(file_progress.as_ref());
                continue;
            }
            ProcessCheck::Process => {}
        }

        // Get audio duration for progress estimation
        let audio_duration = crate::audio::get_audio_duration(file).ok().flatten();

        // Estimate segments for reporter; bat mode uses shorter segments
        let segment_duration = if params.custom_classifier.is_some() {
            crate::constants::bat::SEGMENT_DURATION
        } else {
            classifier.segment_duration()
        };
        let overlap = if params.custom_classifier.is_some() {
            crate::constants::bat::OVERLAP
        } else {
            params.overlap
        };
        #[allow(clippy::cast_possible_truncation)]
        let estimated_segments = progress::estimate_segment_count(
            audio_duration,
            segment_duration,
            overlap,
        )
        .unwrap_or(0) as usize;

        // Report file start
        reporter.file_started(file, index, estimated_segments, audio_duration);

        // Process the file
        let file_start = std::time::Instant::now();
        // Pass reporter in both stdout mode and dual output mode
        let reporter_ref = if params.stdout_mode || params.dual_output_mode {
            Some(reporter.as_ref() as &dyn crate::output::ProgressReporter)
        } else {
            None
        };
        let proc_config = ProcessingConfig {
            input_path: file,
            output_dir: &file_output_dir,
            formats: params.formats,
            min_confidence: params.min_confidence,
            overlap: params.overlap,
            batch_size: params.batch_size,
            csv_columns: params.csv_columns,
            progress_enabled: params.progress_enabled,
            csv_bom_enabled: params.csv_bom,
            model_name: params.model_name,
            range_filter_params: params.range_filter_params,
            bsg_params: params.bsg_params,
            reporter: reporter_ref,
            dual_output_mode: params.dual_output_mode,
            custom_classifier: params.custom_classifier,
            bat_mode: params.custom_classifier.is_some(),
        };
        match process_file(&proc_config, classifier) {
            Ok(result) => {
                #[allow(clippy::cast_possible_truncation)]
                let duration_ms = file_start.elapsed().as_millis() as u64;
                reporter.file_completed_success(file, result.detections, duration_ms);
                stats.processed += 1;
                stats.total_detections += result.detections;
                stats.total_segments += result.segments;
                stats.total_audio_duration += result.audio_duration_secs;
            }
            Err(e) => {
                if let Err(e) =
                    record_file_failure(file, e, stats, reporter.as_ref(), params.fail_fast)
                {
                    progress::finish_progress(file_progress, "Failed");
                    return Err(e);
                }
            }
        }
        progress::inc_progress(file_progress.as_ref());
    }

    progress::finish_progress(file_progress, "Complete");
    Ok(())
}

/// Reporter error code for a file that failed during processing; goes into the
/// JSON envelope's `error.code` field.
const PROCESSING_ERROR_CODE: &str = "processing_error";

/// Fold a per-file processing failure into the run stats, distinguishing the
/// check-to-use lock race (#344) from a genuine error.
///
/// `should_process` advisory-checks the per-input lock with `is_locked`, but the
/// lock is only actually taken much later, in `process_file`. Two runs started
/// on the same file can both pass the advisory check, and the loser then fails
/// `FileLock::acquire` with `Error::FileLocked`. That is the graceful "skip
/// locked" the pre-check exists to produce, not a processing failure: the
/// check-to-use window is wide (collection to when the file is reached, minutes
/// apart for a large batch), so counting it as an error, or aborting the whole
/// batch under `--fail-fast`, is the wrong outcome. Count it as a skip, exactly
/// as the pre-check does.
///
/// Returns `Err(e)` only when the run must abort: a genuine error under
/// `fail_fast`. A lost-lock skip and every non-fatal error return `Ok(())`.
fn record_file_failure(
    file: &Path,
    e: Error,
    stats: &mut ProcessingStats,
    reporter: &dyn ProgressReporter,
    fail_fast: bool,
) -> Result<()> {
    if matches!(e, Error::FileLocked { .. }) {
        info!("Skipping (locked): {}", file.display());
        reporter.file_skipped(file, FileStatus::Locked);
        stats.skipped += 1;
        return Ok(());
    }

    error!("Failed to process {}: {}", file.display(), e);
    reporter.file_completed_failure(file, PROCESSING_ERROR_CODE, &e.to_string());
    stats.errors += 1;
    if fail_fast {
        return Err(e);
    }
    Ok(())
}

/// Analyze input files with the given options.
fn analyze_files(
    inputs: &[PathBuf],
    args: &AnalyzeArgs,
    config: &Config,
    output_mode: OutputMode,
    reporter: &Arc<dyn ProgressReporter>,
) -> Result<()> {
    use std::time::Instant;

    let total_start = Instant::now();

    // Fail fast on configuration errors before scanning filesystem
    // Resolve model configuration using priority-based resolution
    let (model_config, model_name) = resolve_model_config(args, config)?;
    validate_model_files(&model_config)?;

    // Bat mode: validate backbone is BirdNET v2.4 and build custom classifier
    let bat_classifier: Option<birdnet_onnx::CustomClassifier> = if let Some(region) = args.bat {
        // Bat mode requires BirdNET v2.4 as the backbone (for embedding extraction)
        if model_config.model_type != ModelType::BirdnetV24 {
            return Err(Error::ConfigValidation {
                message: format!(
                    "--bat requires a BirdNET v2.4 model as backbone, but '{}' is {:?}",
                    model_name, model_config.model_type
                ),
            });
        }

        // Resolve bat models directory: <data_dir>/models/bat/
        let bat_models_dir = registry::models_dir()?.join("bat");
        let bat_config = BatConfig::resolve(region, &bat_models_dir)?;

        info!(
            "Bat mode: region={}, classifier={}",
            region,
            bat_config.classifier_path.display()
        );

        let cc = birdnet_onnx::CustomClassifier::builder()
            .model_path(&bat_config.classifier_path)
            .labels_path(&bat_config.labels_path)
            .build()
            .map_err(|e| Error::ClassifierBuild {
                reason: format!("failed to build bat classifier: {e}"),
            })?;

        info!(
            "Bat classifier loaded: {} classes, {}-dim embeddings",
            cc.num_classes(),
            cc.input_dim()
        );

        Some(cc)
    } else {
        None
    };

    // Collect input files only after config is validated
    let files = collect_input_files(inputs)?;
    if files.is_empty() {
        return Err(Error::NoValidAudioFiles);
    }

    info!("Found {} audio file(s) to process", files.len());

    // Resolve other settings
    let min_confidence = args
        .min_confidence
        .unwrap_or(config.defaults.min_confidence);

    // In bat mode, override overlap to the bat-specific value unless the user
    // explicitly provided one via CLI
    let overlap = if bat_classifier.is_some() && args.overlap.is_none() {
        info!(
            "Bat mode: using overlap {}s (override with --overlap)",
            constants::bat::OVERLAP
        );
        constants::bat::OVERLAP
    } else {
        args.overlap.unwrap_or(config.defaults.overlap)
    };

    // Store user's explicit batch size choice (if any)
    // CLI takes precedence, then config file, then smart default (calculated later)
    let requested_batch_size = args.batch_size.or(config.defaults.batch_size);

    let formats = args
        .format
        .clone()
        .unwrap_or_else(|| config.defaults.formats.clone());
    let output_dir = args.output_dir.clone();
    let force = args.force;
    let fail_fast = args.fail_fast;

    // Resolve device from command-line flags or config
    let device = resolve_device(args, config);

    // Reject a malformed threshold BEFORE resolving the geomodel, because
    // resolution may prompt for and download roughly 15 MB and aborting after
    // that spends the user's bandwidth to report something knowable up front.
    //
    // This is a hard error, unlike the availability failures described below,
    // and the distinction is deliberate. Those are transient (a network blip, a
    // stale cached registry, a checksum mismatch) and degrade to unfiltered
    // analysis so a pipeline survives them. A threshold outside 0.0-1.0 is
    // neither transient nor safe to continue past: continuing is precisely the
    // reported bug, where a negative value silently disabled filtering while
    // the JSON envelope still called it active, and NaN dropped every species.
    //
    // Gated on `wants_range_filter`, which no longer changes the outcome: since
    // #295 the config-file value is rejected at load and the flag is bounded by
    // `parse_confidence`, so a threshold reaching here is already in range. The
    // gate's original promise, that a stale threshold in a config not doing
    // range filtering cannot fail an unrelated run, does not survive that: an
    // out-of-range value now fails the run regardless, which is the same answer
    // `config set` has always given it.
    if config::range_filter::wants_range_filter(args, config, model_config.model_type) {
        config::range_filter::validate_threshold(args, config)?;
    }

    // Build range filter config, resolving the shared geomodel first.
    //
    // A geomodel that cannot be found or fetched disables range filtering with
    // a warning rather than failing the run: coordinates in config enable range
    // filtering implicitly, so erroring would break existing pipelines on
    // upgrade.
    let range_filter_config =
        match resolve_range_filter_geomodel(args, config, &model_config, output_mode) {
            Some(geomodel) => build_range_filter_config(args, config, &model_config, &geomodel)?,
            None => None,
        };

    // Log if range filtering is enabled
    if let Some(ref rf_config) = range_filter_config {
        info!(
            "Range filter enabled: lat={:.4}, lon={:.4}, month={}, day={}, threshold={:.3}{}",
            rf_config.latitude,
            rf_config.longitude,
            rf_config.month,
            rf_config.day,
            rf_config.threshold,
            if rf_config.rerank {
                ", rerank=true"
            } else {
                ""
            }
        );
    }

    // Resolve species list filter
    let species_list = resolve_species_filter(args, config, range_filter_config.is_some())?;

    // Extract range filter params and BSG params before moving range_filter_config
    #[allow(clippy::cast_possible_truncation)]
    let range_filter_params = range_filter_config.as_ref().map(|rf| {
        let week = crate::utils::date::date_to_week(rf.month, rf.day) as u8;
        (rf.latitude, rf.longitude, week)
    });

    // Build BSG SDM parameters (latitude, longitude, day_of_year)
    // day_of_year is None for auto-detection from file timestamp
    // Use same latitude/longitude as range filter if available
    let bsg_params = range_filter_config.as_ref().map_or_else(
        || {
            if let (Some(lat), Some(lon)) = (args.lat, args.lon) {
                let day_of_year = args.day_of_year.or(config.defaults.day_of_year);
                Some((lat, lon, day_of_year))
            } else {
                None
            }
        },
        |rf| {
            let day_of_year = args.day_of_year.or(config.defaults.day_of_year);
            Some((rf.latitude, rf.longitude, day_of_year))
        },
    );

    // Build classifier
    info!("Loading model: {}", model_name);
    let classifier = BirdClassifier::from_config(
        &model_config,
        device,
        min_confidence,
        DEFAULT_TOP_K,
        range_filter_config,
        species_list,
    )?;

    // Determine final batch size: user choice > smart default based on actual EP
    let batch_size = requested_batch_size.unwrap_or_else(|| {
        let default = determine_default_batch_size(
            model_config.model_type,
            classifier.execution_provider_status(),
        );
        info!(
            "Using default batch size {} for {} with {} provider",
            default,
            model_config.model_type,
            classifier.execution_provider_status().actual
        );
        default
    });

    // Warm up the classifier (handles TensorRT spinner internally)
    warmup_classifier(&classifier, batch_size)?;

    // Report pipeline start with execution provider info
    let ep_info = classifier.execution_provider_status().clone().into();
    let range_filter_info = classifier.range_filter_info();
    reporter.pipeline_started(
        files.len(),
        &model_name,
        min_confidence,
        &ep_info,
        range_filter_info.as_ref(),
    );

    // Build processing parameters
    let is_json_output = matches!(output_mode, OutputMode::Json | OutputMode::Ndjson);
    let progress_enabled = !args.quiet && !args.no_progress && !is_json_output;

    // Dual output mode: progress events to stdout + detections to files
    // Enabled when output_dir is set AND output_mode is NDJSON
    let dual_output_mode =
        output_dir.is_some() && matches!(output_mode, OutputMode::Ndjson) && !args.stdout;

    let params = ProcessingParams {
        formats: &formats,
        output_dir: output_dir.as_deref(),
        min_confidence,
        overlap,
        batch_size,
        csv_columns: &config.defaults.csv_columns.include,
        csv_bom: !args.no_csv_bom,
        model_name: &model_name,
        range_filter_params,
        force,
        fail_fast,
        progress_enabled,
        stdout_mode: args.stdout,
        dual_output_mode,
        bsg_params,
        custom_classifier: bat_classifier.as_ref(),
        stale_lock_timeout: args.stale_lock_timeout,
    };

    // Process all files - stats owned here so partial results available on fail-fast
    let mut stats = ProcessingStats::default();
    let result = process_all_files(&files, &classifier, &params, reporter, &mut stats);

    // analyze_files is sole authority for all reporting (success or failure)
    report_summary(&stats, total_start, fail_fast, reporter);

    // Propagate any error after reporting
    result
}

fn init_logging(verbose: u8, quiet: bool) {
    use tracing_subscriber::{EnvFilter, fmt};

    // Build filter string based on verbosity level.
    // ORT logging is suppressed by default because CUDA fallback is expected in auto mode.
    // Use -v to see ORT warnings, -vv for info, -vvv for full trace.
    let filter_str = if quiet {
        "warn,ort=off".to_string()
    } else {
        match verbose {
            0 => "info,ort=off".to_string(),
            1 => "debug,ort=warn".to_string(),
            2 => "trace,ort=info".to_string(),
            _ => "trace".to_string(), // -vvv: no ORT filter, full trace
        }
    };

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&filter_str));

    // Write logs to stderr to keep stdout clean for JSON output
    // Use try_init() to avoid panic if subscriber is already set (e.g., in tests)
    let _ = fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

fn handle_command(
    command: Command,
    config: &config::Config,
    output_mode: OutputMode,
    _reporter: &Arc<dyn ProgressReporter>,
    geomodel_request: config::GeomodelRequest<'_>,
) -> Result<()> {
    match command {
        Command::Config { action } => handle_config_command(action, output_mode),
        Command::Models { action } => {
            handle_models_command(action, config, output_mode, geomodel_request.assume_yes)
        }
        Command::Providers => {
            handle_providers_command(output_mode);
            Ok(())
        }
        Command::Species {
            output,
            lat,
            lon,
            week,
            month,
            day,
            threshold,
            sort,
            model,
        } => cli::species::generate_species_list(
            output,
            lat,
            lon,
            week,
            month,
            day,
            threshold,
            sort,
            model,
            output_mode,
            geomodel_request,
        ),
        Command::Clip(args) => clipper::command::execute(&args, output_mode),
        Command::Update { check } => handle_update_command(check, output_mode),
    }
}

fn handle_providers_command(output_mode: OutputMode) {
    use crate::inference::provider_metadata;
    use birdnet_onnx::available_execution_providers;

    let providers = available_execution_providers();

    // Build provider info list
    let provider_infos: Vec<ProviderInfo> = providers
        .iter()
        .map(|provider| {
            let meta = provider_metadata(*provider);
            ProviderInfo {
                id: meta.id.to_string(),
                name: meta.name.to_string(),
                description: meta.description.to_string(),
            }
        })
        .collect();

    // JSON/NDJSON output
    if output_mode.is_structured() {
        let payload = ProvidersPayload {
            result_type: ResultType::Providers,
            providers: provider_infos,
        };
        emit_json_result(&payload);
        return;
    }

    // Human-readable output
    println!("Available execution providers:");
    println!();

    for info in &provider_infos {
        println!("  ✓ {}", info.description);
    }

    println!();
    println!("Usage:");
    println!("  (default)      Auto-select (GPU if available, silent CPU fallback)");
    println!("  --cpu          Force CPU only");
    #[cfg(target_os = "macos")]
    println!(
        "  --gpu          Auto-select best GPU (TensorRT → CUDA → DirectML → ROCm → OpenVINO)"
    );
    #[cfg(not(target_os = "macos"))]
    println!(
        "  --gpu          Auto-select best GPU (TensorRT → CUDA → DirectML → CoreML → ROCm → OpenVINO)"
    );
    println!();
    println!("Explicit providers (fail if unavailable):");
    let explicit_providers = [
        ("cuda", "Use CUDA"),
        ("tensorrt", "Use TensorRT"),
        ("directml", "Use DirectML"),
        ("coreml", "Use CoreML"),
        ("rocm", "Use ROCm"),
        ("openvino", "Use OpenVINO"),
        ("onednn", "Use oneDNN"),
        ("qnn", "Use QNN"),
        ("acl", "Use ACL"),
        ("armnn", "Use ArmNN"),
        ("xnnpack", "Use XNNPACK"),
    ];
    for (flag, description) in explicit_providers {
        println!("  --{flag:<13} {description}");
    }
    println!();
    println!("Note: This shows compile-time availability. Runtime availability may");
    println!("      differ based on drivers and hardware. Check log output for actual");
    println!("      provider selection during inference.");
}

/// Handle the `update` subcommand.
///
/// Checks for a newer release on GitHub and optionally downloads and installs it.
fn handle_update_command(check_only: bool, output_mode: OutputMode) -> Result<()> {
    let runtime = tokio::runtime::Runtime::new().map_err(|e| Error::Internal {
        message: format!("Failed to create async runtime: {e}"),
    })?;

    let client = reqwest::Client::builder()
        .user_agent(format!("birda/{}", env!("CARGO_PKG_VERSION")))
        .connect_timeout(std::time::Duration::from_secs(
            update::constants::CONNECT_TIMEOUT_SECS,
        ))
        .timeout(std::time::Duration::from_secs(
            update::constants::DOWNLOAD_TIMEOUT_SECS,
        ))
        .build()
        .map_err(|e| Error::Internal {
            message: format!("Failed to create HTTP client: {e}"),
        })?;

    let check_result = runtime.block_on(update::check_for_update(&client))?;

    match check_result {
        update::UpdateCheck::UpToDate { version } => {
            if output_mode.is_structured() {
                emit_json_result(&serde_json::json!({
                    "result_type": "update_check",
                    "status": "up_to_date",
                    "version": version,
                }));
            } else {
                println!("birda is up to date (v{version})");
            }
            Ok(())
        }
        update::UpdateCheck::Available {
            current,
            available,
            manifest,
        } => {
            if check_only {
                if output_mode.is_structured() {
                    emit_json_result(&serde_json::json!({
                        "result_type": "update_check",
                        "status": "available",
                        "current_version": current,
                        "available_version": available,
                    }));
                } else {
                    println!("Update available: v{current} -> v{available}");
                    println!("Run 'birda update' to install.");
                }
                return Ok(());
            }

            let result = runtime.block_on(update::perform_update(&client, &manifest, &current))?;

            if output_mode.is_structured() {
                emit_json_result(&serde_json::json!({
                    "result_type": "update_result",
                    "status": "updated",
                    "old_version": result.old_version,
                    "new_version": result.new_version,
                    "backup_path": result.backup_path.as_ref().map(|p| p.display().to_string()),
                    "warnings": result.warnings,
                }));
            } else {
                println!("Verifying checksum... ok");
                if let Some(ref path) = result.backup_path {
                    println!("Previous version saved as {}", path.display());
                }
                println!(
                    "Updated birda v{} -> v{}",
                    result.old_version, result.new_version
                );

                for warning in &result.warnings {
                    println!("\nNote: {warning}");
                }
            }

            Ok(())
        }
    }
}

fn handle_config_command(action: cli::ConfigAction, output_mode: OutputMode) -> Result<()> {
    use cli::ConfigAction;

    match action {
        ConfigAction::Init => {
            let path = config_file_path()?;
            // Re-check existence and create the file under the config lock, so an
            // `init` racing a `config set` cannot clobber the set's file with a
            // fresh default (#313).
            let created = locking::with_config_lock(&path, || {
                if path.exists() {
                    Ok(false)
                } else {
                    config::save_config(&Config::default(), &path)?;
                    Ok(true)
                }
            })?;
            if created {
                println!("Created configuration file: {}", path.display());
                println!("\nNext steps:");
                println!(
                    "  birda models add <name> --path <model.onnx> --labels <labels.txt> --type <type> --default"
                );
            } else {
                println!("Configuration file already exists: {}", path.display());
                println!("Use 'birda models add' to add models.");
            }
            Ok(())
        }
        ConfigAction::Show => {
            let config = load_default_config()?;
            let config_path = config_file_path()?;

            // JSON/NDJSON output
            if output_mode.is_structured() {
                let config_json =
                    serde_json::to_value(&config).map_err(|e| Error::ConfigValidation {
                        message: format!("failed to serialize config to JSON: {e}"),
                    })?;
                let payload = ConfigPayload {
                    result_type: ResultType::Config,
                    config_path,
                    config: config_json,
                };
                emit_json_result(&payload);
                return Ok(());
            }

            // Human-readable output
            println!("{config:#?}");
            Ok(())
        }
        ConfigAction::Set { key, value } => handle_config_set(&key, &value, output_mode),
        ConfigAction::Path => {
            let path = config_file_path()?;

            // JSON/NDJSON output
            if output_mode.is_structured() {
                let payload = ConfigPathPayload {
                    result_type: ResultType::ConfigPath,
                    exists: path.exists(),
                    config_path: path,
                };
                emit_json_result(&payload);
                return Ok(());
            }

            // Human-readable output
            println!("{}", path.display());
            Ok(())
        }
    }
}

/// Apply a `cli::validators` parser to a `config set` value, naming the key.
///
/// These are the same parsers clap runs for the matching flags, so a value
/// typed at `config set` is refused for the same reason, and carries the same
/// rule message, as `--batch-size` or `--day-of-year`; this adds the key name
/// in front of it. The numeric arms this serves used to call a bare
/// `value.parse()` and lean on the whole-config validation inside
/// `save_config`, which knows nothing about the key the user typed and, for
/// `batch_size`, did not carry the upper bound at all (#312). The
/// `defaults.day_of_year` arm is new. The arms that store a string, a path or
/// an enum do not come through here and never parsed.
///
/// The key prefix is not only for the reader. It is what tells this layer's
/// rejection apart from `validate_defaults`', so a test for this arm cannot be
/// satisfied by the save-side check firing instead.
fn parse_config_value<T>(
    key: &str,
    value: &str,
    parse: impl Fn(&str) -> std::result::Result<T, String>,
) -> Result<T> {
    parse(value).map_err(|message| Error::ConfigValidation {
        message: format!("invalid value for '{key}': {message}"),
    })
}

fn handle_config_set(key: &str, value: &str, output_mode: OutputMode) -> Result<()> {
    let ((), config, config_path) = config::update_config(|config| {
        match key {
            "defaults.model" => {
                config.defaults.model = if value.is_empty() {
                    None
                } else {
                    Some(value.to_string())
                };
            }
            "defaults.min_confidence" => {
                config.defaults.min_confidence = if value.is_empty() {
                    config::DefaultsConfig::default().min_confidence
                } else {
                    parse_config_value(key, value, cli::validators::parse_confidence)?
                };
            }
            "defaults.overlap" => {
                config.defaults.overlap = if value.is_empty() {
                    config::DefaultsConfig::default().overlap
                } else {
                    parse_config_value(key, value, cli::validators::parse_overlap)?
                };
            }
            "defaults.latitude" => {
                config.defaults.latitude = if value.is_empty() {
                    None
                } else {
                    Some(parse_config_value(
                        key,
                        value,
                        cli::validators::parse_latitude,
                    )?)
                };
            }
            "defaults.longitude" => {
                config.defaults.longitude = if value.is_empty() {
                    None
                } else {
                    Some(parse_config_value(
                        key,
                        value,
                        cli::validators::parse_longitude,
                    )?)
                };
            }
            "defaults.batch_size" => {
                config.defaults.batch_size = if value.is_empty() {
                    None
                } else {
                    Some(parse_config_value(
                        key,
                        value,
                        cli::validators::parse_batch_size,
                    )?)
                };
            }
            // No arm existed for this key, so hand-editing config.toml was the only
            // way to set it, and that was the one route `validate_defaults` did not
            // check either (#312). Both halves are closed now.
            "defaults.day_of_year" => {
                config.defaults.day_of_year = if value.is_empty() {
                    None
                } else {
                    Some(parse_config_value(
                        key,
                        value,
                        cli::validators::parse_day_of_year,
                    )?)
                };
            }
            "defaults.range_threshold" => {
                config.defaults.range_threshold = if value.is_empty() {
                    config::DefaultsConfig::default().range_threshold
                } else {
                    // `parse_confidence` rather than a threshold-specific parser
                    // because `--range-threshold` already uses it: both are bounded
                    // by `confidence::MIN`/`MAX`, and a second parser would be a
                    // second copy of one rule. The message it produces says
                    // "confidence", which the key prefix disambiguates.
                    parse_config_value(key, value, cli::validators::parse_confidence)?
                };
            }
            // The three keys below are siblings of range_threshold above. They were
            // missed when the geomodel landed, so reaching them meant hand-editing
            // config.toml. (The GUI's only config write is `config set
            // defaults.model`, which reaches this same handler and is serialised
            // by the #313 lock like any other; it never sets these three.)
            "defaults.geomodel" => {
                config.defaults.geomodel = if value.is_empty() {
                    None
                } else {
                    Some(PathBuf::from(value))
                };
            }
            "defaults.geomodel_labels" => {
                config.defaults.geomodel_labels = if value.is_empty() {
                    None
                } else {
                    Some(PathBuf::from(value))
                };
            }
            "defaults.range_unmatched" => {
                config.defaults.range_unmatched = if value.is_empty() {
                    config::DefaultsConfig::default().range_unmatched
                } else {
                    // Parsed through clap's ValueEnum so `config set` accepts
                    // exactly the spellings `--range-unmatched` accepts, rather
                    // than a second, drifting list. The error message derives its
                    // list the same way, so adding a variant cannot leave the two
                    // out of step.
                    <config::UnmatchedPolicy as clap::ValueEnum>::from_str(value, false).map_err(
                        |_| {
                            let accepted =
                                <config::UnmatchedPolicy as clap::ValueEnum>::value_variants()
                                    .iter()
                                    .filter_map(clap::ValueEnum::to_possible_value)
                                    .map(|v| format!("'{}'", v.get_name()))
                                    .collect::<Vec<_>>()
                                    .join(" or ");
                            Error::ConfigValidation {
                                message: format!(
                                    "invalid value for '{key}': {value} (expected {accepted})"
                                ),
                            }
                        },
                    )?
                };
            }
            _ => {
                return Err(Error::InvalidConfigKey {
                    key: key.to_string(),
                });
            }
        }
        Ok(())
    })?;

    // `update_config` validates the whole config inside `save_config` before
    // writing, and holds the config lock across the load-mutate-save so a
    // concurrent writer cannot lose this edit (#313).
    //
    // Note the validation covers the whole config, not just the key being set,
    // so a file carrying two independent faults cannot be repaired one key at a
    // time here. That predates #295 and is tracked separately; the load gate is
    // scoped to analysis so it does not turn that into a lockout.

    if output_mode.is_structured() {
        let config_json = serde_json::to_value(&config).map_err(|e| Error::ConfigValidation {
            message: format!("failed to serialize config to JSON: {e}"),
        })?;
        let payload = ConfigPayload {
            result_type: ResultType::Config,
            config_path,
            config: config_json,
        };
        emit_json_result(&payload);
    } else {
        println!("Set '{key}' = '{value}'");
        println!("Configuration saved to: {}", config_path.display());
    }

    Ok(())
}

fn handle_models_command(
    action: cli::ModelsAction,
    config: &config::Config,
    output_mode: OutputMode,
    assume_yes: bool,
) -> Result<()> {
    use cli::ModelsAction;

    match action {
        ModelsAction::List => {
            // Build model entries
            let mut models: Vec<ModelEntry> = config
                .models
                .iter()
                .map(|(name, model)| {
                    let is_default = config.defaults.model.as_ref().is_some_and(|d| d == name);
                    ModelEntry {
                        id: name.clone(),
                        model_type: model.model_type.to_string(),
                        is_default,
                        path: Some(model.path.clone()),
                        labels_path: Some(model.labels.clone()),
                        registry_id: model.registry_id.clone(),
                        installed_version: model.installed_version.clone(),
                        installed_build: model.installed_build,
                        region: model.region.clone(),
                        variant: model.variant.clone(),
                    }
                })
                .collect();

            // Sort by ID for deterministic output
            models.sort_unstable_by(|a, b| a.id.cmp(&b.id));

            // JSON/NDJSON output
            if output_mode.is_structured() {
                let payload = ModelListPayload {
                    result_type: ResultType::ModelList,
                    models,
                };
                emit_json_result(&payload);
                return Ok(());
            }

            // Human-readable output
            if config.models.is_empty() {
                println!("No models configured.");
            } else {
                println!("Configured models:");
                for entry in &models {
                    println!(
                        "  {} ({}){}",
                        entry.id,
                        entry.model_type,
                        if entry.is_default { " [default]" } else { "" }
                    );
                }
            }
            Ok(())
        }
        ModelsAction::ListAvailable => {
            let registry = registry::load_registry()?;
            registry::list_available(&registry, output_mode);
            Ok(())
        }
        ModelsAction::Add {
            name,
            path,
            labels,
            r#type,
            default,
        } => handle_models_add(&name, &path, &labels, r#type, default),
        ModelsAction::Check => {
            // Leftover <name>.<pid>.part files from a download interrupted
            // before it finished, reported on both output paths so a directory
            // that looks clean but holds an abandoned download stays visible.
            let leftover_downloads = registry::models_dir()
                .and_then(|dir| registry::find_stale_part_files(&dir))
                .unwrap_or_default();

            // JSON/NDJSON output: collect all results then emit
            if output_mode.is_structured() {
                let models: Vec<ModelCheckEntry> = config
                    .models
                    .iter()
                    .map(|(name, model)| {
                        let result = config::validate_model_config(name, model);
                        ModelCheckEntry {
                            id: name.clone(),
                            valid: result.is_ok(),
                            error: result.err().map(|e| e.to_string()),
                        }
                    })
                    .collect();
                let payload = ModelCheckPayload {
                    result_type: ResultType::ModelCheck,
                    models,
                    geomodel: check_geomodel()?,
                    leftover_downloads,
                };
                emit_json_result(&payload);
                return Ok(());
            }

            // Human-readable output (fail on first invalid model)
            for (name, model) in &config.models {
                config::validate_model_config(name, model)?;
                println!("  {name}: OK");
            }

            report_geomodel_status(&check_geomodel()?);
            for path in &leftover_downloads {
                println!(
                    "  {} is a leftover partial download (safe to remove if no download is running)",
                    path.display()
                );
            }
            Ok(())
        }
        ModelsAction::Info { id, languages } => {
            // Try registry first
            let registry = registry::load_registry()?;

            // The geomodel lives in `registry.range_filter`, not
            // `registry.models`, so `find_model` below can never match it and
            // the fallback would die in `config::get_model` with "model not
            // found" for the exact id every error message and doc page tells
            // users to install. Handle it before the lookup, not inside
            // `show_info`, which that lookup would never reach.
            // Only the install handle, deliberately not `asset.id`. That is the
            // id `models install` accepts, the one every error message and doc
            // page names, and the one this command's own JSON reports. Taking
            // the registry's internal id here as well would mean `models info
            // birdnet-geomodel-v3` worked while `models install
            // birdnet-geomodel-v3` did not, which is a worse answer than a
            // single canonical handle.
            // Match the id FIRST, then require the asset. Testing the asset
            // first meant that a registry predating the geomodel fell through
            // to `find_model` and died in `config::get_model` with "model
            // 'geomodel' not found in configuration" -- exactly the message
            // this branch exists to prevent. `handle_models_install` already
            // orders it this way and reports `RangeFilterAssetMissing`, so the
            // two commands disagreed on the same registry state.
            if id == registry::GEOMODEL_INSTALL_ID {
                let asset = registry
                    .range_filter
                    .as_ref()
                    .ok_or(Error::RangeFilterAssetMissing)?;
                if output_mode.is_structured() {
                    let payload = ModelInfoPayload {
                        result_type: ResultType::ModelInfo,
                        model: ModelDetails {
                            id: registry::GEOMODEL_INSTALL_ID.to_string(),
                            // Distinguishes the shared range filter from a
                            // classifier, which is what a consumer needs in
                            // order not to offer it as a selectable model.
                            model_type: "range-filter".to_string(),
                            path: None,
                            labels_path: None,
                            source: "registry".to_string(),
                        },
                    };
                    emit_json_result(&payload);
                } else if languages {
                    // The geomodel ships one English labels file and has no
                    // language variants, so report that rather than rendering
                    // an empty language list that looks like a lookup failure.
                    println!("Range filter: {}", asset.name);
                    println!();
                    println!(
                        "The range filter has no label language variants. Species names in \
                         output come from the active classifier's own labels."
                    );
                } else {
                    registry::show_range_filter_info(asset);
                }
                return Ok(());
            }

            if let Some(reg_model) = registry::find_model(&registry, &id) {
                // JSON/NDJSON output for registry model
                if output_mode.is_structured() {
                    // Note: --languages flag doesn't apply to JSON output - we include all info
                    let payload = ModelInfoPayload {
                        result_type: ResultType::ModelInfo,
                        model: ModelDetails {
                            id: reg_model.id.clone(),
                            model_type: reg_model.model_type.clone(),
                            path: None,
                            labels_path: None,
                            source: "registry".to_string(),
                        },
                    };
                    emit_json_result(&payload);
                    return Ok(());
                }

                // Human-readable output
                if languages {
                    registry::show_languages(&registry, &id)?;
                } else {
                    registry::show_info(&registry, &id)?;
                }
            } else {
                // Fall back to configured model
                let model = config::get_model(config, &id)?;

                // JSON/NDJSON output
                if output_mode.is_structured() {
                    let payload = ModelInfoPayload {
                        result_type: ResultType::ModelInfo,
                        model: ModelDetails {
                            id: id.clone(),
                            model_type: model.model_type.to_string(),
                            path: Some(model.path.clone()),
                            labels_path: Some(model.labels.clone()),
                            source: "configured".to_string(),
                        },
                    };
                    emit_json_result(&payload);
                    return Ok(());
                }

                // Human-readable output
                println!("Model: {id}");
                println!("  Type: {}", model.model_type);
                println!("  Path: {}", model.path.display());
                println!("  Labels: {}", model.labels.display());
            }
            Ok(())
        }
        ModelsAction::Remove { name, purge } => {
            handle_models_remove(&name, purge, output_mode, assume_yes)
        }
        ModelsAction::Install {
            id,
            language,
            region,
            variant,
            default,
        } => handle_models_install(
            &id,
            language.as_deref(),
            region.as_deref(),
            variant.as_deref(),
            default,
            output_mode,
            assume_yes,
        ),
        ModelsAction::Regions { id } => {
            let registry = registry::load_registry()?;
            registry::show_regions(&registry, &id)
        }
        ModelsAction::Manifest { id } => {
            let registry = registry::load_registry()?;
            registry::show_manifest(&registry, &id, output_mode)
        }
    }
}

/// Handle the `models add` command.
fn handle_models_add(
    name: &str,
    path: &Path,
    labels: &Path,
    model_type: ModelType,
    set_default: bool,
) -> Result<()> {
    // Validate files exist
    if !path.exists() {
        return Err(Error::ModelFileNotFound {
            path: path.to_path_buf(),
        });
    }
    if !labels.exists() {
        return Err(Error::LabelsFileNotFound {
            path: labels.to_path_buf(),
        });
    }

    // Load, check, insert and save under the config lock. The existence check
    // and the insert must be atomic against a concurrent writer, so both live
    // inside the locked closure (#313).
    let ((), _config, config_path) = config::update_config(|config| {
        if config.models.contains_key(name) {
            return Err(Error::ModelAlreadyExists {
                name: name.to_string(),
            });
        }

        config.models.insert(
            name.to_string(),
            ModelConfig {
                registry_id: None,
                installed_version: None,
                installed_build: None,
                region: None,
                variant: None,
                path: path.to_path_buf(),
                labels: labels.to_path_buf(),
                model_type,
                meta_model: None,
                bsg_calibration: None,
                bsg_migration: None,
                bsg_distribution_maps: None,
            },
        );

        if set_default {
            config.defaults.model = Some(name.to_string());
        }

        Ok(())
    })?;

    // Print success message
    println!("Added model '{name}' ({model_type})");
    println!("  Model: {}", path.display());
    println!("  Labels: {}", labels.display());
    println!("  Default: {}", if set_default { "yes" } else { "no" });
    println!("\nConfiguration saved to: {}", config_path.display());

    Ok(())
}

/// Remove a model from config and handle default promotion.
///
/// Returns the removed `ModelConfig` and the new default name (if promoted).
/// This is the pure logic extracted for testability.
fn remove_model_from_config(
    config: &mut Config,
    name: &str,
) -> Result<(ModelConfig, Option<String>)> {
    let model = config
        .models
        .get(name)
        .ok_or_else(|| Error::ModelNotFound {
            name: name.to_string(),
        })?
        .clone();

    config.models.remove(name);

    // Handle default model promotion
    let was_default = config.defaults.model.as_ref().is_some_and(|d| d == name);
    let promoted = if was_default {
        config.defaults.model = config.models.keys().min().cloned();
        config.defaults.model.clone()
    } else {
        None
    };

    Ok((model, promoted))
}

/// Collect all file paths referenced by the remaining models in config.
fn referenced_model_paths(config: &Config) -> std::collections::HashSet<PathBuf> {
    let mut paths = std::collections::HashSet::new();
    for model in config.models.values() {
        paths.insert(model.path.clone());
        paths.insert(model.labels.clone());
        if let Some(ref p) = model.meta_model {
            paths.insert(p.clone());
        }
        if let Some(ref p) = model.bsg_calibration {
            paths.insert(p.clone());
        }
        if let Some(ref p) = model.bsg_migration {
            paths.insert(p.clone());
        }
        if let Some(ref p) = model.bsg_distribution_maps {
            paths.insert(p.clone());
        }
    }
    paths
}

/// Handle the `models remove` command.
fn handle_models_remove(
    name: &str,
    purge: bool,
    output_mode: OutputMode,
    assume_yes: bool,
) -> Result<()> {
    use std::io::Write;

    // Confirm before deleting files (skip in structured mode).
    //
    // `--yes` is honoured here because the flag is global and therefore appears
    // in this command's `--help`. A prompt that advertises the flag and then
    // ignores it is the same defect as a flag that never reaches the command.
    //
    // Prompted before the lock, not inside it: the prompt waits on the user, and
    // holding the config lock across that would block every other config write
    // for as long as the user takes to answer.
    if purge && !output_mode.is_structured() && !assume_yes {
        print!("This will delete model files for '{name}' from disk. Continue? [y/N]: ");
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Removal cancelled.");
            return Ok(());
        }
    }

    // Remove from config and handle default promotion under the config lock, so
    // the load-mutate-save is serialised against a concurrent writer (#313).
    // Files are deleted afterward, outside the lock: the config stays consistent
    // even if a delete fails, and deleting model files does not touch the config.
    let ((model, promoted), config, config_path) =
        config::update_config(|config| remove_model_from_config(config, name))?;

    // Build the structured-output payload once for reuse in both the error and success paths.
    let structured_payload = ModelRemovedPayload {
        result_type: ResultType::ModelRemoved,
        id: name.to_string(),
        purge_requested: purge,
        new_default: promoted.clone(),
    };

    // If purge, delete associated files not referenced by other models
    if purge {
        let still_referenced = referenced_model_paths(&config);

        let mut first_error: Option<(PathBuf, std::io::Error)> = None;
        for file in [
            Some(model.path),
            Some(model.labels),
            model.meta_model,
            model.bsg_calibration,
            model.bsg_migration,
            model.bsg_distribution_maps,
        ]
        .into_iter()
        .flatten()
        {
            if still_referenced.contains(&file) {
                if !output_mode.is_structured() {
                    println!("  Skipped (used by another model): {}", file.display());
                }
                continue;
            }
            match std::fs::remove_file(&file) {
                Ok(()) => {
                    if !output_mode.is_structured() {
                        println!("  Deleted: {}", file.display());
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    if !output_mode.is_structured() {
                        println!("  Skipped (not found): {}", file.display());
                    }
                }
                Err(e) => {
                    if !output_mode.is_structured() {
                        println!("  Failed to delete: {}", file.display());
                    }
                    if first_error.is_none() {
                        first_error = Some((file, e));
                    }
                }
            }
        }
        if let Some((path, source)) = first_error {
            // Emit structured output before returning the error so the GUI
            // knows the config change succeeded even though file cleanup failed.
            if output_mode.is_structured() {
                emit_json_result(&structured_payload);
            }
            return Err(Error::FileDeletionFailed { path, source });
        }
    }

    if output_mode.is_structured() {
        emit_json_result(&structured_payload);
    } else {
        if let Some(ref new_name) = promoted {
            println!("Default model changed to '{new_name}'.");
        } else if config.defaults.model.is_none() && config.models.is_empty() {
            println!(
                "Warning: no models remaining. Set a new default with `birda models install`."
            );
        }
        println!("Model '{name}' removed from configuration.");
        println!("Configuration saved to: {}", config_path.display());
    }

    Ok(())
}

/// Handle the `models install` command.
fn handle_models_install(
    id: &str,
    language: Option<&str>,
    region: Option<&str>,
    variant: Option<&str>,
    set_default: bool,
    output_mode: OutputMode,
    assume_yes: bool,
) -> Result<()> {
    use std::io::{IsTerminal, Write};

    // `interactive` means "a human is present and can be asked", nothing more.
    //
    // `assume_yes` is deliberately NOT folded in here. Doing so collapses three
    // states (cannot ask / pre-answered yes / must ask) into two, and the two
    // collapsed states need OPPOSITE answers at the "Set as default model?"
    // prompt below, whose interactive default is yes: `--yes` would have
    // answered no. Each prompt reads `assume_yes` for itself, the way
    // `config::geomodel::acquire` already does.
    let interactive = std::io::stdin().is_terminal() && !output_mode.is_structured();

    // Load registry
    let registry = registry::load_registry()?;

    // The shared range filter asset is installable under its own id, since it
    // is used by every classifier rather than belonging to any one of them.
    if id == registry::GEOMODEL_INSTALL_ID {
        return handle_geomodel_install(&registry, interactive, assume_yes, output_mode);
    }

    let model = registry::find_model(&registry, id)
        .ok_or_else(|| Error::ModelNotFoundInRegistry { id: id.to_string() })?;

    let config = load_default_config()?;

    // Resolve the variant before the licence prompt, so a typo in --region
    // fails immediately rather than after the user has accepted a licence for a
    // download that was never going to start.
    let choice = if model.is_variant_based() {
        let choice = registry::select_variant(
            model,
            region,
            variant,
            config.inference.device,
            &registry::SystemProbe,
        )?;

        if !output_mode.is_structured() {
            println!(
                "Selected variant {} ({}).",
                choice.variant.id,
                choice.reason.describe()
            );
            println!(
                "  {}, {}",
                registry::species_count_label(choice.variant.classes),
                config::geomodel::human_size(choice.variant.model.size_bytes)
            );
            println!();
        }

        Some(choice)
    } else {
        // Silently ignoring these would install the global model and leave the
        // user believing they had a regional one.
        if region.is_some() {
            return Err(Error::RegionsNotSupported {
                model_id: id.to_string(),
            });
        }
        if let Some(requested) = variant {
            return Err(Error::VariantNotFound {
                model_id: id.to_string(),
                variant: requested.to_string(),
                available: "none, this model publishes a single file".to_string(),
            });
        }
        None
    };

    // Prompt for license acceptance
    let asset = registry::LicensedAsset {
        name: &model.name,
        vendor: &model.vendor,
        version: &model.version,
        license: &model.license,
    };
    if !registry::prompt_license_acceptance(asset, interactive, assume_yes)? {
        if !output_mode.is_structured() {
            println!("Installation cancelled.");
        }
        return Ok(());
    }

    // Download model and labels (async operation)
    let runtime = tokio::runtime::Runtime::new().map_err(|e| Error::Internal {
        message: format!("Failed to create async runtime: {e}"),
    })?;

    let installed = match choice.as_ref() {
        Some(choice) => {
            runtime.block_on(async { registry::install_variant(model, choice.variant).await })?
        }
        None => runtime.block_on(async { registry::install_model(model, language).await })?,
    };

    // Ensure the shared range filter is present so a fresh install can range
    // filter immediately. A failure here is a warning, not an error: the
    // classifier itself installed fine and works without range filtering.
    if let Some(asset) = registry.range_filter.as_ref()
        && let Err(e) = runtime.block_on(registry::install_range_filter(asset))
    {
        warn!(
            "Could not install the BirdNET Geomodel v3.0.2 range filter: {e}. \
             Run 'birda models install geomodel' to retry."
        );
    }

    if !output_mode.is_structured() {
        println!();
        println!("Installation complete!");
        println!();
        println!("Model files saved to:");
        println!("  {}", installed.model.display());
        println!("  {}", installed.labels.display());
        if let Some(cal_path) = &installed.bsg_calibration {
            println!("  {}", cal_path.display());
        }
        if let Some(mig_path) = &installed.bsg_migration {
            println!("  {}", mig_path.display());
        }
        if let Some(maps_path) = &installed.bsg_distribution_maps {
            println!("  {}", maps_path.display());
        }
        println!();
    }

    // Prompt to set as default.
    //
    // `assume_yes` answers yes, matching this prompt's own interactive default
    // ([Y/n], bare Enter means yes). Folding `--yes` into `interactive` instead
    // would drop through to the `else` and answer NO, so the flag would mean
    // "accept" at the licence prompt and "decline" here, inside one command.
    // `assume_yes && interactive`, not `assume_yes` alone: where no prompt is
    // printed there is nothing to assume. In a pipe or under --output-mode json
    // this prompt never appears, so answering it "yes" would silently seize
    // `defaults.model` from a provisioning script that passed --yes only to
    // clear the licence gate. That stays at its prior `false`.
    let should_set_default = if set_default || (assume_yes && interactive) {
        true
    } else if interactive {
        print!("Set as default model? [Y/n]: ");
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        !input.trim().eq_ignore_ascii_case("n")
    } else {
        false
    };

    // Parse model_type from string
    let model_type: ModelType = model
        .model_type
        .parse()
        .map_err(|_| Error::InvalidModelType {
            value: model.model_type.clone(),
        })?;

    let model_path = installed.model.clone();
    let labels_path = installed.labels.clone();

    // Captured before the config closure below moves `installed.provenance` and
    // consumes `installed`. `selection_reason` comes from `choice`, which
    // records why the variant was picked; a legacy single-file install has no
    // `choice` and no provenance, so all three stay `None`.
    let resolved_region = installed.provenance.as_ref().and_then(|p| p.region.clone());
    let resolved_variant = installed
        .provenance
        .as_ref()
        .and_then(|p| p.variant.clone());
    let selection_reason = choice.as_ref().map(|c| c.reason.slug().to_string());

    // A regional install occupies its own key, so a global and a regional model
    // coexist and both stay selectable with -m. Derived from the provenance,
    // never from user input.
    let config_key = installed
        .provenance
        .as_ref()
        .map_or_else(|| id.to_string(), registry::InstallProvenance::config_key);

    // Re-read the config now rather than reusing the copy loaded before the
    // download. The load above exists only to read the inference device for
    // variant selection, and the download between them can take minutes on a
    // 557 MB model: saving the stale copy would silently discard any `config
    // set` or second install that landed in the meantime. Under the config lock
    // this load-mutate-save is also serialised, so a concurrent writer cannot
    // lose this install either (#313). The download stays outside the lock, so
    // it never holds the lock for those minutes.
    let (orphans, _config, _config_path) = config::update_config(|config| {
        // Collected before the insert overwrites the entry that names them, and
        // deleted only after the config is saved: a crash in between leaves a
        // config that points exclusively at files which exist.
        let mut keeping = vec![model_path.clone(), labels_path.clone()];
        keeping.extend(installed.bsg_calibration.clone());
        keeping.extend(installed.bsg_migration.clone());
        keeping.extend(installed.bsg_distribution_maps.clone());
        let orphans = registry::orphaned_files(config, &config_key, &keeping);

        let provenance = installed.provenance;
        config.models.insert(
            config_key.clone(),
            ModelConfig {
                registry_id: Some(id.to_string()),
                installed_version: provenance.as_ref().map(|p| p.version.clone()),
                installed_build: provenance.as_ref().and_then(|p| p.build),
                region: provenance.as_ref().and_then(|p| p.region.clone()),
                variant: provenance.as_ref().and_then(|p| p.variant.clone()),
                path: installed.model,
                labels: installed.labels,
                model_type,
                meta_model: None,
                bsg_calibration: installed.bsg_calibration,
                bsg_migration: installed.bsg_migration,
                bsg_distribution_maps: installed.bsg_distribution_maps,
            },
        );

        if should_set_default {
            config.defaults.model = Some(config_key.clone());
        }

        Ok(orphans)
    })?;

    // Files the replaced entry owned and nothing else references. Published
    // filenames never change, so an upgrade writes new files beside the old
    // ones; without this every upgrade would leak the previous download.
    for (path, e) in registry::remove_orphans(&orphans) {
        warn!(
            "Could not remove the superseded model file {}: {e}",
            path.display()
        );
    }

    if output_mode.is_structured() {
        let payload = ModelInstalledPayload {
            result_type: ResultType::ModelInstalled,
            id: config_key,
            set_as_default: should_set_default,
            model_path,
            labels_path,
            region: resolved_region,
            variant: resolved_variant,
            selection_reason,
        };
        emit_json_result(&payload);
    } else {
        if should_set_default {
            println!("Model '{config_key}' added to configuration and set as default.");
        } else {
            println!("Model '{config_key}' added to configuration.");
        }
        println!();
        println!("Ready to analyze:");
        if config_key == *id {
            println!("  birda recording.wav");
        } else {
            // A regional install is not the default unless asked for, so name
            // the flag that reaches it.
            println!("  birda -m {config_key} recording.wav");
        }
    }

    Ok(())
}

/// Collect the status of the shared range filter for `birda models check`.
fn check_geomodel() -> Result<output::GeomodelInfo> {
    let registry = registry::load_registry()?;
    let asset = registry
        .range_filter
        .as_ref()
        .ok_or(Error::RangeFilterAssetMissing)?;

    let paths = registry::geomodel_paths(asset)?;
    let installed = paths.is_installed();
    let obsolete_files = registry::models_dir()
        .and_then(|dir| registry::find_obsolete_files(&dir))
        .unwrap_or_default();

    Ok(output::GeomodelInfo {
        version: asset.version.clone(),
        installed,
        species_count: asset.species_count,
        model_path: installed.then(|| paths.model.clone()),
        labels_path: installed.then(|| paths.labels.clone()),
        obsolete_files,
    })
}

/// Print the shared range filter status for `birda models check`.
#[allow(clippy::print_stdout)]
fn report_geomodel_status(info: &output::GeomodelInfo) {
    if info.installed {
        println!(
            "  BirdNET Geomodel v{}: OK ({} species)",
            info.version, info.species_count
        );
    } else {
        println!(
            "  BirdNET Geomodel v{}: not installed \
             (run 'birda models install geomodel' to enable range filtering)",
            info.version
        );
    }

    for path in &info.obsolete_files {
        println!("  {} is no longer used and can be deleted", path.display());
    }
}

/// Handle `birda models install geomodel`.
///
/// Installs the shared `BirdNET` Geomodel v3.0.2 range filter and records its
/// paths in the configuration, so every installed classifier can use it.
fn handle_geomodel_install(
    registry: &registry::Registry,
    interactive: bool,
    assume_yes: bool,
    output_mode: OutputMode,
) -> Result<()> {
    let asset = registry
        .range_filter
        .as_ref()
        .ok_or(Error::RangeFilterAssetMissing)?;

    let licensed = registry::LicensedAsset {
        name: &asset.name,
        vendor: &asset.vendor,
        version: &asset.version,
        license: &asset.license,
    };
    if !registry::prompt_license_acceptance(licensed, interactive, assume_yes)? {
        if !output_mode.is_structured() {
            println!("Installation cancelled.");
        }
        return Ok(());
    }

    let runtime = tokio::runtime::Runtime::new().map_err(|e| Error::Internal {
        message: format!("Failed to create async runtime: {e}"),
    })?;
    let installed = runtime.block_on(registry::install_range_filter(asset))?;

    // Serialised load-mutate-save (#313); the download above is outside the lock.
    config::update_config(|config| {
        config.defaults.geomodel = Some(installed.model.clone());
        config.defaults.geomodel_labels = Some(installed.labels.clone());
        Ok(())
    })?;

    if !output_mode.is_structured() {
        println!();
        println!("{} installed.", asset.name);
        println!("  {}", installed.model.display());
        println!("  {}", installed.labels.display());
        println!();
        println!("Range filtering covers {} species.", asset.species_count);
        println!("Powered by BirdNET (https://birdnet.cornell.edu/)");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_reclaim_stale_lock_removes_an_aged_lock_when_enabled() {
        use std::time::{Duration, SystemTime};
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("audio.wav");
        let lock = crate::locking::FileLock::lock_path_for(&input, dir.path());
        let f = std::fs::File::create(&lock).unwrap();
        // Age it deterministically instead of sleeping.
        f.set_modified(SystemTime::now() - Duration::from_hours(1))
            .unwrap();

        reclaim_stale_lock(&input, dir.path(), Some(Duration::from_mins(1)));

        assert!(
            !crate::locking::FileLock::is_locked(&input, dir.path()),
            "an hour-old lock must be reclaimed against a one-minute timeout"
        );
    }

    #[test]
    fn test_reclaim_stale_lock_keeps_a_fresh_lock() {
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("audio.wav");
        let lock = crate::locking::FileLock::lock_path_for(&input, dir.path());
        std::fs::File::create(&lock).unwrap();

        reclaim_stale_lock(&input, dir.path(), Some(Duration::from_hours(1)));

        assert!(
            crate::locking::FileLock::is_locked(&input, dir.path()),
            "a lock younger than the timeout must be kept"
        );
    }

    #[test]
    fn test_reclaim_stale_lock_is_a_noop_without_a_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("audio.wav");
        let lock = crate::locking::FileLock::lock_path_for(&input, dir.path());
        std::fs::File::create(&lock).unwrap();

        reclaim_stale_lock(&input, dir.path(), None);

        assert!(
            crate::locking::FileLock::is_locked(&input, dir.path()),
            "without --stale-lock-timeout every lock is honoured"
        );
    }

    /// A stand-in audio path for the `record_file_failure` tests; only ever
    /// displayed, never touched on disk.
    const TEST_AUDIO_PATH: &str = "/x/rec.wav";
    /// The lock path paired with [`TEST_AUDIO_PATH`].
    const TEST_LOCK_PATH: &str = "/x/rec.wav.lock";
    /// A stand-in message for a genuine (non-lock) processing error.
    const TEST_DECODE_ERROR: &str = "decode blew up";

    #[test]
    fn test_record_file_failure_counts_a_lost_lock_as_a_skip() {
        // #344: a file locked between should_process's advisory is_locked check
        // and FileLock::acquire must be the graceful "skip locked", not an
        // error, even under --fail-fast. Reverting the FileLocked special-case
        // makes this errors=1 and returns Err, turning both assertions red.
        let mut stats = ProcessingStats::default();
        let reporter = crate::output::NullReporter;
        let err = Error::FileLocked {
            path: PathBuf::from(TEST_LOCK_PATH),
        };

        let outcome = record_file_failure(
            Path::new(TEST_AUDIO_PATH),
            err,
            &mut stats,
            &reporter,
            true, // even under --fail-fast, a lost lock is not fatal
        );

        assert!(outcome.is_ok(), "a lost lock must not abort the run");
        assert_eq!(stats.skipped, 1);
        assert_eq!(stats.errors, 0);
    }

    #[test]
    fn test_record_file_failure_counts_a_real_error_and_continues_without_fail_fast() {
        let mut stats = ProcessingStats::default();
        let reporter = crate::output::NullReporter;
        let err = Error::Internal {
            message: TEST_DECODE_ERROR.to_string(),
        };

        let outcome = record_file_failure(
            Path::new(TEST_AUDIO_PATH),
            err,
            &mut stats,
            &reporter,
            false,
        );

        assert!(outcome.is_ok(), "without --fail-fast the run continues");
        assert_eq!(stats.errors, 1);
        assert_eq!(stats.skipped, 0);
    }

    #[test]
    fn test_record_file_failure_aborts_a_real_error_under_fail_fast() {
        let mut stats = ProcessingStats::default();
        let reporter = crate::output::NullReporter;
        let err = Error::Internal {
            message: TEST_DECODE_ERROR.to_string(),
        };

        let outcome =
            record_file_failure(Path::new(TEST_AUDIO_PATH), err, &mut stats, &reporter, true);

        assert!(
            outcome.is_err(),
            "a real error under --fail-fast must abort"
        );
        assert_eq!(stats.errors, 1);
    }

    /// Records which terminal reporter callbacks fire, so a test can pin the
    /// #344 fix to emitting a `locked` skip rather than a `failed` completion
    /// (the machine-readable event birda-gui consumes). Every other callback is
    /// a no-op, like `NullReporter`. Counters are atomic because the trait is
    /// `Send + Sync`.
    #[derive(Default)]
    struct RecordingReporter {
        skipped_locked: std::sync::atomic::AtomicUsize,
        completed_failure: std::sync::atomic::AtomicUsize,
    }

    impl crate::output::ProgressReporter for RecordingReporter {
        fn pipeline_started(
            &self,
            _total_files: usize,
            _model: &str,
            _min_confidence: f32,
            _execution_provider: &crate::output::ExecutionProviderInfo,
            _range_filter: Option<&crate::output::RangeFilterInfo>,
        ) {
        }
        fn file_started(
            &self,
            _file: &Path,
            _index: usize,
            _estimated_segments: usize,
            _duration_seconds: Option<f64>,
        ) {
        }
        fn progress(
            &self,
            _batch: Option<&crate::output::BatchProgress>,
            _file: Option<&crate::output::FileProgress>,
        ) {
        }
        fn file_completed_success(&self, _file: &Path, _detections: usize, _duration_ms: u64) {}
        fn file_completed_failure(&self, _file: &Path, _error_code: &str, _error_message: &str) {
            self.completed_failure
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        fn file_skipped(&self, _file: &Path, reason: crate::output::FileStatus) {
            if matches!(reason, crate::output::FileStatus::Locked) {
                self.skipped_locked
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        fn pipeline_completed(&self, _summary: &crate::output::PipelineSummary) {}
        fn error(
            &self,
            _code: &str,
            _severity: crate::output::ErrorSeverity,
            _message: &str,
            _suggestion: Option<&str>,
        ) {
        }
        fn cancelled(
            &self,
            _reason: crate::output::CancelReason,
            _files_completed: usize,
            _files_total: usize,
        ) {
        }
        fn detections(
            &self,
            _file: &Path,
            _detections: &[crate::output::Detection],
            _bsg_metadata: Option<&crate::output::BsgMetadata>,
        ) {
        }
    }

    #[test]
    fn test_record_file_failure_emits_a_locked_skip_not_a_failure_event() {
        // #344 on the machine-readable surface: a lost lock must reach the
        // reporter as file_skipped(Locked) (the JSON "locked" event birda-gui
        // consumes), never file_completed_failure ("failed"). The counter tests
        // above use NullReporter and cannot see this: swapping the two reporter
        // calls would keep them green while corrupting the event contract.
        use std::sync::atomic::Ordering;
        let mut stats = ProcessingStats::default();
        let reporter = RecordingReporter::default();
        let err = Error::FileLocked {
            path: PathBuf::from(TEST_LOCK_PATH),
        };

        let outcome =
            record_file_failure(Path::new(TEST_AUDIO_PATH), err, &mut stats, &reporter, true);

        assert!(outcome.is_ok());
        assert_eq!(
            reporter.skipped_locked.load(Ordering::Relaxed),
            1,
            "a lost lock must emit a locked skip event"
        );
        assert_eq!(
            reporter.completed_failure.load(Ordering::Relaxed),
            0,
            "a lost lock must not emit a failure event"
        );
    }

    /// Create a minimal Config with a named model.
    fn config_with_model(name: &str) -> Config {
        let mut models = HashMap::new();
        models.insert(
            name.to_string(),
            ModelConfig {
                registry_id: None,
                installed_version: None,
                installed_build: None,
                region: None,
                variant: None,
                path: PathBuf::from("/path/to/model.onnx"),
                labels: PathBuf::from("/path/to/labels.txt"),
                model_type: ModelType::BirdnetV24,
                meta_model: None,
                bsg_calibration: None,
                bsg_migration: None,
                bsg_distribution_maps: None,
            },
        );
        Config {
            models,
            defaults: config::DefaultsConfig::default(),
            ..Default::default()
        }
    }

    /// Create default `AnalyzeArgs` (all None/false).
    fn default_args() -> AnalyzeArgs {
        AnalyzeArgs::default()
    }

    #[test]
    fn test_priority_1_explicit_named_model() {
        let config = config_with_model("birdnet");
        let mut args = default_args();
        args.model = Some("birdnet".to_string());

        let result = resolve_model_config(&args, &config);
        assert!(result.is_ok());

        let (model_config, name) = result.unwrap();
        assert_eq!(name, "birdnet");
        assert_eq!(model_config.model_type, ModelType::BirdnetV24);
    }

    #[test]
    fn test_priority_1_named_model_with_path_override() {
        let config = config_with_model("birdnet");
        let mut args = default_args();
        args.model = Some("birdnet".to_string());
        args.model_path = Some(PathBuf::from("/custom/path.onnx"));

        let result = resolve_model_config(&args, &config);
        assert!(result.is_ok());

        let (model_config, _) = result.unwrap();
        assert_eq!(model_config.path, PathBuf::from("/custom/path.onnx"));
        // Type should still be from config
        assert_eq!(model_config.model_type, ModelType::BirdnetV24);
    }

    #[test]
    fn test_priority_2_adhoc_model() {
        let config = Config::default();
        let mut args = default_args();
        args.model_type = Some(ModelType::PerchV2);
        args.model_path = Some(PathBuf::from("/adhoc/model.onnx"));
        args.labels_path = Some(PathBuf::from("/adhoc/labels.txt"));

        let result = resolve_model_config(&args, &config);
        assert!(result.is_ok());

        let (model_config, name) = result.unwrap();
        assert_eq!(name, "<ad-hoc>");
        assert_eq!(model_config.model_type, ModelType::PerchV2);
        assert_eq!(model_config.path, PathBuf::from("/adhoc/model.onnx"));
        assert_eq!(model_config.labels, PathBuf::from("/adhoc/labels.txt"));
    }

    #[test]
    fn test_model_type_only_falls_through_to_no_model() {
        // When --model-type is set but no --model-path (and no default),
        // should fall through to Priority 5 (no model specified)
        let config = Config::default();
        let mut args = default_args();
        args.model_type = Some(ModelType::BirdnetV24);
        // Missing model_path - should NOT trigger ad-hoc mode
        args.labels_path = Some(PathBuf::from("/adhoc/labels.txt"));

        let result = resolve_model_config(&args, &config);
        assert!(result.is_err());

        let err = result.unwrap_err();
        // Should be "no model specified", not "--model-path required"
        assert!(err.to_string().contains("no model specified"));
    }

    #[test]
    fn test_model_type_only_falls_through_to_default() {
        // When --model-type is set (e.g., via env var) but no --model-path,
        // should use default model, not error about missing --model-path
        let mut config = config_with_model("birdnet");
        config.defaults.model = Some("birdnet".to_string());

        let mut args = default_args();
        args.model_type = Some(ModelType::PerchV2); // e.g., from BIRDA_MODEL_TYPE env var

        let result = resolve_model_config(&args, &config);
        assert!(result.is_ok());

        let (model_config, name) = result.unwrap();
        // Should fall through to default, not ad-hoc
        assert_eq!(name, "birdnet");
        // Type should be from config, NOT from args.model_type
        assert_eq!(model_config.model_type, ModelType::BirdnetV24);
    }

    #[test]
    fn test_priority_2_adhoc_missing_labels_path() {
        let config = Config::default();
        let mut args = default_args();
        args.model_type = Some(ModelType::BirdnetV24);
        args.model_path = Some(PathBuf::from("/adhoc/model.onnx"));
        // Missing labels_path

        let result = resolve_model_config(&args, &config);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err.to_string().contains("--labels-path required"));
    }

    #[test]
    fn test_priority_2_adhoc_ignores_defaults_model() {
        // Even if defaults.model is set, --model-type should trigger ad-hoc mode
        let mut config = config_with_model("birdnet");
        config.defaults.model = Some("birdnet".to_string());

        let mut args = default_args();
        args.model_type = Some(ModelType::PerchV2);
        args.model_path = Some(PathBuf::from("/adhoc/model.onnx"));
        args.labels_path = Some(PathBuf::from("/adhoc/labels.txt"));

        let result = resolve_model_config(&args, &config);
        assert!(result.is_ok());

        let (model_config, name) = result.unwrap();
        // Should be ad-hoc, not the default
        assert_eq!(name, "<ad-hoc>");
        assert_eq!(model_config.model_type, ModelType::PerchV2);
    }

    #[test]
    fn test_priority_3_implicit_default_model() {
        let mut config = config_with_model("birdnet");
        config.defaults.model = Some("birdnet".to_string());

        let args = default_args();
        // No -m, no --model-type

        let result = resolve_model_config(&args, &config);
        assert!(result.is_ok());

        let (model_config, name) = result.unwrap();
        assert_eq!(name, "birdnet");
        assert_eq!(model_config.model_type, ModelType::BirdnetV24);
    }

    #[test]
    fn test_priority_3_default_model_with_path_override() {
        let mut config = config_with_model("birdnet");
        config.defaults.model = Some("birdnet".to_string());

        let mut args = default_args();
        args.model_path = Some(PathBuf::from("/custom/model.onnx"));
        // No -m, no --model-type (so uses default and patches path)

        let result = resolve_model_config(&args, &config);
        assert!(result.is_ok());

        let (model_config, name) = result.unwrap();
        assert_eq!(name, "birdnet");
        assert_eq!(model_config.path, PathBuf::from("/custom/model.onnx"));
        // Type unchanged
        assert_eq!(model_config.model_type, ModelType::BirdnetV24);
    }

    #[test]
    fn test_priority_4_incomplete_adhoc() {
        let config = Config::default();
        let mut args = default_args();
        args.model_path = Some(PathBuf::from("/some/model.onnx"));
        // Missing --model-type

        let result = resolve_model_config(&args, &config);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err.to_string().contains("--model-type"));
    }

    #[test]
    fn test_priority_5_no_model() {
        let config = Config::default();
        let args = default_args();

        let result = resolve_model_config(&args, &config);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err.to_string().contains("no model specified"));
    }

    #[test]
    fn test_adhoc_ignores_the_deprecated_meta_model_flag() {
        // --meta-model-path is retained as a hidden flag so scripts do not
        // break, but it no longer configures anything.
        let args = AnalyzeArgs {
            model_path: Some(PathBuf::from("/adhoc/model.onnx")),
            labels_path: Some(PathBuf::from("/adhoc/labels.txt")),
            model_type: Some(ModelType::BirdnetV24),
            meta_model_path: Some(PathBuf::from("/adhoc/meta.onnx")),
            ..Default::default()
        };

        let config = Config::default();
        let (model_config, name) = resolve_model_config(&args, &config).unwrap();

        assert_eq!(name, ADHOC_MODEL_NAME);
        assert!(
            model_config.meta_model.is_none(),
            "the deprecated flag must not populate the config"
        );
    }

    // Tests for validate_model_files

    #[test]
    fn test_validate_model_files_all_exist() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.onnx");
        let labels_path = dir.path().join("labels.txt");
        std::fs::write(&model_path, "model").unwrap();
        std::fs::write(&labels_path, "labels").unwrap();

        let config = ModelConfig {
            registry_id: None,
            installed_version: None,
            installed_build: None,
            region: None,
            variant: None,
            path: model_path,
            labels: labels_path,
            model_type: ModelType::BirdnetV24,
            meta_model: None,
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
        };

        assert!(validate_model_files(&config).is_ok());
    }

    #[test]
    fn test_validate_model_files_missing_model() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("missing.onnx");
        let labels_path = dir.path().join("labels.txt");
        std::fs::write(&labels_path, "labels").unwrap();

        let config = ModelConfig {
            registry_id: None,
            installed_version: None,
            installed_build: None,
            region: None,
            variant: None,
            path: model_path,
            labels: labels_path,
            model_type: ModelType::BirdnetV24,
            meta_model: None,
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
        };

        let err = validate_model_files(&config).unwrap_err();
        assert!(matches!(err, Error::ModelFileNotFound { .. }));
    }

    #[test]
    fn test_validate_model_files_missing_labels() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.onnx");
        let labels_path = dir.path().join("missing.txt");
        std::fs::write(&model_path, "model").unwrap();

        let config = ModelConfig {
            registry_id: None,
            installed_version: None,
            installed_build: None,
            region: None,
            variant: None,
            path: model_path,
            labels: labels_path,
            model_type: ModelType::BirdnetV24,
            meta_model: None,
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
        };

        let err = validate_model_files(&config).unwrap_err();
        assert!(matches!(err, Error::LabelsFileNotFound { .. }));
    }

    #[test]
    fn test_validate_model_files_ignores_a_deprecated_meta_model() {
        // meta_model is parsed only so a stale key can be reported. It must
        // never gate validation: the file it points at is expected to be gone.
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.onnx");
        let labels_path = dir.path().join("labels.txt");
        std::fs::write(&model_path, "model").unwrap();
        std::fs::write(&labels_path, "labels").unwrap();

        let config = ModelConfig {
            registry_id: None,
            installed_version: None,
            installed_build: None,
            region: None,
            variant: None,
            path: model_path,
            labels: labels_path,
            model_type: ModelType::BirdnetV24,
            meta_model: Some(dir.path().join("long_gone_meta.onnx")),
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
        };

        assert!(validate_model_files(&config).is_ok());
    }

    // Tests for resolve_device

    #[test]
    fn test_resolve_device_defaults_to_config() {
        let args = default_args();
        let mut config = Config::default();
        config.inference.device = InferenceDevice::Cuda;

        let device = resolve_device(&args, &config);
        assert_eq!(device, InferenceDevice::Cuda);
    }

    #[test]
    fn test_resolve_device_cli_flag_overrides_config() {
        let mut args = default_args();
        args.cpu = true;

        let mut config = Config::default();
        config.inference.device = InferenceDevice::Cuda;

        let device = resolve_device(&args, &config);
        assert_eq!(device, InferenceDevice::Cpu);
    }

    #[test]
    fn test_resolve_device_first_flag_wins() {
        let mut args = default_args();
        args.cpu = true;
        args.cuda = true; // Both set, but cpu comes first in the array

        let config = Config::default();

        let device = resolve_device(&args, &config);
        // cpu is checked before cuda in the array (order: gpu, cpu, cuda, ...)
        assert_eq!(device, InferenceDevice::Cpu);
    }

    #[test]
    fn test_resolve_device_tensorrt_flag() {
        let mut args = default_args();
        args.tensorrt = true;

        let config = Config::default();

        let device = resolve_device(&args, &config);
        assert_eq!(device, InferenceDevice::TensorRt);
    }

    // Tests for resolve_species_filter

    #[test]
    fn test_resolve_species_filter_returns_none_when_range_filter_active() {
        let args = default_args();
        let config = Config::default();

        // When range filter is active, species list should be None
        let result = resolve_species_filter(&args, &config, true).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_species_filter_returns_none_when_no_filter() {
        let args = default_args();
        let config = Config::default();

        // No range filter, no species list file
        let result = resolve_species_filter(&args, &config, false).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_species_filter_loads_from_args() {
        let dir = tempfile::tempdir().unwrap();
        let slist_path = dir.path().join("species.txt");
        std::fs::write(&slist_path, "Species One\nSpecies Two\nSpecies Three\n").unwrap();

        let mut args = default_args();
        args.slist = Some(slist_path);

        let config = Config::default();

        let result = resolve_species_filter(&args, &config, false).unwrap();
        assert!(result.is_some());

        let species = result.unwrap();
        assert_eq!(species.len(), 3);
        assert!(species.contains("Species One"));
        assert!(species.contains("Species Two"));
        assert!(species.contains("Species Three"));
    }

    #[test]
    fn test_resolve_species_filter_loads_from_config() {
        let dir = tempfile::tempdir().unwrap();
        let slist_path = dir.path().join("species.txt");
        std::fs::write(&slist_path, "Config Species\n").unwrap();

        let args = default_args();
        let mut config = Config::default();
        config.defaults.species_list_file = Some(slist_path);

        let result = resolve_species_filter(&args, &config, false).unwrap();
        assert!(result.is_some());

        let species = result.unwrap();
        assert_eq!(species.len(), 1);
        assert!(species.contains("Config Species"));
    }

    #[test]
    fn test_resolve_species_filter_args_takes_precedence_over_config() {
        let dir = tempfile::tempdir().unwrap();
        let args_slist = dir.path().join("args_species.txt");
        let config_slist = dir.path().join("config_species.txt");
        std::fs::write(&args_slist, "Args Species\n").unwrap();
        std::fs::write(&config_slist, "Config Species\n").unwrap();

        let mut args = default_args();
        args.slist = Some(args_slist);

        let mut config = Config::default();
        config.defaults.species_list_file = Some(config_slist);

        let result = resolve_species_filter(&args, &config, false).unwrap();
        let species = result.unwrap();

        // Args should take precedence
        assert!(species.contains("Args Species"));
        assert!(!species.contains("Config Species"));
    }

    #[test]
    fn test_batch_size_cpu() {
        let ep_status = inference::ExecutionProviderStatus {
            requested: "cpu".to_string(),
            actual: "CPU".to_string(),
            fallback_reason: None,
        };

        assert_eq!(
            determine_default_batch_size(ModelType::BirdnetV24, &ep_status),
            8
        );
        assert_eq!(
            determine_default_batch_size(ModelType::BirdnetV30, &ep_status),
            8
        );
        assert_eq!(
            determine_default_batch_size(ModelType::PerchV2, &ep_status),
            8
        );
        assert_eq!(
            determine_default_batch_size(ModelType::BsgFinland, &ep_status),
            8
        );
    }

    #[test]
    fn test_batch_size_cuda() {
        let ep_status = inference::ExecutionProviderStatus {
            requested: "cuda".to_string(),
            actual: "CUDA".to_string(),
            fallback_reason: None,
        };

        assert_eq!(
            determine_default_batch_size(ModelType::BirdnetV24, &ep_status),
            64
        );
        assert_eq!(
            determine_default_batch_size(ModelType::BsgFinland, &ep_status),
            64
        );
        assert_eq!(
            determine_default_batch_size(ModelType::BirdnetV30, &ep_status),
            32
        );
        assert_eq!(
            determine_default_batch_size(ModelType::PerchV2, &ep_status),
            32
        );
    }

    #[test]
    fn test_batch_size_tensorrt() {
        let ep_status = inference::ExecutionProviderStatus {
            requested: "tensorrt".to_string(),
            actual: "TensorRT".to_string(),
            fallback_reason: None,
        };

        assert_eq!(
            determine_default_batch_size(ModelType::BirdnetV24, &ep_status),
            32
        );
        assert_eq!(
            determine_default_batch_size(ModelType::BsgFinland, &ep_status),
            32
        );
        assert_eq!(
            determine_default_batch_size(ModelType::BirdnetV30, &ep_status),
            32
        );
        assert_eq!(
            determine_default_batch_size(ModelType::PerchV2, &ep_status),
            32
        );
    }

    #[test]
    fn test_batch_size_other_gpu() {
        let ep_status = inference::ExecutionProviderStatus {
            requested: "directml".to_string(),
            actual: "DirectML".to_string(),
            fallback_reason: None,
        };

        // Should return conservative default of 16
        assert_eq!(
            determine_default_batch_size(ModelType::BirdnetV24, &ep_status),
            16
        );
    }

    #[test]
    fn test_batch_size_cpu_fallback() {
        let ep_status = inference::ExecutionProviderStatus {
            requested: "gpu".to_string(),
            actual: "CPU".to_string(),
            fallback_reason: Some("No GPU providers available".to_string()),
        };

        // Should use CPU batch size (8) when falling back to CPU
        assert_eq!(
            determine_default_batch_size(ModelType::BirdnetV24, &ep_status),
            8
        );
    }

    #[test]
    fn test_batch_size_unknown_provider() {
        let ep_status = inference::ExecutionProviderStatus {
            requested: "future-provider".to_string(),
            actual: "FutureGPU".to_string(),
            fallback_reason: None,
        };

        // Should use conservative default of 16 for unknown providers
        assert_eq!(
            determine_default_batch_size(ModelType::BirdnetV24, &ep_status),
            16
        );
        assert_eq!(
            determine_default_batch_size(ModelType::BirdnetV30, &ep_status),
            16
        );
    }

    // ── models remove tests ──────────────────────────────────────

    #[test]
    fn test_remove_default_model_promotes_next_alphabetically() {
        let mut config = config_with_model("beta");
        config
            .models
            .insert("alpha".to_string(), config.models["beta"].clone());
        config.defaults.model = Some("beta".to_string());

        let (_, promoted) = remove_model_from_config(&mut config, "beta").unwrap();

        assert_eq!(promoted.as_deref(), Some("alpha"));
        assert_eq!(config.defaults.model.as_deref(), Some("alpha"));
        assert!(!config.models.contains_key("beta"));
    }

    #[test]
    fn test_remove_last_model_clears_default() {
        let mut config = config_with_model("only");
        config.defaults.model = Some("only".to_string());

        let (_, promoted) = remove_model_from_config(&mut config, "only").unwrap();

        assert!(promoted.is_none());
        assert!(config.defaults.model.is_none());
        assert!(config.models.is_empty());
    }

    #[test]
    fn test_remove_non_default_model_preserves_default() {
        let mut config = config_with_model("primary");
        config
            .models
            .insert("secondary".to_string(), config.models["primary"].clone());
        config.defaults.model = Some("primary".to_string());

        let (_, promoted) = remove_model_from_config(&mut config, "secondary").unwrap();

        assert!(promoted.is_none());
        assert_eq!(config.defaults.model.as_deref(), Some("primary"));
        assert!(config.models.contains_key("primary"));
        assert!(!config.models.contains_key("secondary"));
    }

    #[test]
    fn test_remove_nonexistent_model_returns_error() {
        let mut config = Config::default();
        let result = remove_model_from_config(&mut config, "nonexistent");

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("nonexistent"));
    }

    #[test]
    fn test_remove_returns_model_config() {
        let mut config = config_with_model("birdnet");

        let (model, _) = remove_model_from_config(&mut config, "birdnet").unwrap();

        assert_eq!(model.path, PathBuf::from("/path/to/model.onnx"));
        assert_eq!(model.labels, PathBuf::from("/path/to/labels.txt"));
        assert_eq!(model.model_type, ModelType::BirdnetV24);
    }

    /// Which commands run against a validated config (#295).
    ///
    /// The predicate decides whether a hand-edited `config.toml` can reach a
    /// command at all, so both answers need pinning. An exemption that shrinks
    /// locks a user out of the commands that repair the file; one that spreads
    /// starts rejecting runs over values they never read.
    // A test command line that fails to parse is a broken test, not a runtime
    // condition to handle, so `expect` is the right call here. Matches the
    // convention the other test modules in this crate use.
    mod requires_valid_config {
        use super::*;

        /// Answer the predicate for a real command line.
        ///
        /// Parsing through clap rather than building `Command` variants by hand
        /// keeps the test honest about two things a hand-built variant hides:
        /// which arguments are actually required on a subcommand, and that
        /// `inputs` is empty whenever a subcommand is present, since clap routes
        /// positionals to the subcommand.
        fn gated(argv: &[&str]) -> bool {
            let cli = Cli::try_parse_from(argv).expect("test command line should parse");
            command_requires_valid_config(cli.command.as_ref(), cli.inputs.is_empty())
        }

        #[test]
        fn test_analysis_requires_a_valid_config() {
            // The reported case, and the only gated path. `range_threshold =
            // nan` dropped every mapped species here and the only symptom was
            // an info log reading "0 species in range".
            assert!(gated(&["birda", "recording.wav"]));
        }

        #[test]
        fn test_bare_invocation_without_inputs_is_exempt() {
            // Falls through to `print_smart_help`. Refusing to print help
            // because the config is wrong is the least useful moment to refuse.
            assert!(!gated(&["birda"]));
        }

        #[test]
        fn test_config_subcommand_is_exempt() {
            // The repair surface. `config show` diagnoses the bad value and
            // `config set` replaces it, so validating here would mean the only
            // way out of a broken file is the hand-editing that broke it.
            for argv in [
                vec!["birda", "config", "show"],
                vec!["birda", "config", "path"],
                vec!["birda", "config", "init"],
                vec!["birda", "config", "set", "defaults.latitude", "60.17"],
            ] {
                assert!(
                    !gated(&argv),
                    "{argv:?} must stay reachable with an invalid config"
                );
            }
        }

        #[test]
        fn test_models_subcommand_is_exempt() {
            // `defaults.model` naming a model that is not in the file is the
            // one fault `config set` cannot fix by setting the offending key,
            // and `models install` is its natural repair. Gating this arm made
            // that repair unreachable.
            for argv in [
                vec!["birda", "models", "list"],
                vec!["birda", "models", "check"],
                vec!["birda", "models", "install", "birdnet-v24"],
                vec!["birda", "models", "remove", "birdnet-v24"],
            ] {
                assert!(
                    !gated(&argv),
                    "{argv:?} must stay reachable so a dangling default can be repaired"
                );
            }
        }

        #[test]
        fn test_commands_that_receive_no_config_are_exempt() {
            // `handle_providers_command`, `clipper::command::execute` and
            // `handle_update_command` each take only an output mode, so an
            // invalid value cannot reach them. `update` matters most: it is how
            // a user installs a build that fixes the config handling, and
            // gating it on the config would block that route.
            for argv in [
                vec!["birda", "providers"],
                vec!["birda", "clip", "results.csv"],
                vec!["birda", "update", "--check"],
            ] {
                assert!(
                    !gated(&argv),
                    "{argv:?} receives no Config and must not be gated on one"
                );
            }
        }

        #[test]
        fn test_species_is_exempt_because_its_coordinates_are_required_args() {
            // `--lat` and `--lon` are required here, so `species` never reads
            // `defaults.latitude` or `defaults.longitude`. Failing it on a stale
            // coordinate would reject a run that could not have used it.
            assert!(!gated(&[
                "birda", "species", "--lat", "60.17", "--lon", "24.94", "--week", "24"
            ]));
        }
    }
}
