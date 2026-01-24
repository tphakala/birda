//! Birda - Bird species detection CLI tool.
//!
//! This crate provides audio analysis capabilities using `BirdNET` and Perch models.

#![warn(missing_docs)]

pub mod audio;
pub mod cli;
pub mod clipper;
pub mod config;
pub mod constants;
pub mod error;
pub mod gpu;
pub mod inference;
pub mod locking;
pub mod output;
pub mod pipeline;
pub mod registry;
pub mod utils;

use clap::Parser;
use cli::{AnalyzeArgs, Cli, Command};
use config::{
    Config, InferenceDevice, ModelConfig, ModelType, OutputMode, config_file_path,
    load_default_config, range_filter::build_range_filter_config, save_default_config,
};
use constants::DEFAULT_TOP_K;
use inference::BirdClassifier;
use output::{
    ConfigPayload, FileStatus, ModelDetails, ModelEntry, ModelInfoPayload, ModelListPayload,
    PipelineSummary, ProgressReporter, ProviderInfo, ProvidersPayload, ResultType, create_reporter,
    emit_json_result,
};
use pipeline::{ProcessCheck, collect_input_files, output_dir_for, process_file, should_process};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{error, info, warn};

pub use error::{Error, Result};

/// Model name used for ad-hoc (CLI-specified) models.
const ADHOC_MODEL_NAME: &str = "<ad-hoc>";

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
            path: path.clone(),
            labels,
            model_type,
            meta_model: args.meta_model_path.clone(),
        };

        return Ok((model_config, ADHOC_MODEL_NAME.to_string()));
    }

    // Priority 3: Implicit default model from config
    if let Some(ref name) = config.defaults.model {
        let mut model_config = config::get_model(config, name)?.clone();

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

/// Apply CLI overrides to a model configuration.
fn apply_model_overrides(model_config: &mut ModelConfig, args: &AnalyzeArgs) {
    if let Some(ref path) = args.model_path {
        model_config.path.clone_from(path);
    }
    if let Some(ref labels) = args.labels_path {
        model_config.labels.clone_from(labels);
    }
    if let Some(ref meta) = args.meta_model_path {
        model_config.meta_model = Some(meta.clone());
    }
}

/// Main entry point for birda CLI.
pub fn run() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    init_logging(cli.analyze.verbose, cli.analyze.quiet);

    // Install Ctrl+C handler to clean up lock files on interrupt
    if let Err(e) = ctrlc::set_handler(|| {
        locking::cleanup_all_locks();
        std::process::exit(130); // 128 + SIGINT(2)
    }) {
        warn!("Failed to install Ctrl+C handler: {e}");
    }

    // Initialize ONNX Runtime (for load-dynamic builds)
    birdnet_onnx::init_runtime().map_err(|e| Error::RuntimeInitialization {
        reason: e.to_string(),
    })?;

    // Load configuration
    let config = load_default_config()?;

    // Determine output mode (CLI flag takes precedence over config)
    let output_mode = cli.output_mode.unwrap_or(config.output.default_format);

    // Create reporter based on output mode
    let reporter: Arc<dyn ProgressReporter> = Arc::from(create_reporter(output_mode));

    // Handle subcommands
    if let Some(command) = cli.command {
        return handle_command(command, &config, output_mode, &reporter);
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

/// Analyze input files with the given options.
fn analyze_files(
    inputs: &[PathBuf],
    args: &AnalyzeArgs,
    config: &Config,
    output_mode: OutputMode,
    reporter: &Arc<dyn ProgressReporter>,
) -> Result<()> {
    use crate::output::progress;
    use std::time::Instant;

    let total_start = Instant::now();

    // Collect all input files
    let files = collect_input_files(inputs)?;
    if files.is_empty() {
        return Err(Error::NoValidAudioFiles);
    }

    info!("Found {} audio file(s) to process", files.len());

    // Resolve model configuration using priority-based resolution
    let (model_config, model_name) = resolve_model_config(args, config)?;

    // Validate model files exist
    if !model_config.path.exists() {
        return Err(Error::ModelFileNotFound {
            path: model_config.path,
        });
    }
    if !model_config.labels.exists() {
        return Err(Error::LabelsFileNotFound {
            path: model_config.labels,
        });
    }
    if let Some(ref meta) = model_config.meta_model
        && !meta.exists()
    {
        return Err(Error::MetaModelNotFound { path: meta.clone() });
    }

    // Resolve other settings
    let min_confidence = args
        .min_confidence
        .unwrap_or(config.defaults.min_confidence);

    // Report pipeline start
    reporter.pipeline_started(files.len(), &model_name, min_confidence);
    let overlap = args.overlap.unwrap_or(config.defaults.overlap);
    let batch_size = args.batch_size.unwrap_or(config.defaults.batch_size);
    let formats = args
        .format
        .clone()
        .unwrap_or_else(|| config.defaults.formats.clone());
    let output_dir = args.output_dir.clone();
    let force = args.force;
    let fail_fast = args.fail_fast;

    // Resolve device from command-line flags or config
    let device = [
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
    ]
    .into_iter()
    .find(|(flag, _)| *flag)
    .map_or(config.inference.device, |(_, device)| device);

    // Build range filter config
    let range_filter_config = build_range_filter_config(args, config, &model_config, &model_name)?;

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

    // Priority: lat/lon (dynamic) > species list file (static) > no filtering
    let species_list = if range_filter_config.is_some() {
        // Dynamic filtering - species list will come from range filter
        None
    } else if let Some(slist_path) = args
        .slist
        .as_ref()
        .or(config.defaults.species_list_file.as_ref())
    {
        // Static filtering from file
        info!("Loading species list: {}", slist_path.display());
        Some(
            utils::species_list::read_species_list(slist_path)?
                .into_iter()
                .collect::<HashSet<_>>(),
        )
    } else {
        // No filtering
        None
    };

    // Log if species list filtering is enabled
    if let Some(ref species) = species_list {
        info!(
            "Species list filter enabled: {} species loaded",
            species.len()
        );
    }

    // Extract range filter params before moving range_filter_config
    #[allow(clippy::cast_possible_truncation)]
    let range_filter_params = range_filter_config.as_ref().map(|rf| {
        let week = crate::utils::date::date_to_week(rf.month, rf.day) as u8;
        (rf.latitude, rf.longitude, week)
    });

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

    // Warm up the classifier to trigger any deferred initialization.
    // TensorRT compiles/loads its engine during the first inference, which can
    // take several minutes on first run. We do this before starting the processing
    // loop so the inference watchdog doesn't kill the process during engine build.
    //
    // TensorRT builds separate engines for each batch size, so we must warm up
    // with the actual batch size that will be used for inference.
    if classifier.uses_tensorrt() {
        use indicatif::{ProgressBar, ProgressStyle};
        use std::time::{Duration, Instant};

        /// Threshold in seconds to distinguish engine build from cache load.
        /// Warmup taking >= this long indicates `TensorRT` compiled a new engine.
        const WARMUP_BUILD_THRESHOLD_SECS: u64 = 2;

        // Create a spinner to show activity during warmup
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
        let result = classifier.warmup(batch_size);
        let warmup_duration = warmup_start.elapsed();

        spinner.finish_and_clear();

        // Propagate any warmup error
        result?;

        if warmup_duration.as_secs() >= WARMUP_BUILD_THRESHOLD_SECS {
            // Engine was built - this was a slow initialization
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
    } else {
        classifier.warmup(batch_size)?;
    }

    // Create file progress bar (disabled in JSON mode - reporter handles progress)
    let is_json_output = matches!(output_mode, OutputMode::Json | OutputMode::Ndjson);
    let progress_enabled = !args.quiet && !args.no_progress && !is_json_output;
    let file_progress = progress::create_file_progress(files.len(), progress_enabled);

    // Process files
    let mut processed = 0;
    let mut skipped = 0;
    let mut errors = 0;
    let mut total_detections = 0;
    let mut total_segments = 0;
    let mut total_audio_duration = 0.0f64;

    for (index, file) in files.iter().enumerate() {
        let file_output_dir = output_dir_for(file, output_dir.as_deref());

        // Check if should process
        match should_process(file, &file_output_dir, &formats, force) {
            ProcessCheck::SkipExists => {
                info!("Skipping (output exists): {}", file.display());
                reporter.file_skipped(file, FileStatus::Skipped);
                skipped += 1;
                progress::inc_progress(file_progress.as_ref());
                continue;
            }
            ProcessCheck::SkipLocked => {
                info!("Skipping (locked): {}", file.display());
                reporter.file_skipped(file, FileStatus::Locked);
                skipped += 1;
                progress::inc_progress(file_progress.as_ref());
                continue;
            }
            ProcessCheck::Process => {}
        }

        // Estimate segments for reporter (using overlap from config)
        let segment_duration = classifier.segment_duration();
        #[allow(clippy::cast_possible_truncation)]
        let estimated_segments =
            progress::estimate_segment_count(None, segment_duration, overlap).unwrap_or(0) as usize;

        // Report file start
        reporter.file_started(file, index, estimated_segments, None);

        // Process the file
        let file_start = std::time::Instant::now();
        match process_file(
            file,
            &file_output_dir,
            &classifier,
            &formats,
            min_confidence,
            overlap,
            batch_size,
            &config.defaults.csv_columns.include,
            progress_enabled,
            !args.no_csv_bom,
            &model_name,
            range_filter_params,
        ) {
            Ok(result) => {
                #[allow(clippy::cast_possible_truncation)]
                let duration_ms = file_start.elapsed().as_millis() as u64;
                reporter.file_completed_success(file, result.detections, duration_ms);
                processed += 1;
                total_detections += result.detections;
                total_segments += result.segments;
                total_audio_duration += result.audio_duration_secs;
            }
            Err(e) => {
                error!("Failed to process {}: {}", file.display(), e);
                reporter.file_completed_failure(file, "processing_error", &e.to_string());
                errors += 1;
                if fail_fast {
                    progress::finish_progress(file_progress, "Failed");
                    // Report pipeline failure
                    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                    let duration_ms = total_start.elapsed().as_millis() as u64;
                    #[allow(clippy::cast_precision_loss)]
                    let realtime_factor = if duration_ms > 0 {
                        total_audio_duration / (duration_ms as f64 / 1000.0)
                    } else {
                        0.0
                    };
                    reporter.pipeline_completed(&PipelineSummary {
                        files_processed: processed,
                        files_failed: errors,
                        files_skipped: skipped,
                        total_detections,
                        total_segments,
                        duration_ms,
                        realtime_factor,
                    });
                    return Err(e);
                }
            }
        }
        progress::inc_progress(file_progress.as_ref());
    }

    progress::finish_progress(file_progress, "Complete");

    // Summary
    let total_duration = total_start.elapsed().as_secs_f64();
    #[allow(clippy::cast_possible_truncation)]
    let duration_ms = total_start.elapsed().as_millis() as u64;
    info!(
        "Complete: {} processed, {} skipped, {} errors, {} total detections in {:.2}s",
        processed, skipped, errors, total_detections, total_duration
    );

    #[allow(clippy::cast_precision_loss)]
    let overall_realtime_factor = if total_duration > 0.0 {
        total_audio_duration / total_duration
    } else {
        0.0
    };

    if processed > 0 {
        #[allow(clippy::cast_precision_loss)]
        let avg_segments_per_sec = if total_duration > 0.0 {
            total_segments as f64 / total_duration
        } else {
            0.0
        };
        info!(
            "Performance: {:.1} segments/sec overall, {:.1}x realtime ({} total audio)",
            avg_segments_per_sec,
            overall_realtime_factor,
            progress::format_duration(total_audio_duration)
        );
    }

    if errors > 0 && !fail_fast {
        warn!("{} file(s) had errors", errors);
    }

    // Report pipeline completion
    reporter.pipeline_completed(&PipelineSummary {
        files_processed: processed,
        files_failed: errors,
        files_skipped: skipped,
        total_detections,
        total_segments,
        duration_ms,
        realtime_factor: overall_realtime_factor,
    });

    Ok(())
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
    fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}

fn handle_command(
    command: Command,
    config: &config::Config,
    output_mode: OutputMode,
    _reporter: &Arc<dyn ProgressReporter>,
) -> Result<()> {
    match command {
        Command::Config { action } => handle_config_command(action, output_mode),
        Command::Models { action } => handle_models_command(action, config, output_mode),
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
        ),
        Command::Clip(args) => clipper::command::execute(&args, output_mode),
    }
}

fn handle_providers_command(output_mode: OutputMode) {
    use birdnet_onnx::available_execution_providers;

    let providers = available_execution_providers();

    // Build provider info list
    let provider_infos: Vec<ProviderInfo> = providers
        .iter()
        .map(|provider| {
            let (id, name, description) = match provider {
                birdnet_onnx::ExecutionProviderInfo::Cpu => {
                    ("cpu", "CPU", "CPU (always available)")
                }
                birdnet_onnx::ExecutionProviderInfo::Cuda => {
                    ("cuda", "CUDA", "CUDA (NVIDIA GPU acceleration)")
                }
                birdnet_onnx::ExecutionProviderInfo::TensorRt => (
                    "tensorrt",
                    "TensorRT",
                    "TensorRT (NVIDIA optimized inference)",
                ),
                birdnet_onnx::ExecutionProviderInfo::DirectMl => (
                    "directml",
                    "DirectML",
                    "DirectML (Windows GPU acceleration)",
                ),
                birdnet_onnx::ExecutionProviderInfo::CoreMl => {
                    ("coreml", "CoreML", "CoreML (Apple GPU/Neural Engine)")
                }
                birdnet_onnx::ExecutionProviderInfo::Rocm => {
                    ("rocm", "ROCm", "ROCm (AMD GPU acceleration)")
                }
                birdnet_onnx::ExecutionProviderInfo::OpenVino => {
                    ("openvino", "OpenVINO", "OpenVINO (Intel optimization)")
                }
                birdnet_onnx::ExecutionProviderInfo::OneDnn => {
                    ("onednn", "oneDNN", "oneDNN (Intel CPU optimization)")
                }
                birdnet_onnx::ExecutionProviderInfo::Qnn => {
                    ("qnn", "QNN", "QNN (Qualcomm Neural Network)")
                }
                birdnet_onnx::ExecutionProviderInfo::Acl => {
                    ("acl", "ACL", "ACL (Arm Compute Library)")
                }
                birdnet_onnx::ExecutionProviderInfo::ArmNn => {
                    ("armnn", "ArmNN", "ArmNN (Arm Neural Network)")
                }
            };
            ProviderInfo {
                id: id.to_string(),
                name: name.to_string(),
                description: description.to_string(),
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
    println!("  --gpu          Auto-select best GPU (TensorRT → CUDA → DirectML → ...)");
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
    ];
    for (flag, description) in explicit_providers {
        println!("  --{flag:<13} {description}");
    }
    println!();
    println!("Note: This shows compile-time availability. Runtime availability may");
    println!("      differ based on drivers and hardware. Check log output for actual");
    println!("      provider selection during inference.");
}

fn handle_config_command(action: cli::ConfigAction, output_mode: OutputMode) -> Result<()> {
    use cli::ConfigAction;

    match action {
        ConfigAction::Init => {
            let path = config_file_path()?;
            if path.exists() {
                println!("Configuration file already exists: {}", path.display());
                println!("Use 'birda models add' to add models.");
            } else {
                let config = Config::default();
                let saved_path = save_default_config(&config)?;
                println!("Created configuration file: {}", saved_path.display());
                println!("\nNext steps:");
                println!(
                    "  birda models add <name> --path <model.onnx> --labels <labels.txt> --type <type> --default"
                );
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
        ConfigAction::Path => {
            let path = config_file_path()?;
            println!("{}", path.display());
            Ok(())
        }
    }
}

fn handle_models_command(
    action: cli::ModelsAction,
    config: &config::Config,
    output_mode: OutputMode,
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
                        has_meta_model: model.meta_model.is_some(),
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
            registry::list_available(&registry);
            Ok(())
        }
        ModelsAction::Add {
            name,
            path,
            labels,
            r#type,
            default,
        } => handle_models_add(name, path, labels, r#type, default),
        ModelsAction::Check => {
            for (name, model) in &config.models {
                config::validate_model_config(name, model)?;
                println!("  {name}: OK");
            }
            Ok(())
        }
        ModelsAction::Info { id, languages } => {
            // Try registry first
            let registry = registry::load_registry()?;
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
                            meta_model_path: None,
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
                            meta_model_path: model.meta_model.clone(),
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
        ModelsAction::Install {
            id,
            language,
            default,
        } => handle_models_install(&id, language.as_deref(), default),
    }
}

/// Handle the `models add` command.
fn handle_models_add(
    name: String,
    path: PathBuf,
    labels: PathBuf,
    model_type: ModelType,
    set_default: bool,
) -> Result<()> {
    // Validate files exist
    if !path.exists() {
        return Err(Error::ModelFileNotFound { path });
    }
    if !labels.exists() {
        return Err(Error::LabelsFileNotFound { path: labels });
    }

    // Load existing config
    let mut config = load_default_config()?;

    // Check if model already exists
    if config.models.contains_key(&name) {
        return Err(Error::ModelAlreadyExists { name });
    }

    // Add the model
    config.models.insert(
        name.clone(),
        ModelConfig {
            path: path.clone(),
            labels: labels.clone(),
            model_type,
            meta_model: None,
        },
    );

    // Set as default if requested
    if set_default {
        config.defaults.model = Some(name.clone());
    }

    // Save config
    let config_path = save_default_config(&config)?;

    // Print success message
    println!("Added model '{name}' ({model_type})");
    println!("  Model: {}", path.display());
    println!("  Labels: {}", labels.display());
    println!("  Default: {}", if set_default { "yes" } else { "no" });
    println!("\nConfiguration saved to: {}", config_path.display());

    Ok(())
}

/// Handle the `models install` command.
fn handle_models_install(id: &str, language: Option<&str>, set_default: bool) -> Result<()> {
    use std::io::Write;

    // Load registry
    let registry = registry::load_registry()?;
    let model = registry::find_model(&registry, id)
        .ok_or_else(|| Error::ModelNotFoundInRegistry { id: id.to_string() })?;

    // Prompt for license acceptance
    if !registry::prompt_license_acceptance(model)? {
        println!("Installation cancelled.");
        return Ok(());
    }

    // Download model and labels (async operation)
    let runtime = tokio::runtime::Runtime::new().map_err(|e| Error::Internal {
        message: format!("Failed to create async runtime: {e}"),
    })?;

    let installed = runtime.block_on(async { registry::install_model(model, language).await })?;

    println!();
    println!("Installation complete!");
    println!();
    println!("Model files saved to:");
    println!("  {}", installed.model.display());
    println!("  {}", installed.labels.display());
    if let Some(meta_path) = &installed.meta_model {
        println!("  {}", meta_path.display());
    }
    println!();

    // Prompt to set as default
    let should_set_default = if set_default {
        true
    } else {
        print!("Set as default model? [Y/n]: ");
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        !input.trim().eq_ignore_ascii_case("n")
    };

    // Add to config
    let mut config = load_default_config()?;

    // Parse model_type from string
    let model_type: ModelType = model
        .model_type
        .parse()
        .map_err(|_| Error::InvalidModelType {
            value: model.model_type.clone(),
        })?;

    config.models.insert(
        id.to_string(),
        ModelConfig {
            path: installed.model,
            labels: installed.labels,
            model_type,
            meta_model: installed.meta_model,
        },
    );

    if should_set_default {
        config.defaults.model = Some(id.to_string());
    }

    save_default_config(&config)?;

    if should_set_default {
        println!("Model '{id}' added to configuration and set as default.");
    } else {
        println!("Model '{id}' added to configuration.");
    }

    println!();
    println!("Ready to analyze:");
    println!("  birda recording.wav");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Create a minimal Config with a named model.
    fn config_with_model(name: &str) -> Config {
        let mut models = HashMap::new();
        models.insert(
            name.to_string(),
            ModelConfig {
                path: PathBuf::from("/path/to/model.onnx"),
                labels: PathBuf::from("/path/to/labels.txt"),
                model_type: ModelType::BirdnetV24,
                meta_model: None,
            },
        );
        Config {
            models,
            defaults: config::DefaultsConfig::default(),
            ..Default::default()
        }
    }

    /// Create default AnalyzeArgs (all None/false).
    fn default_args() -> AnalyzeArgs {
        AnalyzeArgs {
            model: None,
            model_path: None,
            labels_path: None,
            model_type: None,
            meta_model_path: None,
            format: None,
            output_dir: None,
            min_confidence: None,
            overlap: None,
            batch_size: None,
            combine: false,
            force: false,
            fail_fast: false,
            quiet: false,
            verbose: 0,
            no_progress: false,
            no_csv_bom: false,
            gpu: false,
            cpu: false,
            cuda: false,
            tensorrt: false,
            directml: false,
            coreml: false,
            rocm: false,
            openvino: false,
            onednn: false,
            qnn: false,
            acl: false,
            armnn: false,
            lat: None,
            lon: None,
            week: None,
            month: None,
            day: None,
            range_threshold: None,
            rerank: false,
            slist: None,
            stale_lock_timeout: None,
        }
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
    fn test_adhoc_with_meta_model() {
        let config = Config::default();
        let mut args = default_args();
        args.model_type = Some(ModelType::BirdnetV24);
        args.model_path = Some(PathBuf::from("/adhoc/model.onnx"));
        args.labels_path = Some(PathBuf::from("/adhoc/labels.txt"));
        args.meta_model_path = Some(PathBuf::from("/adhoc/meta.onnx"));

        let result = resolve_model_config(&args, &config);
        assert!(result.is_ok());

        let (model_config, _) = result.unwrap();
        assert_eq!(
            model_config.meta_model,
            Some(PathBuf::from("/adhoc/meta.onnx"))
        );
    }
}
