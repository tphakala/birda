//! Birda - Bird species detection CLI tool.
//!
//! This crate provides audio analysis capabilities using `BirdNET` and Perch models.

#![warn(missing_docs)]

pub mod audio;
pub mod cli;
pub mod config;
pub mod constants;
pub mod error;
pub mod inference;
pub mod locking;
pub mod output;
pub mod pipeline;
pub mod registry;
pub mod utils;

use clap::Parser;
use cli::{AnalyzeArgs, Cli, Command};
use config::{
    Config, InferenceDevice, ModelConfig, ModelType, config_file_path, load_default_config,
    range_filter::build_range_filter_config, save_default_config,
};
use constants::DEFAULT_TOP_K;
use inference::BirdClassifier;
use pipeline::{ProcessCheck, collect_input_files, output_dir_for, process_file, should_process};
use std::collections::HashSet;
use std::path::PathBuf;
use tracing::{error, info, warn};

pub use error::{Error, Result};

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

    // Initialize ONNX Runtime (auto-detects bundled libraries)
    birdnet_onnx::init_runtime().map_err(|e| Error::RuntimeInitialization {
        reason: e.to_string(),
    })?;

    // Load configuration
    let config = load_default_config()?;

    // Handle subcommands
    if let Some(command) = cli.command {
        return handle_command(command, &config);
    }

    // Default: analyze files
    // Show help if no inputs provided
    if cli.inputs.is_empty() {
        cli::help::print_smart_help(&config);
        std::process::exit(0);
    }

    // Run analysis
    analyze_files(&cli.inputs, &cli.analyze, &config)
}

/// Analyze input files with the given options.
fn analyze_files(inputs: &[PathBuf], args: &AnalyzeArgs, config: &Config) -> Result<()> {
    use crate::output::progress;
    use std::time::Instant;

    let total_start = Instant::now();

    // Collect all input files
    let files = collect_input_files(inputs)?;
    if files.is_empty() {
        return Err(Error::NoValidAudioFiles);
    }

    info!("Found {} audio file(s) to process", files.len());

    // Resolve model configuration
    let model_name = args
        .model
        .clone()
        .or_else(|| config.defaults.model.clone())
        .ok_or_else(|| Error::ConfigValidation {
            message: "no model specified (use -m or set defaults.model in config)".to_string(),
        })?;

    let model_config = config::get_model(config, &model_name)?;

    // Resolve other settings
    let min_confidence = args
        .min_confidence
        .unwrap_or(config.defaults.min_confidence);
    let overlap = args.overlap.unwrap_or(config.defaults.overlap);
    let batch_size = args.batch_size.unwrap_or(config.defaults.batch_size);
    let formats = args
        .format
        .clone()
        .unwrap_or_else(|| config.defaults.formats.clone());
    let output_dir = args.output_dir.clone();
    let force = args.force;
    let fail_fast = args.fail_fast;

    // Resolve device
    let device = if args.gpu {
        InferenceDevice::Gpu
    } else if args.cpu {
        InferenceDevice::Cpu
    } else {
        config.inference.device
    };

    // Build range filter config
    let range_filter_config = build_range_filter_config(args, config, model_config, &model_name)?;

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

    // Build classifier
    info!("Loading model: {}", model_name);
    let classifier = BirdClassifier::from_config(
        model_config,
        device,
        min_confidence,
        DEFAULT_TOP_K,
        range_filter_config,
        species_list,
    )?;

    // Create file progress bar
    let progress_enabled = !args.quiet && !args.no_progress;
    let file_progress = progress::create_file_progress(files.len(), progress_enabled);

    // Process files
    let mut processed = 0;
    let mut skipped = 0;
    let mut errors = 0;
    let mut total_detections = 0;
    let mut total_segments = 0;

    for file in &files {
        let file_output_dir = output_dir_for(file, output_dir.as_deref());

        // Check if should process
        match should_process(file, &file_output_dir, &formats, force) {
            ProcessCheck::SkipExists => {
                info!("Skipping (output exists): {}", file.display());
                skipped += 1;
                progress::inc_progress(file_progress.as_ref());
                continue;
            }
            ProcessCheck::SkipLocked => {
                info!("Skipping (locked): {}", file.display());
                skipped += 1;
                progress::inc_progress(file_progress.as_ref());
                continue;
            }
            ProcessCheck::Process => {}
        }

        // Process the file
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
        ) {
            Ok(result) => {
                processed += 1;
                total_detections += result.detections;
                total_segments += result.segments;
            }
            Err(e) => {
                error!("Failed to process {}: {}", file.display(), e);
                errors += 1;
                if fail_fast {
                    progress::finish_progress(file_progress, "Failed");
                    return Err(e);
                }
            }
        }
        progress::inc_progress(file_progress.as_ref());
    }

    progress::finish_progress(file_progress, "Complete");

    // Summary
    let total_duration = total_start.elapsed().as_secs_f64();
    info!(
        "Complete: {} processed, {} skipped, {} errors, {} total detections in {:.2}s",
        processed, skipped, errors, total_detections, total_duration
    );

    if processed > 0 {
        #[allow(clippy::cast_precision_loss)]
        let avg_segments_per_sec = if total_duration > 0.0 {
            total_segments as f64 / total_duration
        } else {
            0.0
        };
        info!(
            "Performance: {:.1} segments/sec overall",
            avg_segments_per_sec
        );
    }

    if errors > 0 && !fail_fast {
        warn!("{} file(s) had errors", errors);
    }

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

    fmt().with_env_filter(filter).init();
}

fn handle_command(command: Command, config: &config::Config) -> Result<()> {
    match command {
        Command::Config { action } => handle_config_command(action),
        Command::Models { action } => handle_models_command(action, config),
        Command::Providers => {
            handle_providers_command();
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
            output, lat, lon, week, month, day, threshold, sort, model,
        ),
    }
}

fn handle_providers_command() {
    use birdnet_onnx::available_execution_providers;

    let providers = available_execution_providers();

    println!("Available execution providers:");
    println!();

    for provider in &providers {
        let description = match provider {
            birdnet_onnx::ExecutionProviderInfo::Cpu => "CPU (always available)",
            birdnet_onnx::ExecutionProviderInfo::Cuda => "CUDA (NVIDIA GPU acceleration)",
            birdnet_onnx::ExecutionProviderInfo::TensorRt => {
                "TensorRT (NVIDIA optimized inference)"
            }
            birdnet_onnx::ExecutionProviderInfo::DirectMl => "DirectML (Windows GPU acceleration)",
            birdnet_onnx::ExecutionProviderInfo::CoreMl => "CoreML (Apple GPU/Neural Engine)",
            birdnet_onnx::ExecutionProviderInfo::Rocm => "ROCm (AMD GPU acceleration)",
            birdnet_onnx::ExecutionProviderInfo::OpenVino => "OpenVINO (Intel optimization)",
            birdnet_onnx::ExecutionProviderInfo::OneDnn => "oneDNN (Intel CPU optimization)",
            birdnet_onnx::ExecutionProviderInfo::Qnn => "QNN (Qualcomm Neural Network)",
            birdnet_onnx::ExecutionProviderInfo::Acl => "ACL (Arm Compute Library)",
            birdnet_onnx::ExecutionProviderInfo::ArmNn => "ArmNN (Arm Neural Network)",
        };
        println!("  ✓ {description}");
    }

    println!();
    println!("To use a specific provider:");
    println!("  --gpu       Use CUDA (if available)");
    println!("  --cpu       Use CPU only");
    println!("  (default)   Auto-select (GPU if available, fallback to CPU)");
    println!();
    println!("Note: This shows compile-time availability. Runtime availability may");
    println!("      differ based on drivers and hardware. Check log output for actual");
    println!("      provider selection during inference.");
}

fn handle_config_command(action: cli::ConfigAction) -> Result<()> {
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
            println!("{config:#?}");
            Ok(())
        }
        ConfigAction::Path => {
            let path = config::config_file_path()?;
            println!("{}", path.display());
            Ok(())
        }
    }
}

fn handle_models_command(action: cli::ModelsAction, config: &config::Config) -> Result<()> {
    use cli::ModelsAction;

    match action {
        ModelsAction::List => {
            if config.models.is_empty() {
                println!("No models configured.");
            } else {
                println!("Configured models:");
                for (name, model) in &config.models {
                    let default_marker = config.defaults.model.as_ref().is_some_and(|d| d == name);
                    println!(
                        "  {} ({}){}",
                        name,
                        model.model_type,
                        if default_marker { " [default]" } else { "" }
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
            if registry::find_model(&registry, &id).is_some() {
                if languages {
                    registry::show_languages(&registry, &id)?;
                } else {
                    registry::show_info(&registry, &id)?;
                }
            } else {
                // Fall back to configured model
                let model = config::get_model(config, &id)?;
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
