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

use clap::Parser;
use cli::{AnalyzeArgs, Cli, Command};
use config::{
    Config, InferenceDevice, ModelConfig, ModelType, config_file_path, load_default_config,
    save_default_config,
};
use inference::BirdClassifier;
use pipeline::{ProcessCheck, collect_input_files, output_dir_for, process_file, should_process};
use std::path::PathBuf;
use tracing::{error, info, warn};

pub use error::{Error, Result};

/// Main entry point for birda CLI.
pub fn run() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    init_logging(cli.analyze.verbose, cli.analyze.quiet);

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
    if cli.inputs.is_empty() {
        return Err(Error::NoInputFiles);
    }

    // Run analysis
    analyze_files(&cli.inputs, &cli.analyze, &config)
}

/// Analyze input files with the given options.
fn analyze_files(inputs: &[PathBuf], args: &AnalyzeArgs, config: &Config) -> Result<()> {
    // Collect all input files
    let files = collect_input_files(inputs)?;
    if files.is_empty() {
        return Err(Error::NoInputFiles);
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

    // Build classifier
    info!("Loading model: {}", model_name);
    let classifier = BirdClassifier::from_config(model_config, device, min_confidence, 10)?;

    // Process files
    let mut processed = 0;
    let mut skipped = 0;
    let mut errors = 0;
    let mut total_detections = 0;

    for file in &files {
        let file_output_dir = output_dir_for(file, output_dir.as_deref());

        // Check if should process
        match should_process(file, &file_output_dir, &formats, force) {
            ProcessCheck::SkipExists => {
                info!("Skipping (output exists): {}", file.display());
                skipped += 1;
                continue;
            }
            ProcessCheck::SkipLocked => {
                info!("Skipping (locked): {}", file.display());
                skipped += 1;
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
        ) {
            Ok(result) => {
                processed += 1;
                total_detections += result.detections;
            }
            Err(e) => {
                error!("Failed to process {}: {}", file.display(), e);
                errors += 1;
                if fail_fast {
                    return Err(e);
                }
            }
        }
    }

    // Summary
    info!(
        "Complete: {} processed, {} skipped, {} errors, {} total detections",
        processed, skipped, errors, total_detections
    );

    if errors > 0 && !fail_fast {
        warn!("{} file(s) had errors", errors);
    }

    Ok(())
}

fn init_logging(verbose: u8, quiet: bool) {
    use tracing_subscriber::{EnvFilter, fmt};

    let (level, ort_level) = if quiet {
        ("warn", "warn")
    } else {
        match verbose {
            0 => ("info", "warn"),
            1 => ("debug", "warn"),
            2 => ("trace", "info"),
            _ => ("trace", "debug"), // -vvv enables ORT debug logging
        }
    };

    // Verbosity level also controls the ONNX Runtime (ort) log level.
    let filter_str = format!("{level},ort={ort_level}");

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&filter_str));

    fmt().with_env_filter(filter).init();
}

fn handle_command(command: Command, config: &config::Config) -> Result<()> {
    match command {
        Command::Config { action } => handle_config_command(action),
        Command::Models { action } => handle_models_command(action, config),
    }
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
        ModelsAction::Info { name } => {
            let model = config::get_model(config, &name)?;
            println!("Model: {name}");
            println!("  Type: {}", model.model_type);
            println!("  Path: {}", model.path.display());
            println!("  Labels: {}", model.labels.display());
            Ok(())
        }
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
