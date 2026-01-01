//! Birda - Bird species detection CLI tool.
//!
//! This crate provides audio analysis capabilities using `BirdNET` and Perch models.

#![warn(missing_docs)]

pub mod audio;
pub mod cli;
pub mod config;
pub mod constants;
pub mod error;
pub mod locking;
pub mod output;

use clap::Parser;
use cli::{Cli, Command};
use config::load_default_config;
use tracing::info;

pub use error::{Error, Result};

/// Main entry point for birda CLI.
pub fn run() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    init_logging(cli.analyze.verbose, cli.analyze.quiet);

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

    info!(
        "Would analyze {} input(s) with config: {:?}",
        cli.inputs.len(),
        config.defaults
    );

    Ok(())
}

fn init_logging(verbose: u8, quiet: bool) {
    use tracing_subscriber::{fmt, EnvFilter};

    let level = if quiet {
        "warn"
    } else {
        match verbose {
            0 => "info",
            1 => "debug",
            _ => "trace",
        }
    };

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

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
            let path = config::config_file_path()?;
            println!("Would create config at: {}", path.display());
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
                    println!("  {name}: {}", model.path.display());
                }
            }
            Ok(())
        }
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
            println!("  Path: {}", model.path.display());
            println!("  Labels: {}", model.labels.display());
            if let Some(ref t) = model.model_type {
                println!("  Type: {t}");
            }
            Ok(())
        }
    }
}
