//! Birda - Bird species detection CLI tool.
//!
//! This crate provides audio analysis capabilities using `BirdNET` and Perch models.

#![warn(missing_docs)]

pub mod error;

pub use error::{Error, Result};

/// Main entry point for birda CLI.
pub fn run() -> Result<()> {
    Ok(())
}
