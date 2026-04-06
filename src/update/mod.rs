//! Self-update functionality for birda.
//!
//! Downloads and installs new releases from GitHub, replacing only the binary.
//! Warns when CUDA or ONNX Runtime library versions change between releases.

pub mod checksum;
pub mod constants;
pub mod manifest;
pub mod platform;
pub mod replace;
