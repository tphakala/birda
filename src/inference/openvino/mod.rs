//! Native `OpenVINO` backend for Intel GPUs.
//!
//! Compiled on every platform and loaded at run time: the `openvino` crate is
//! built with `runtime-linking`, so the user's `OpenVINO` install is opened by
//! [`OpenVinoEngine::new`] and [`listing::list`] and not before.

pub mod cache;
pub mod device;
pub mod engine;
pub mod listing;
pub mod parity;
pub mod runtime;

pub use device::{DeviceKind, GpuClass, OvDevice, ResolveError, resolve, selectors_for};
pub use engine::{DeviceReport, OpenVinoEngine};
pub use listing::{DeviceListing, list};
