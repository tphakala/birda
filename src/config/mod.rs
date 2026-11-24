//! Configuration loading and management.

pub mod bat;
pub mod calibration;
mod file;
pub mod geomodel;
pub mod openvino;
mod paths;
pub mod range_filter;
mod types;
mod validate;

pub use bat::{BatConfig, BatRegion};
pub use file::{
    load_config_file, load_default_config, save_config, update_config, update_config_repairing,
};
pub use geomodel::{GeomodelRequest, GeomodelResolution, configured_paths, resolve_geomodel};
pub use openvino::{OpenVinoDeviceSpec, OpenVinoSettings, PciAddress};
pub use paths::{config_dir, config_file_path, data_dir, openvino_cache_dir, tensorrt_cache_dir};
pub use types::{
    AcceptedDocument, Config, CsvColumnsConfig, DefaultsConfig, InferenceConfig, InferenceDevice,
    ModelConfig, ModelType, OutputConfig, OutputFormat, OutputMode, UnmatchedPolicy,
};
pub use validate::{get_model, validate_config, validate_model_config};
