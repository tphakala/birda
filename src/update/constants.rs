//! Constants for the update command.

/// GitHub repository used for release downloads.
pub const GITHUB_REPO: &str = "tphakala/birda";

/// URL pattern for downloading from the latest GitHub release.
/// The `{repo}` placeholder is replaced with `GITHUB_REPO`.
/// The `{file}` placeholder is replaced with the asset filename.
pub const RELEASE_DOWNLOAD_URL: &str = "https://github.com/{repo}/releases/latest/download/{file}";

/// Filename of the release manifest.
pub const MANIFEST_FILENAME: &str = "manifest.json";

/// Temporary file suffix used during extraction.
pub const UPDATE_TEMP_SUFFIX: &str = ".birda-update-new.tmp";

/// Backup file extension for the old binary on Unix.
pub const BACKUP_EXTENSION: &str = ".old";

/// Embedded ONNX Runtime version from build time.
pub const BUILT_ONNXRUNTIME_VERSION: &str = env!("BIRDA_ONNXRUNTIME_VERSION");

/// Embedded CUDA toolkit version from build time.
pub const BUILT_CUDA_TOOLKIT_VERSION: &str = env!("BIRDA_CUDA_TOOLKIT_VERSION");

/// Embedded cuDNN version from build time.
pub const BUILT_CUDNN_VERSION: &str = env!("BIRDA_CUDNN_VERSION");
