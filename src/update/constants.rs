//! Constants for the update command.

/// GitHub repository used for release downloads.
pub const GITHUB_REPO: &str = "tphakala/birda";

/// URL pattern for downloading from the latest GitHub release.
/// The `{repo}` placeholder is replaced with `GITHUB_REPO`.
/// The `{file}` placeholder is replaced with the asset filename.
pub const RELEASE_DOWNLOAD_URL: &str = "https://github.com/{repo}/releases/latest/download/{file}";

/// Filename of the release manifest.
pub const MANIFEST_FILENAME: &str = "manifest.json";

/// Filename prefix for the extracted-binary temporary written during a self-update.
///
/// A unique random suffix is appended per run (see `reserve_temp_path`), so two
/// concurrent `birda update` invocations sharing this directory cannot write the
/// same temporary and corrupt each other's extraction.
pub const UPDATE_TEMP_PREFIX: &str = "birda-update-new-";

/// Maximum manifest response size in bytes (1 MiB).
pub const MANIFEST_MAX_BYTES: u64 = 1024 * 1024;

/// Read-buffer size (64 KiB) for streaming a file through the SHA256 hasher.
///
/// Bounds peak memory during checksum verification to this size regardless of
/// the file's length, so hashing a several-hundred-MB model does not first
/// materialise the whole file in memory.
pub const HASH_CHUNK_BYTES: usize = 64 * 1024;

/// Length of a SHA-256 digest rendered as a lowercase hex string.
///
/// A SHA-256 digest is 32 bytes, and each byte becomes two hex characters.
pub const SHA256_HEX_LEN: usize = 64;

/// HTTP connect timeout in seconds for update requests.
pub const CONNECT_TIMEOUT_SECS: u64 = 30;

/// Total HTTP timeout in seconds for update download requests.
pub const DOWNLOAD_TIMEOUT_SECS: u64 = 300;

/// Embedded ONNX Runtime version from build time.
pub const BUILT_ONNXRUNTIME_VERSION: &str = env!("BIRDA_ONNXRUNTIME_VERSION");

/// Embedded CUDA toolkit version from build time.
pub const BUILT_CUDA_TOOLKIT_VERSION: &str = env!("BIRDA_CUDA_TOOLKIT_VERSION");

/// Embedded cuDNN version from build time.
pub const BUILT_CUDNN_VERSION: &str = env!("BIRDA_CUDNN_VERSION");
