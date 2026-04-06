//! Self-update functionality for birda.
//!
//! Downloads and installs new releases from GitHub, replacing only the binary.
//! Warns when CUDA or ONNX Runtime library versions change between releases.

pub mod checksum;
pub mod constants;
pub mod manifest;
pub mod platform;
pub mod replace;

use crate::error::{Error, Result};
use constants::{
    BUILT_CUDA_TOOLKIT_VERSION, BUILT_CUDNN_VERSION, BUILT_ONNXRUNTIME_VERSION, GITHUB_REPO,
    RELEASE_DOWNLOAD_URL, UPDATE_TEMP_SUFFIX,
};
use indicatif::{ProgressBar, ProgressStyle};
use manifest::Manifest;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Result of a version check.
pub enum UpdateCheck {
    /// Already running the latest version.
    UpToDate {
        /// Current version string.
        version: String,
    },
    /// A newer version is available.
    Available {
        /// Current version string.
        current: String,
        /// Available version string.
        available: String,
        /// The fetched manifest.
        manifest: Manifest,
    },
}

/// Result of performing an update.
pub struct UpdateResult {
    /// Previous version.
    pub old_version: String,
    /// New version.
    pub new_version: String,
    /// Whether a backup of the old binary was kept.
    pub backup_kept: bool,
    /// Path to the backup file (if kept).
    pub backup_path: Option<PathBuf>,
    /// Warnings about library version changes.
    pub warnings: Vec<String>,
}

/// Check if an update is available.
///
/// Fetches the manifest from the latest GitHub release and compares
/// versions using semver.
pub async fn check_for_update(client: &reqwest::Client) -> Result<UpdateCheck> {
    let manifest = manifest::fetch_manifest(client).await?;

    let current =
        semver::Version::parse(env!("CARGO_PKG_VERSION")).map_err(|e| Error::Internal {
            message: format!("failed to parse current version: {e}"),
        })?;

    let remote =
        semver::Version::parse(&manifest.version).map_err(|e| Error::UpdateFetchFailed {
            reason: format!(
                "manifest contains invalid version '{}': {e}",
                manifest.version
            ),
        })?;

    if current >= remote {
        Ok(UpdateCheck::UpToDate {
            version: current.to_string(),
        })
    } else {
        Ok(UpdateCheck::Available {
            current: current.to_string(),
            available: remote.to_string(),
            manifest,
        })
    }
}

/// Perform the full update: download, verify, extract, and replace.
pub async fn perform_update(
    client: &reqwest::Client,
    manifest: &Manifest,
    current_version: &str,
) -> Result<UpdateResult> {
    // 1. Resolve current exe path
    let exe_path = std::env::current_exe().map_err(|source| Error::UpdateExeNotFound { source })?;
    let exe_path = exe_path.canonicalize().unwrap_or_else(|_| exe_path.clone());

    // 2. Dev build guard
    if replace::is_dev_build(&exe_path) {
        return Err(Error::UpdateDevBuild);
    }

    // 3. Check library version compatibility
    let warnings = check_library_versions(manifest)?;

    // 4. Select the right asset
    let platform_key = platform::asset_key();
    let asset =
        manifest
            .assets
            .bin
            .get(platform_key)
            .ok_or_else(|| Error::UpdateUnsupportedPlatform {
                platform: platform_key.to_string(),
            })?;

    // 5. Check write permissions
    replace::check_write_permission(&exe_path)?;

    // 6. Validate asset filename and download the archive
    if asset.file.contains('/') || asset.file.contains('\\') || asset.file.contains("..") {
        return Err(Error::UpdateFetchFailed {
            reason: format!("invalid asset filename: {}", asset.file),
        });
    }

    let download_url = RELEASE_DOWNLOAD_URL
        .replace("{repo}", GITHUB_REPO)
        .replace("{file}", &asset.file);

    let parent_dir = exe_path
        .parent()
        .ok_or_else(|| Error::UpdateReplaceFailed {
            reason: "cannot determine parent directory of current binary".to_string(),
        })?;

    let archive_path = parent_dir.join(format!("{}.download", asset.file));
    if let Err(e) =
        download_with_progress(client, &download_url, &archive_path, &manifest.version).await
    {
        let _ = std::fs::remove_file(&archive_path);
        return Err(e);
    }

    // 7. Verify checksum
    info!("Verifying checksum...");
    if let Err(e) = checksum::verify_sha256(&archive_path, &asset.sha256) {
        // Clean up on failure
        let _ = std::fs::remove_file(&archive_path);
        return Err(e);
    }

    // 8. Extract to temp file in same directory (avoids EXDEV)
    let temp_binary = parent_dir.join(UPDATE_TEMP_SUFFIX);
    if let Err(e) = extract_binary(&archive_path, &temp_binary) {
        let _ = std::fs::remove_file(&archive_path);
        let _ = std::fs::remove_file(&temp_binary);
        return Err(e);
    }

    // Clean up the archive
    let _ = std::fs::remove_file(&archive_path);

    // 9. Set executable permissions (Unix)
    if let Err(e) = replace::set_executable(&temp_binary) {
        let _ = std::fs::remove_file(&temp_binary);
        return Err(e);
    }

    // 10. Replace binary
    let backup_kept = match replace::replace_binary(&exe_path, &temp_binary) {
        Ok(kept) => kept,
        Err(e) => {
            let _ = std::fs::remove_file(&temp_binary);
            return Err(e);
        }
    };

    let backup_path = if backup_kept {
        Some(exe_path.with_extension("old"))
    } else {
        None
    };

    Ok(UpdateResult {
        old_version: current_version.to_string(),
        new_version: manifest.version.clone(),
        backup_kept,
        backup_path,
        warnings,
    })
}

/// Check library versions and return warnings or block the update.
///
/// Blocks (returns `Err`) if ONNX Runtime major.minor changed.
/// Returns warnings (`Vec<String>`) for CUDA/cuDNN changes.
fn check_library_versions(manifest: &Manifest) -> Result<Vec<String>> {
    let mut warnings = Vec::new();

    // Skip all checks for dev builds (versions are "unknown")
    if BUILT_ONNXRUNTIME_VERSION == "unknown" {
        debug!("dev build detected, skipping library version checks");
        return Ok(warnings);
    }

    // ONNX Runtime: block on major.minor change (ABI break)
    if ort_major_minor_changed(
        BUILT_ONNXRUNTIME_VERSION,
        &manifest.dependencies.onnxruntime,
    ) {
        let tag = format!("v{}", manifest.version);
        return Err(Error::UpdateBlocked {
            current: BUILT_ONNXRUNTIME_VERSION.to_string(),
            required: manifest.dependencies.onnxruntime.clone(),
            release_url: format!("https://github.com/{GITHUB_REPO}/releases/tag/{tag}"),
        });
    }

    // CUDA checks only for CUDA builds
    if cfg!(feature = "cuda") && BUILT_CUDA_TOOLKIT_VERSION != "unknown" {
        if manifest.cuda.cuda_toolkit != BUILT_CUDA_TOOLKIT_VERSION {
            warnings.push(format!(
                "CUDA toolkit requirement changed ({} -> {}). If you use GPU acceleration, you may need to update your CUDA installation.",
                BUILT_CUDA_TOOLKIT_VERSION, manifest.cuda.cuda_toolkit,
            ));
        }
        if manifest.cuda.cudnn != BUILT_CUDNN_VERSION {
            warnings.push(format!(
                "cuDNN requirement changed ({} -> {}). If you use GPU acceleration, you may need to update cuDNN.",
                BUILT_CUDNN_VERSION, manifest.cuda.cudnn,
            ));
        }
    }

    Ok(warnings)
}

/// Check if the ONNX Runtime major.minor version has changed.
fn ort_major_minor_changed(current: &str, required: &str) -> bool {
    let Ok(current_ver) = semver::Version::parse(current) else {
        // Can't parse; assume changed to be safe if strings differ
        return current != required;
    };
    let Ok(required_ver) = semver::Version::parse(required) else {
        // Can't parse; assume changed to be safe if strings differ
        return current != required;
    };

    // Compare major.minor only
    current_ver.major != required_ver.major || current_ver.minor != required_ver.minor
}

/// Download a file with a progress bar.
async fn download_with_progress(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    version: &str,
) -> Result<()> {
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;

    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| Error::UpdateFetchFailed {
            reason: format!("download failed: {e}"),
        })?;

    if !response.status().is_success() {
        return Err(Error::UpdateFetchFailed {
            reason: format!("HTTP {} downloading {url}", response.status()),
        });
    }

    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg}\n{bar:40.cyan/blue} {percent}% ({bytes}/{total_bytes})")
            .map_err(|e| Error::Internal {
                message: format!("progress bar template error: {e}"),
            })?
            .progress_chars("##-"),
    );
    pb.set_message(format!("Downloading birda v{version}..."));

    let mut file = tokio::fs::File::create(dest).await.map_err(Error::Io)?;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| Error::UpdateFetchFailed {
            reason: format!("download stream error: {e}"),
        })?;
        file.write_all(&chunk).await.map_err(Error::Io)?;
        pb.inc(chunk.len() as u64);
    }

    file.flush().await.map_err(Error::Io)?;
    pb.finish_and_clear();

    Ok(())
}

/// Extract the binary from a downloaded archive.
fn extract_binary(archive_path: &Path, dest: &Path) -> Result<()> {
    let archive_name = archive_path.to_string_lossy();

    if archive_name.ends_with(".tar.gz.download") || archive_name.ends_with(".tar.gz") {
        extract_tar_gz(archive_path, dest)
    } else if archive_name.ends_with(".zip.download") || archive_name.ends_with(".zip") {
        extract_zip(archive_path, dest)
    } else {
        Err(Error::UpdateExtractFailed {
            reason: format!("unknown archive format: {}", archive_path.display()),
        })
    }
}

/// Extract the binary from a `.tar.gz` archive.
fn extract_tar_gz(archive_path: &Path, dest: &Path) -> Result<()> {
    use flate2::read::GzDecoder;
    use tar::Archive;

    let file = std::fs::File::open(archive_path).map_err(Error::Io)?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);

    let binary_name = platform::binary_name();

    for entry in archive.entries().map_err(|e| Error::UpdateExtractFailed {
        reason: format!("failed to read archive entries: {e}"),
    })? {
        let mut entry = entry.map_err(|e| Error::UpdateExtractFailed {
            reason: format!("failed to read archive entry: {e}"),
        })?;

        let path = entry.path().map_err(|e| Error::UpdateExtractFailed {
            reason: format!("failed to read entry path: {e}"),
        })?;

        // Security: reject entries with path traversal or absolute paths
        if path.is_absolute()
            || path
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(Error::UpdateExtractFailed {
                reason: "archive contains an unsafe path entry".to_string(),
            });
        }

        let filename = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        // Skip non-file entries (directories, symlinks) to avoid extracting
        // a directory named "birda" as a 0-byte file
        if !entry.header().entry_type().is_file() {
            continue;
        }

        if filename == binary_name {
            let mut output = std::fs::File::create(dest).map_err(Error::Io)?;
            std::io::copy(&mut entry, &mut output).map_err(|e| Error::UpdateExtractFailed {
                reason: format!("failed to extract binary: {e}"),
            })?;
            return Ok(());
        }
    }

    Err(Error::UpdateExtractFailed {
        reason: format!("binary '{binary_name}' not found in archive"),
    })
}

/// Extract the binary from a `.zip` archive.
fn extract_zip(archive_path: &Path, dest: &Path) -> Result<()> {
    let file = std::fs::File::open(archive_path).map_err(Error::Io)?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| Error::UpdateExtractFailed {
        reason: format!("failed to open zip archive: {e}"),
    })?;

    let binary_name = platform::binary_name();

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| Error::UpdateExtractFailed {
                reason: format!("failed to read zip entry: {e}"),
            })?;

        // Skip directories
        if entry.is_dir() {
            continue;
        }

        let path = entry
            .enclosed_name()
            .ok_or_else(|| Error::UpdateExtractFailed {
                reason: "zip entry has unsafe path".to_string(),
            })?;

        let filename = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        if filename == binary_name {
            let mut output = std::fs::File::create(dest).map_err(Error::Io)?;
            std::io::copy(&mut entry, &mut output).map_err(|e| Error::UpdateExtractFailed {
                reason: format!("failed to extract binary: {e}"),
            })?;
            return Ok(());
        }
    }

    Err(Error::UpdateExtractFailed {
        reason: format!("binary '{binary_name}' not found in zip archive"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ort_major_minor_changed_same() {
        assert!(!ort_major_minor_changed("1.24.2", "1.24.3"));
    }

    #[test]
    fn test_ort_major_minor_changed_minor_bump() {
        assert!(ort_major_minor_changed("1.24.2", "1.25.0"));
    }

    #[test]
    fn test_ort_major_minor_changed_major_bump() {
        assert!(ort_major_minor_changed("1.24.2", "2.0.0"));
    }

    #[test]
    fn test_ort_major_minor_changed_same_short() {
        assert!(!ort_major_minor_changed("1.24", "1.24"));
    }

    #[test]
    fn test_ort_major_minor_changed_unparseable() {
        // Single segment can't be split; falls back to string comparison
        assert!(ort_major_minor_changed("unknown", "1.24.2"));
    }
}
