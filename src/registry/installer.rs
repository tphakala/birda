//! Model download and installation logic.

use super::types::ModelEntry;
use crate::error::{Error, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use std::path::{Path, PathBuf};
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

/// Result of model installation.
#[derive(Debug)]
pub struct InstalledModel {
    /// Path to downloaded model file.
    pub model: PathBuf,
    /// Path to downloaded labels file.
    pub labels: PathBuf,
    /// Path to downloaded BSG calibration file (if available).
    pub bsg_calibration: Option<PathBuf>,
    /// Path to downloaded BSG migration file (if available).
    pub bsg_migration: Option<PathBuf>,
    /// Path to downloaded BSG distribution maps file (if available).
    pub bsg_distribution_maps: Option<PathBuf>,
}

/// Download a file with progress bar.
pub async fn download_file(client: &Client, url: &str, dest: &Path) -> Result<()> {
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| Error::DownloadFailed {
            url: url.to_string(),
            source: Box::new(e),
        })?;

    if !response.status().is_success() {
        return Err(Error::DownloadFailed {
            url: url.to_string(),
            source: format!("HTTP {}", response.status()).into(),
        });
    }

    let total_size = response.content_length().unwrap_or(0);

    // Create progress bar
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg}\n{bar:40.cyan/blue} {percent}% ({bytes}/{total_bytes})")
            .map_err(|e| Error::Internal {
                message: format!("Failed to create progress bar: {e}"),
            })?
            .progress_chars("█▓▒░ "),
    );
    // Use to_string_lossy() to handle non-UTF-8 filenames gracefully
    pb.set_message(format!(
        "Downloading {}...",
        dest.file_name().map_or_else(
            || std::borrow::Cow::Borrowed("file"),
            |n| n.to_string_lossy()
        )
    ));

    // Stream the download to a part file, then rename it onto the destination.
    // Rename is atomic within a filesystem, so a concurrent download or an
    // interrupted transfer can never leave a truncated file that a later
    // existence check would accept as complete.
    let part = part_path(dest);
    let result = stream_to_file(response, url, &part, &pb).await;

    if let Err(e) = result {
        // Best effort cleanup: the part file is useless without the rename,
        // and leaving it behind would waste disk on a retry loop.
        drop(tokio::fs::remove_file(&part).await);
        return Err(e);
    }

    finalize_download(&part, dest)?;
    pb.finish_with_message("Download complete");

    Ok(())
}

/// Path of the in-progress download file for a destination.
fn part_path(dest: &Path) -> PathBuf {
    // `file_name` is None only for paths ending in `..`, which a download
    // destination never is. Defaulting avoids an unwrap per the project lints.
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".");
    name.push(crate::constants::download::PARTIAL_SUFFIX);
    dest.with_file_name(name)
}

/// Move a completed download onto its destination, consuming the part file.
fn finalize_download(part: &Path, dest: &Path) -> Result<()> {
    std::fs::rename(part, dest).map_err(Error::Io)
}

/// Stream a response body into `dest`, updating the progress bar as it goes.
async fn stream_to_file(
    response: reqwest::Response,
    url: &str,
    dest: &Path,
    pb: &ProgressBar,
) -> Result<()> {
    let mut file = File::create(dest).await.map_err(Error::Io)?;
    let mut stream = response.bytes_stream();
    let mut downloaded = 0u64;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| Error::DownloadFailed {
            url: url.to_string(),
            source: Box::new(e),
        })?;

        file.write_all(&chunk).await.map_err(Error::Io)?;

        downloaded += chunk.len() as u64;
        pb.set_position(downloaded);
    }

    file.flush().await.map_err(Error::Io)?;

    Ok(())
}

/// Get models directory path.
pub fn models_dir() -> Result<PathBuf> {
    let data_dir = directories::ProjectDirs::from("", "", "birda")
        .ok_or(Error::ConfigDirNotFound)?
        .data_dir()
        .to_path_buf();

    Ok(data_dir.join("models"))
}

/// Install model from registry entry.
///
/// Downloads the model file, all available language label files,
/// and meta model if available. Returns paths to all downloaded files.
/// The `language` parameter determines which labels file is set as the default.
pub async fn install_model(model: &ModelEntry, language: Option<&str>) -> Result<InstalledModel> {
    let models_dir = models_dir()?;
    std::fs::create_dir_all(&models_dir).map_err(Error::Io)?;

    // Determine which language to use as default
    let language_code = language.unwrap_or(&model.files.labels.default_language);

    // Validate the requested language exists before downloading anything
    let default_language_variant = model
        .files
        .labels
        .languages
        .iter()
        .find(|l| l.code == language_code)
        .ok_or_else(|| Error::LanguageNotFound {
            code: language_code.to_string(),
            model_id: model.id.clone(),
        })?;

    // Create HTTP client with timeouts for all downloads
    let client = Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_mins(5))
        .build()
        .map_err(|e| Error::Internal {
            message: format!("Failed to create HTTP client: {e}"),
        })?;

    // Download model file
    let model_dest = models_dir.join(&model.files.model.filename);
    download_file(&client, &model.files.model.url, &model_dest).await?;

    // Download ALL language label files
    for language_variant in &model.files.labels.languages {
        let labels_dest = models_dir.join(&language_variant.filename);
        download_file(&client, &language_variant.url, &labels_dest).await?;
    }

    // Set the default labels path to the requested/default language
    let labels_dest = models_dir.join(&default_language_variant.filename);

    // Download BSG calibration file if available
    let bsg_calibration_path = if let Some(cal_info) = &model.files.bsg_calibration {
        let cal_dest = models_dir.join(&cal_info.filename);
        download_file(&client, &cal_info.url, &cal_dest).await?;
        Some(cal_dest)
    } else {
        None
    };

    // Download BSG migration file if available
    let bsg_migration_path = if let Some(mig_info) = &model.files.bsg_migration {
        let mig_dest = models_dir.join(&mig_info.filename);
        download_file(&client, &mig_info.url, &mig_dest).await?;
        Some(mig_dest)
    } else {
        None
    };

    // Download BSG distribution maps file if available
    let bsg_maps_path = if let Some(maps_info) = &model.files.bsg_distribution_maps {
        let maps_dest = models_dir.join(&maps_info.filename);
        download_file(&client, &maps_info.url, &maps_dest).await?;
        Some(maps_dest)
    } else {
        None
    };

    Ok(InstalledModel {
        model: model_dest,
        labels: labels_dest,
        bsg_calibration: bsg_calibration_path,
        bsg_migration: bsg_migration_path,
        bsg_distribution_maps: bsg_maps_path,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // Test setup code - panics are acceptable
mod tests {
    use super::*;

    #[test]
    fn test_part_path_appends_suffix() {
        let p = part_path(Path::new("/models/birdnet-geomodel-v3.0.2.onnx"));
        assert_eq!(
            p.file_name().unwrap(),
            "birdnet-geomodel-v3.0.2.onnx.part",
            "part file must sit beside the destination"
        );
        assert_eq!(p.parent(), Some(Path::new("/models")));
    }

    #[test]
    fn test_part_path_preserves_multi_dot_filenames() {
        // with_extension would mangle "v3.0.2.onnx" into "v3.0.2.part".
        let p = part_path(Path::new("/models/birdnet-geomodel-v3.0.2.onnx"));
        assert!(p.to_string_lossy().ends_with("v3.0.2.onnx.part"));
    }

    #[test]
    fn test_finalize_download_renames_and_clears_part() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("m.onnx");
        let part = part_path(&dest);
        std::fs::write(&part, b"data").unwrap();

        finalize_download(&part, &dest).unwrap();

        assert_eq!(std::fs::read(&dest).unwrap(), b"data");
        assert!(!part.exists(), "part file must not survive finalization");
    }

    #[test]
    fn test_finalize_download_overwrites_an_existing_destination() {
        // A checksum mismatch triggers a re-download onto an existing file.
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("m.onnx");
        std::fs::write(&dest, b"stale").unwrap();
        let part = part_path(&dest);
        std::fs::write(&part, b"fresh").unwrap();

        finalize_download(&part, &dest).unwrap();

        assert_eq!(std::fs::read(&dest).unwrap(), b"fresh");
    }

    #[test]
    fn test_finalize_download_errors_without_a_part_file() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("m.onnx");

        assert!(finalize_download(&part_path(&dest), &dest).is_err());
        assert!(!dest.exists());
    }

    #[test]
    fn test_models_dir_path() {
        let result = models_dir();
        assert!(result.is_ok());

        let path = result.unwrap();
        assert!(path.to_string_lossy().contains("birda"));
        assert!(path.to_string_lossy().ends_with("models"));
    }

    #[test]
    fn test_installed_model_default_labels_path() {
        let installed = InstalledModel {
            model: PathBuf::from("/models/birdnet-v24.onnx"),
            labels: PathBuf::from("/models/birdnet-v24-en.txt"),
            bsg_calibration: None,
            bsg_migration: None,
            bsg_distribution_maps: None,
        };

        assert_eq!(
            installed.labels.to_string_lossy(),
            "/models/birdnet-v24-en.txt"
        );
    }
}
