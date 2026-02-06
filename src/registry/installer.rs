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
    /// Path to downloaded meta model file (if available).
    pub meta_model: Option<PathBuf>,
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

    // Stream download
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

    pb.finish_with_message("Download complete");

    // TODO: Implement SHA256 checksum verification for downloaded files
    // Currently all SHA256 checksums in registry.json are null, which is a security risk.
    // Future work should:
    // 1. Generate SHA256 checksums for all model and label files
    // 2. Update registry.json with the checksums
    // 3. Verify downloaded files against checksums here before returning
    // See: https://github.com/tphakala/birda/issues/XX

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

    // Create HTTP client with timeouts for all downloads
    let client = Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(|e| Error::Internal {
            message: format!("Failed to create HTTP client: {e}"),
        })?;

    // Download model file
    let model_dest = models_dir.join(&model.files.model.filename);
    download_file(&client, &model.files.model.url, &model_dest).await?;

    // Determine which language to use as default
    let language_code = language.unwrap_or(&model.files.labels.default_language);

    // Validate the requested language exists
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

    // Download ALL language label files
    for language_variant in &model.files.labels.languages {
        let labels_dest = models_dir.join(&language_variant.filename);
        download_file(&client, &language_variant.url, &labels_dest).await?;
    }

    // Set the default labels path to the requested/default language
    let labels_dest = models_dir.join(&default_language_variant.filename);

    // Download meta model if available
    let meta_model_path = if let Some(meta_info) = &model.files.meta_model {
        let meta_dest = models_dir.join(&meta_info.filename);
        download_file(&client, &meta_info.url, &meta_dest).await?;
        Some(meta_dest)
    } else {
        None
    };

    Ok(InstalledModel {
        model: model_dest,
        labels: labels_dest,
        meta_model: meta_model_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
            meta_model: None,
        };

        assert_eq!(
            installed.labels.to_string_lossy(),
            "/models/birdnet-v24-en.txt"
        );
    }
}
