//! Model download and installation logic.

use super::types::ModelEntry;
use crate::error::{Error, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use std::path::{Path, PathBuf};
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

/// Download a file with progress bar.
pub async fn download_file(url: &str, dest: &Path) -> Result<()> {
    let client = Client::builder()
        .build()
        .map_err(|e| Error::DownloadFailed {
            url: url.to_string(),
            source: Box::new(e),
        })?;

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
    pb.set_message(format!(
        "Downloading {}...",
        dest.file_name().and_then(|n| n.to_str()).unwrap_or("file")
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
/// Downloads the model file and labels file for the specified language (or default).
/// Returns paths to the downloaded model and labels files.
pub async fn install_model(
    model: &ModelEntry,
    language: Option<&str>,
) -> Result<(PathBuf, PathBuf)> {
    let models_dir = models_dir()?;
    std::fs::create_dir_all(&models_dir).map_err(Error::Io)?;

    // Download model file
    let model_dest = models_dir.join(&model.files.model.filename);
    download_file(&model.files.model.url, &model_dest).await?;

    // Determine which language to download
    let language_code = language.unwrap_or(&model.files.labels.default_language);
    let language_variant = model
        .files
        .labels
        .languages
        .iter()
        .find(|l| l.code == language_code)
        .ok_or_else(|| Error::LanguageNotFound {
            code: language_code.to_string(),
            model_id: model.id.clone(),
        })?;

    // Download labels file
    let labels_dest = models_dir.join(&language_variant.filename);
    download_file(&language_variant.url, &labels_dest).await?;

    Ok((model_dest, labels_dest))
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
}
