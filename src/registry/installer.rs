//! Model download and installation logic.

use super::types::{ModelEntry, RangeFilterAsset};
use crate::error::{Error, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use std::path::{Path, PathBuf};
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

/// Identifier accepted by `birda models install` for the shared range filter.
pub const GEOMODEL_INSTALL_ID: &str = "geomodel";

/// Paths to the installed shared range filter files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstalledRangeFilter {
    /// Path to the geomodel ONNX file.
    pub model: PathBuf,
    /// Path to the geomodel labels file.
    pub labels: PathBuf,
}

impl InstalledRangeFilter {
    /// Whether both files are present on disk.
    #[must_use]
    pub fn is_installed(&self) -> bool {
        self.model.is_file() && self.labels.is_file()
    }

    /// Verify both files against the checksums declared in the registry.
    ///
    /// Files whose registry entry carries no checksum are accepted as-is.
    pub fn verify(&self, asset: &RangeFilterAsset) -> Result<()> {
        if let Some(sum) = asset.model.sha256.as_deref() {
            crate::update::checksum::verify_sha256(&self.model, sum)?;
        }
        if let Some(sum) = asset.labels.sha256.as_deref() {
            crate::update::checksum::verify_sha256(&self.labels, sum)?;
        }
        Ok(())
    }
}

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
    let part = part_path(dest)?;
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
///
/// The name is qualified with this process's id so two concurrent birda
/// processes downloading the same destination cannot write into one file. A
/// shared part name lets their writes interleave, and lets one process's error
/// cleanup unlink the other's in-progress transfer.
fn part_path(dest: &Path) -> Result<PathBuf> {
    let name = dest.file_name().ok_or_else(|| Error::Internal {
        message: format!("download destination has no file name: {}", dest.display()),
    })?;

    let mut part = name.to_os_string();
    part.push(format!(
        ".{}.{}",
        std::process::id(),
        crate::constants::download::PARTIAL_SUFFIX
    ));

    Ok(dest.with_file_name(part))
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

    // Flush the userspace buffer and then the kernel's, so a crash after the
    // rename cannot publish a short file at the destination.
    file.flush().await.map_err(Error::Io)?;
    file.sync_all().await.map_err(Error::Io)?;

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

/// Expected on-disk paths for the shared range filter asset. Performs no I/O.
pub fn geomodel_paths(asset: &RangeFilterAsset) -> Result<InstalledRangeFilter> {
    let dir = models_dir()?;
    Ok(InstalledRangeFilter {
        model: dir.join(&asset.model.filename),
        labels: dir.join(&asset.labels.filename),
    })
}

/// Download and verify the shared range filter asset.
///
/// Idempotent: when both files are present and their checksums match, nothing
/// is downloaded. A checksum mismatch triggers a fresh download, since the
/// likeliest cause is a file left behind by an older birda version.
pub async fn install_range_filter(asset: &RangeFilterAsset) -> Result<InstalledRangeFilter> {
    let paths = geomodel_paths(asset)?;
    if paths.is_installed() && paths.verify(asset).is_ok() {
        return Ok(paths);
    }

    let dir = models_dir()?;
    std::fs::create_dir_all(&dir).map_err(Error::Io)?;
    let client = http_client()?;

    download_file(&client, &asset.model.url, &paths.model).await?;
    download_file(&client, &asset.labels.url, &paths.labels).await?;

    // A post-download checksum failure must not leave the bad files installed.
    // Consumers gate on presence, so a corrupt file left behind here would be
    // loaded on every later run without ever being re-verified.
    if let Err(e) = paths.verify(asset) {
        drop(std::fs::remove_file(&paths.model));
        drop(std::fs::remove_file(&paths.labels));
        return Err(e);
    }

    Ok(paths)
}

/// Report files in the models directory that birda no longer uses.
///
/// Currently this is the `BirdNET` v2.4 meta model, superseded by the shared
/// `BirdNET` Geomodel v3.0.2.
pub fn find_obsolete_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut found = Vec::new();

    for name in crate::constants::obsolete_files::NAMES {
        let candidate = dir.join(name);
        if candidate.is_file() {
            found.push(candidate);
        }
    }

    Ok(found)
}

/// Build the HTTP client used for every registry download.
fn http_client() -> Result<Client> {
    Client::builder()
        .connect_timeout(std::time::Duration::from_secs(
            crate::constants::download::CONNECT_TIMEOUT_SECS,
        ))
        .timeout(std::time::Duration::from_mins(
            crate::constants::download::REQUEST_TIMEOUT_MINS,
        ))
        .build()
        .map_err(|e| Error::Internal {
            message: format!("Failed to create HTTP client: {e}"),
        })
}

/// Install model from registry entry.
///
/// Downloads the model file and all available language label files, plus any
/// BSG companion files. Returns paths to all downloaded files. The `language`
/// parameter determines which labels file is set as the default.
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
    let client = http_client()?;

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

    use crate::registry::types::{FileInfo, LicenseInfo};

    /// SHA256 of the byte string `b"right"`, used by the checksum tests.
    const RIGHT_SHA256: &str = "27042f4e6eca7d0b2a7ee4026df2ecfa51d3339e6d122aa099118ecd8563bad9";

    fn test_asset() -> RangeFilterAsset {
        RangeFilterAsset {
            id: "birdnet-geomodel-v3".into(),
            name: "BirdNET Geomodel v3.0.2".into(),
            version: "3.0.2".into(),
            vendor: "Cornell Lab".into(),
            license: LicenseInfo {
                r#type: "CC-BY-SA-4.0".into(),
                url: "https://creativecommons.org/licenses/by-sa/4.0/".into(),
                commercial_use: true,
                attribution_required: true,
                share_alike: true,
            },
            species_count: 12012,
            model: FileInfo {
                url: "https://example.com/birdnet-geomodel-v3.0-fp32.onnx".into(),
                filename: "birdnet-geomodel-v3.0.2.onnx".into(),
                sha256: Some(RIGHT_SHA256.into()),
                size_bytes: None,
            },
            labels: FileInfo {
                url: "https://example.com/geomodel_v3.0.2_labels.txt".into(),
                filename: "birdnet-geomodel-v3.0.2-labels.txt".into(),
                sha256: Some(RIGHT_SHA256.into()),
                size_bytes: None,
            },
        }
    }

    #[test]
    fn test_geomodel_paths_use_asset_filenames() {
        let asset = test_asset();
        let paths = geomodel_paths(&asset).unwrap();

        assert!(paths.model.ends_with("birdnet-geomodel-v3.0.2.onnx"));
        assert!(paths.labels.ends_with("birdnet-geomodel-v3.0.2-labels.txt"));
        assert_eq!(
            paths.model.parent(),
            paths.labels.parent(),
            "both files live in the models directory"
        );
    }

    #[test]
    fn test_is_installed_false_when_files_absent() {
        let dir = tempfile::tempdir().unwrap();
        let paths = InstalledRangeFilter {
            model: dir.path().join("absent.onnx"),
            labels: dir.path().join("absent.txt"),
        };

        assert!(!paths.is_installed());
    }

    #[test]
    fn test_is_installed_requires_both_files() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("m.onnx");
        std::fs::write(&model, b"x").unwrap();
        let paths = InstalledRangeFilter {
            model,
            labels: dir.path().join("absent.txt"),
        };

        assert!(
            !paths.is_installed(),
            "missing labels must count as not installed"
        );
    }

    #[test]
    fn test_verify_accepts_matching_checksums() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("m.onnx");
        let labels = dir.path().join("l.txt");
        std::fs::write(&model, b"right").unwrap();
        std::fs::write(&labels, b"right").unwrap();
        let paths = InstalledRangeFilter { model, labels };

        paths.verify(&test_asset()).unwrap();
    }

    #[test]
    fn test_verify_rejects_wrong_content() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("m.onnx");
        let labels = dir.path().join("l.txt");
        std::fs::write(&model, b"wrong").unwrap();
        std::fs::write(&labels, b"right").unwrap();
        let paths = InstalledRangeFilter { model, labels };

        assert!(paths.verify(&test_asset()).is_err());
    }

    #[test]
    fn test_verify_rejects_wrong_labels_with_a_correct_model() {
        // The sibling test above pairs a wrong model with right labels, so the
        // labels-side checksum block could be deleted outright and every verify
        // test would still pass. A birda that downloaded the labels and never
        // checked them would then ship with a green suite. This is the only
        // test that pins the labels side, so the pair has to stay asymmetric.
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("m.onnx");
        let labels = dir.path().join("l.txt");
        std::fs::write(&model, b"right").unwrap();
        std::fs::write(&labels, b"wrong").unwrap();
        let paths = InstalledRangeFilter { model, labels };

        assert!(
            paths.verify(&test_asset()).is_err(),
            "a correct model must not excuse corrupt labels"
        );
    }

    #[test]
    fn test_is_installed_requires_the_model_too() {
        // The sibling test covers a present model with absent labels. Without
        // this direction the model-existence check could be dropped and both
        // would still pass.
        let dir = tempfile::tempdir().unwrap();
        let labels = dir.path().join("l.txt");
        std::fs::write(&labels, b"present").unwrap();
        let paths = InstalledRangeFilter {
            model: dir.path().join("absent.onnx"),
            labels,
        };

        assert!(
            !paths.is_installed(),
            "a missing model must count as not installed"
        );
    }

    #[test]
    fn test_verify_skips_files_without_a_declared_checksum() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("m.onnx");
        let labels = dir.path().join("l.txt");
        std::fs::write(&model, b"anything").unwrap();
        std::fs::write(&labels, b"anything").unwrap();
        let mut asset = test_asset();
        asset.model.sha256 = None;
        asset.labels.sha256 = None;
        let paths = InstalledRangeFilter { model, labels };

        paths.verify(&asset).unwrap();
    }

    #[test]
    fn test_find_obsolete_files_detects_the_v24_meta_model() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("birdnet-v24-meta.onnx"), b"x").unwrap();

        let found = find_obsolete_files(dir.path()).unwrap();

        assert_eq!(found.len(), 1);
        assert!(found[0].ends_with("birdnet-v24-meta.onnx"));
    }

    #[test]
    fn test_find_obsolete_files_reports_nothing_on_a_clean_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("birdnet-geomodel-v3.0.2.onnx"), b"x").unwrap();

        assert!(find_obsolete_files(dir.path()).unwrap().is_empty());
    }

    #[test]
    fn test_part_path_appends_suffix() {
        let p = part_path(Path::new("/models/birdnet-geomodel-v3.0.2.onnx")).unwrap();
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        assert!(
            name.starts_with("birdnet-geomodel-v3.0.2.onnx."),
            "part file must sit beside the destination, got {name}"
        );
        assert!(
            name.ends_with(".part"),
            "part file must carry the partial suffix, got {name}"
        );
        assert!(
            name.contains(&std::process::id().to_string()),
            "part file must be qualified by pid so concurrent writers cannot collide, got {name}"
        );
        assert_eq!(p.parent(), Some(Path::new("/models")));
    }

    #[test]
    fn test_part_path_preserves_multi_dot_filenames() {
        // with_extension would mangle "v3.0.2.onnx" into "v3.0.2.part".
        let p = part_path(Path::new("/models/birdnet-geomodel-v3.0.2.onnx")).unwrap();
        assert!(p.to_string_lossy().contains("v3.0.2.onnx."));
    }

    #[test]
    fn test_part_path_differs_from_the_plain_suffix_form() {
        // A shared "<dest>.part" let two processes interleave writes into one
        // file and let one process's error cleanup unlink the other's transfer.
        let dest = Path::new("/models/m.onnx");
        let p = part_path(dest).unwrap();

        assert_ne!(
            p,
            dest.with_file_name("m.onnx.part"),
            "the part name must be process-qualified, not a fixed suffix"
        );
    }

    #[test]
    fn test_part_path_rejects_a_destination_without_a_file_name() {
        assert!(part_path(Path::new("/models/..")).is_err());
    }

    #[test]
    fn test_finalize_download_renames_and_clears_part() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("m.onnx");
        let part = part_path(&dest).unwrap();
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
        let part = part_path(&dest).unwrap();
        std::fs::write(&part, b"fresh").unwrap();

        finalize_download(&part, &dest).unwrap();

        assert_eq!(std::fs::read(&dest).unwrap(), b"fresh");
    }

    #[test]
    fn test_finalize_download_errors_without_a_part_file() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("m.onnx");

        assert!(finalize_download(&part_path(&dest).unwrap(), &dest).is_err());
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
