//! Model download and installation logic.

use super::types::{ModelEntry, ModelVariant, RangeFilterAsset};
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
    /// What was installed, for variant-based entries.
    pub provenance: Option<InstallProvenance>,
}

/// What was installed, recorded in `config.toml`.
///
/// Two jobs: telling the user an update is available, and knowing exactly which
/// files an install owns so an upgrade can delete the ones it replaces. Model
/// filenames are immutable by the publishing policy, so without this a
/// preview-to-GA upgrade would leave the old files on disk forever, at roughly
/// 150 MB per regional slice and 557 MB for a global fp32.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallProvenance {
    /// Registry id the files came from.
    pub registry_id: String,
    /// Exact upstream version installed.
    pub version: String,
    /// Our conversion revision.
    pub build: Option<u32>,
    /// Region slug, `None` for the global model.
    pub region: Option<String>,
    /// Variant id.
    pub variant: Option<String>,
}

impl InstallProvenance {
    /// Key this install occupies in `config.models`.
    ///
    /// Regional installs get an `<id>-<region>` key so a global and a regional
    /// model coexist and both stay reachable with `-m`. Derived, never
    /// user-supplied, so it cannot collide with a registry id.
    #[must_use]
    pub fn config_key(&self) -> String {
        self.region.as_ref().map_or_else(
            || self.registry_id.clone(),
            |region| format!("{}-{region}", self.registry_id),
        )
    }
}

/// Rewrite a Hugging Face URL to the mirror named by `HF_ENDPOINT`.
///
/// Users on networks that block huggingface.co set `HF_ENDPOINT` to a mirror.
/// Applied at request time rather than baked into `registry.json`, so it also
/// covers entries whose URLs were pinned before mirrors were supported, and so
/// changing the mirror needs no registry rewrite.
#[must_use]
pub fn resolve_url(url: &str) -> String {
    let Ok(endpoint) = std::env::var(crate::constants::download::HF_ENDPOINT_ENV) else {
        return url.to_string();
    };

    // An exported-but-empty HF_ENDPOINT is a common shell accident. Treating it
    // as an endpoint would rewrite every URL into a relative path.
    let endpoint = endpoint.trim().trim_end_matches('/');
    if endpoint.is_empty() {
        return url.to_string();
    }

    url.strip_prefix(crate::constants::download::HUGGING_FACE_ENDPOINT)
        .map_or_else(|| url.to_string(), |rest| format!("{endpoint}{rest}"))
}

/// Download a file with progress bar.
pub async fn download_file(client: &Client, url: &str, dest: &Path) -> Result<()> {
    download_verified(client, url, dest, None).await
}

/// Download a file and check it against a checksum before it replaces `dest`.
///
/// The check happens on the part file, while the destination is still whatever
/// was there before. Verifying after the rename would mean a corrupt download
/// had already replaced a good file, and deleting it then would take the good
/// file with it: a failed reinstall would destroy the working install it was
/// meant to upgrade, and every other config entry naming that file with it.
///
/// A file whose registry entry declares no checksum is accepted, matching
/// [`InstalledRangeFilter::verify`]. The registry is the authority on whether a
/// checksum exists, and refusing to install without one would break every entry
/// that predates checksums.
pub async fn download_verified(
    client: &Client,
    url: &str,
    dest: &Path,
    expected_sha256: Option<&str>,
) -> Result<()> {
    // Resolved once, and every message below reports the resolved URL, so a
    // mirror failure names the host actually contacted rather than the one the
    // registry happens to record.
    let resolved = resolve_url(url);
    let url: &str = &resolved;

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

    // Before the rename, so a bad download is discarded rather than published
    // over a file that was fine.
    if let Some(sum) = expected_sha256
        && let Err(e) = crate::update::checksum::verify_sha256(&part, sum)
    {
        drop(tokio::fs::remove_file(&part).await);
        return Err(e);
    }

    // Run the publish on a blocking thread: its Windows sharing-violation backoff
    // sleeps, which must not park this async worker. `part` is owned and unused
    // afterwards, so it moves in; `dest` is cloned for the 'static closure.
    let dest_owned = dest.to_path_buf();
    tokio::task::spawn_blocking(move || finalize_download(&part, &dest_owned))
        .await
        .map_err(|e| Error::Internal {
            message: format!("download finalization task failed to run: {e}"),
        })??;
    pb.finish_with_message("Download complete");

    Ok(())
}

/// Path of the in-progress download file for a destination.
///
/// The name is qualified with this process's id so two concurrent birda
/// processes on the same host downloading the same destination cannot write into
/// one file: a shared part name lets their writes interleave, and lets one
/// process's error cleanup unlink the other's in-progress transfer.
///
/// This holds within one pid namespace, not globally. Two birda containers that
/// bind-mount the same models directory each see their own low pids, so their
/// part names can still collide; kernel-unique naming (as in
/// `crate::utils::fs::new_temp_in`) is the fix if that case ever needs to be
/// safe. The pid form is kept because it lets `find_stale_part_files` liveness-
/// check a leftover's writer.
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
///
/// A failed publish also removes the part file, best effort via [`roll_back`], so
/// the unlink failing is reported rather than guaranteed away. The part file
/// carries no value on its own: nothing resumes from it, and `part_path` names it
/// after this process, so in practice a retry is a new invocation that picks a
/// different name and leaves the old one stranded for good.
///
/// What the directory fsync buys here, stated precisely because it is easy to
/// overclaim: it stops a crash costing the user the download again. `stream_to_file`
/// has already `sync_all`ed the part file, so the data behind the new name is
/// durable when the name appears; without the directory fsync a crash can lose only
/// the name, `is_installed` then correctly reports the model missing (it stats the
/// file), and the next run re-downloads it. Wasted bandwidth on a large model, never
/// a corrupt install.
///
/// The order matters and is the reason this is not the same as the self-update
/// swap: fsync the FILE first, then rename, then fsync the directory. A directory
/// fsync over a file that was never flushed publishes a durable name over
/// non-durable data, which is worse than losing the rename.
fn finalize_download(part: &Path, dest: &Path) -> Result<()> {
    // Named rather than a bare `Error::Io`, which renders as "I/O error: Device or
    // resource busy (os error 16)" with neither path in it. That EBUSY is a real
    // case: a destination bind-mounted as a file cannot be renamed over, and it is
    // the same failure this change documents for the registry one directory away.
    // Retried, not a single rename: on Windows a concurrent reader holding the
    // destination open without FILE_SHARE_DELETE makes MoveFileExW fail with a
    // transient sharing violation. The backoff rides that out; `roll_back` below
    // fires only once the retries are exhausted, so the `.part` file survives to
    // be renamed on a later attempt. On non-Windows this is a single rename.
    let published = {
        let mut backoff = crate::constants::publish::BASE_BACKOFF;
        let mut attempt = 1u32;
        loop {
            match std::fs::rename(part, dest) {
                Ok(()) => break Ok(()),
                Err(source) => {
                    if crate::utils::fs::backoff_after_transient_publish(
                        &source,
                        attempt,
                        &mut backoff,
                    ) {
                        attempt += 1;
                        continue;
                    }
                    break Err(source);
                }
            }
        }
    };
    if let Err(source) = published {
        // The part file is useless without the rename, and nothing sweeps it up:
        // `find_obsolete_files` matches a fixed list of names that does not include
        // it, and `part_path` qualifies the name with this process's id, so the
        // realistic retry (a second invocation, with a different pid) never
        // reclaims it either. Within one process the name is stable and
        // `stream_to_file` would truncate it, so the leak is across runs.
        //
        // Through `roll_back` rather than a bare `drop(remove_file(..))`, which is
        // what the two earlier failure paths in `download_verified` use: it
        // tolerates an already-absent file and WARNS with the path otherwise. The
        // causes are correlated (whatever broke the rename in this directory,
        // EACCES or a read-only remount, tends to break the unlink too), so the
        // one case where the cleanup silently fails to happen is the one where the
        // user most needs to be told which file to go and delete.
        roll_back(&[part]);
        return Err(Error::DownloadInstallFailed {
            dest: dest.to_path_buf(),
            source,
        });
    }
    crate::utils::fs::sync_parent_directory(dest);
    Ok(())
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
    if paths.is_installed() {
        match paths.verify(asset) {
            Ok(()) => return Ok(paths),
            // A read error on an installed file is not proof it is wrong, and
            // re-downloading hundreds of MB will not fix a failing disk. Surface
            // it, mirroring `resolve_geomodel`, rather than silently redownload
            // a copy that is fine. Only a genuine mismatch falls through.
            Err(e) if !crate::update::checksum::is_checksum_mismatch(&e) => return Err(e),
            Err(_) => {}
        }
    }

    let dir = models_dir()?;
    std::fs::create_dir_all(&dir).map_err(Error::Io)?;
    let client = http_client()?;

    download_file(&client, &asset.model.url, &paths.model).await?;
    download_file(&client, &asset.labels.url, &paths.labels).await?;

    verify_or_remove(&paths, asset)?;

    Ok(paths)
}

/// Verify freshly-downloaded range-filter files, removing them only on a
/// genuine checksum mismatch.
///
/// Consumers gate on presence, so a corrupt file left behind here would be
/// loaded on every later run without ever being re-verified; a genuine
/// [`Error::UpdateChecksumMismatch`] therefore removes both files so the caller
/// downloads a fresh copy. A read error (EACCES/EIO on a failing disk) is not
/// proof the bytes are wrong: removing a possibly-correct model to force a
/// re-download is destructive and loops on failing hardware, so on that error
/// the files are left in place and the error surfaced.
fn verify_or_remove(paths: &InstalledRangeFilter, asset: &RangeFilterAsset) -> Result<()> {
    if let Err(e) = paths.verify(asset) {
        if crate::update::checksum::is_checksum_mismatch(&e) {
            drop(std::fs::remove_file(&paths.model));
            drop(std::fs::remove_file(&paths.labels));
        }
        return Err(e);
    }
    Ok(())
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

/// The pid embedded in a `<name>.<pid>.part` path, if it has that shape.
///
/// Returns the parsed pid so a caller can both recognise a part download and
/// check whether the writing process is still alive. Requiring a numeric pid
/// segment keeps an unrelated `.part` file a user left in the models directory
/// from being misreported as an interrupted download.
fn part_download_pid(path: &Path) -> Option<u32> {
    if !path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case(crate::constants::download::PARTIAL_SUFFIX))
    {
        return None;
    }
    // With the `.part` extension removed the stem is `<name>.<pid>`; the segment
    // after its last '.' must be the numeric pid.
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| stem.rsplit('.').next())
        .filter(|pid| !pid.is_empty() && pid.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|pid| pid.parse::<u32>().ok())
}

/// Whether a process with `pid` is running on this host.
///
/// Best effort, used only to keep an in-progress download from being reported as
/// a leftover. Determinable on Linux via `/proc/<pid>`, which is where the
/// container-sharing case in [`part_path`] arises; other platforms return `false`
/// (not known to be running), so a leftover is reported as before rather than
/// hidden.
///
/// The signal is inexact and only ever narrows a report-only advisory, never
/// deletes: a pid can be reused by an unrelated live process, a pid from another
/// container's namespace can coincidentally match, and a `/proc` mounted
/// `hidepid=1`/`2` hides another user's process (so it is reported as a leftover,
/// the safe direction). The caller additionally excepts `CONTAINER_INIT_PID`;
/// see [`find_stale_part_files`].
#[cfg(target_os = "linux")]
fn pid_is_running(pid: u32) -> bool {
    Path::new("/proc").join(pid.to_string()).exists()
}

/// See the Linux definition. Off Linux, liveness is not checked here.
#[cfg(not(target_os = "linux"))]
const fn pid_is_running(_pid: u32) -> bool {
    false
}

/// Report leftover partial-download files in the models directory.
///
/// A download in progress is written to `<name>.<pid>.part` (see `part_path`)
/// and renamed onto its destination only on success. A process killed
/// mid-download leaves that part file behind: the Ctrl+C handler calls
/// `std::process::exit`, which runs no cleanup, and because the name is
/// pid-qualified nothing ever reuses or reclaims it. Model files run to hundreds
/// of MB, so `models check` reporting the directory "clean" while it holds an
/// abandoned download is a silent waste of disk.
///
/// This only reports; it never deletes, since another live birda may own a
/// different pid's part file mid-transfer. A missing directory yields an empty
/// list, matching [`find_obsolete_files`].
///
/// A part file whose writer pid is still running on this host is skipped (best
/// effort; see `pid_is_running`), so a concurrent birda's in-progress transfer
/// is not reported as abandoned. `CONTAINER_INIT_PID` is excepted: `/proc/1`
/// always exists and cannot attribute liveness to this file's writer, and a
/// `<name>.1.part` is the common crashed-container leftover worth reporting.
/// Only regular files are reported; a symlink named like a part file is neither
/// followed nor listed.
pub fn find_stale_part_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut found = Vec::new();

    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(found),
        Err(e) => return Err(Error::Io(e)),
    };

    for entry in entries {
        let entry = entry.map_err(Error::Io)?;
        let path = entry.path();
        // Gate every stat behind the name check, so only a matching name is ever
        // examined further.
        let Some(pid) = part_download_pid(&path) else {
            continue;
        };
        // DirEntry::file_type does not follow symlinks (like lstat) and on Linux is
        // usually served from the readdir d_type with no extra syscall. A symlink
        // named like a part file is therefore neither followed nor reported; only a
        // regular file is a leftover download.
        if !entry.file_type().is_ok_and(|ft| ft.is_file()) {
            continue;
        }
        // Skip a download still in progress by a live writer on this host, so a
        // concurrent birda's transfer is not reported as an abandoned leftover.
        // CONTAINER_INIT_PID is excepted; see its definition and this fn's doc.
        if pid != crate::constants::download::CONTAINER_INIT_PID && pid_is_running(pid) {
            continue;
        }
        found.push(path);
    }

    // read_dir order is unspecified, so sort for deterministic reporting.
    found.sort();
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

    // Legacy entries carry `files`; variant-based ones are installed through
    // `install_variant` instead. Reaching here without `files` means the caller
    // did not branch on `ModelEntry::is_variant_based`, which is a bug rather
    // than a user error.
    let files = model.files.as_ref().ok_or_else(|| Error::Internal {
        message: format!(
            "model '{}' publishes variants and must be installed with install_variant",
            model.id
        ),
    })?;

    // Determine which language to use as default
    let language_code = language.unwrap_or(&files.labels.default_language);

    // Validate the requested language exists before downloading anything
    let default_language_variant = files
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
    let model_dest = models_dir.join(&files.model.filename);
    download_file(&client, &files.model.url, &model_dest).await?;

    // Download ALL language label files
    for language_variant in &files.labels.languages {
        let labels_dest = models_dir.join(&language_variant.filename);
        download_file(&client, &language_variant.url, &labels_dest).await?;
    }

    // Set the default labels path to the requested/default language
    let labels_dest = models_dir.join(&default_language_variant.filename);

    // Download BSG calibration file if available
    let bsg_calibration_path = if let Some(cal_info) = &files.bsg_calibration {
        let cal_dest = models_dir.join(&cal_info.filename);
        download_file(&client, &cal_info.url, &cal_dest).await?;
        Some(cal_dest)
    } else {
        None
    };

    // Download BSG migration file if available
    let bsg_migration_path = if let Some(mig_info) = &files.bsg_migration {
        let mig_dest = models_dir.join(&mig_info.filename);
        download_file(&client, &mig_info.url, &mig_dest).await?;
        Some(mig_dest)
    } else {
        None
    };

    // Download BSG distribution maps file if available
    let bsg_maps_path = if let Some(maps_info) = &files.bsg_distribution_maps {
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
        // Legacy entries predate provenance. Their filenames are stable across
        // versions, so there is nothing for cleanup to reclaim either.
        provenance: None,
    })
}

/// Download one variant's model and labels.
///
/// A file is checked against its declared checksum before it replaces anything
/// at its destination, so a bad download is discarded rather than published.
/// Only the model file carries one today: the published manifests record a
/// checksum per model but reference their labels by path alone, so the labels
/// file is accepted on the transport's word. `download_verified` will start
/// checking it the moment the registry declares one.
///
/// If the labels step fails after the model file landed, the model file is
/// removed, but only when this install is what created it. The install never
/// reaches `config.toml`, so nothing would record that a half-installed 557 MB
/// file exists and cleanup could not reclaim it. Removing a file that was
/// already there would be the opposite error: published filenames are shared
/// between variants of the same region, so deleting one can break a working
/// install that this one never touched.
pub async fn install_variant(entry: &ModelEntry, variant: &ModelVariant) -> Result<InstalledModel> {
    let models_dir = models_dir()?;
    std::fs::create_dir_all(&models_dir).map_err(Error::Io)?;

    let client = http_client()?;

    let model_dest = models_dir.join(&variant.model.filename);
    let model_existed = model_dest.exists();
    download_verified(
        &client,
        &variant.model.url,
        &model_dest,
        variant.model.sha256.as_deref(),
    )
    .await?;

    let labels_dest = models_dir.join(&variant.labels.filename);
    if let Err(e) = download_verified(
        &client,
        &variant.labels.url,
        &labels_dest,
        variant.labels.sha256.as_deref(),
    )
    .await
    {
        if !model_existed {
            roll_back(&[&model_dest]);
        }
        return Err(e);
    }

    Ok(InstalledModel {
        model: model_dest,
        labels: labels_dest,
        bsg_calibration: None,
        bsg_migration: None,
        bsg_distribution_maps: None,
        provenance: Some(InstallProvenance {
            registry_id: entry.id.clone(),
            version: entry.version.clone(),
            build: entry.build,
            region: variant.region.clone(),
            variant: Some(variant.id.clone()),
        }),
    })
}

/// Best-effort removal of files from a failed install.
fn roll_back(paths: &[&Path]) {
    for path in paths {
        if let Err(e) = std::fs::remove_file(path)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                "Could not remove {} after a failed install: {e}",
                path.display()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::registry::types::{FileInfo, LicenseInfo};
    use serial_test::serial;

    #[test]
    fn test_install_provenance_config_key_appends_the_region() {
        let provenance = InstallProvenance {
            registry_id: "birdnet-v30".to_string(),
            version: "3.0-preview3.1".to_string(),
            build: Some(1),
            region: Some("nordic".to_string()),
            variant: Some("fp32".to_string()),
        };
        assert_eq!(provenance.config_key(), "birdnet-v30-nordic");
    }

    #[test]
    fn test_install_provenance_config_key_is_the_bare_id_for_a_global_install() {
        // A global and a regional install must not fight over one config key,
        // and the global one keeps the name a user would type.
        let provenance = InstallProvenance {
            registry_id: "birdnet-v30".to_string(),
            version: "3.0-preview3.1".to_string(),
            build: Some(1),
            region: None,
            variant: Some("fp16".to_string()),
        };
        assert_eq!(provenance.config_key(), "birdnet-v30");
    }

    #[test]
    fn test_a_bad_download_never_replaces_the_file_already_there() {
        // The failure this guards: a reinstall whose download is corrupt used
        // to overwrite the destination and then delete it on verification
        // failure, destroying the working install it was meant to upgrade,
        // plus every other config entry naming that same file.
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("m.onnx");
        std::fs::write(&dest, b"right").unwrap();

        // Stand in for the streamed download: a part file holding bad bytes.
        let part = part_path(&dest).unwrap();
        std::fs::write(&part, b"wrong").unwrap();

        let verdict = crate::update::checksum::verify_sha256(&part, RIGHT_SHA256);
        assert!(verdict.is_err(), "the part file must fail verification");

        // The destination is untouched because the rename never happened.
        assert_eq!(std::fs::read(&dest).unwrap(), b"right");
    }

    #[test]
    fn test_finalize_only_publishes_bytes_that_verified() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("m.onnx");
        std::fs::write(&dest, b"stale").unwrap();

        let part = part_path(&dest).unwrap();
        std::fs::write(&part, b"right").unwrap();

        crate::update::checksum::verify_sha256(&part, RIGHT_SHA256).unwrap();
        finalize_download(&part, &dest).unwrap();

        assert_eq!(std::fs::read(&dest).unwrap(), b"right");
        assert!(!part.exists(), "the part file must be consumed");
    }

    #[test]
    fn test_a_failed_publish_does_not_orphan_its_part_file() {
        // download_verified cleans up on its stream-failed and checksum-failed
        // paths and did not on this one, so a rename that failed left the part
        // file behind forever: part_path qualifies the name with this process's
        // id, so the next attempt picks a different name, and find_obsolete_files
        // matches a fixed list that does not include it. Model files are tens to
        // hundreds of MB, so the user silently loses that much disk per failed
        // attempt with no way to find it short of `find`.
        let dir = tempfile::tempdir().unwrap();

        // A directory cannot be renamed over by a file. The errno differs by
        // platform and this test deliberately does not assert it: POSIX mandates
        // EISDIR, while Windows `MoveFileExW` with MOVEFILE_REPLACE_EXISTING
        // refuses a directory destination with ERROR_ACCESS_DENIED (verified on
        // Windows 11, not inferred). Both are an Err, which is all this needs, so
        // the fixture is portable even though CI only ever runs it on Linux.
        //
        // It stands in for the real case, which is awkward to provoke here: the
        // EBUSY documented for a destination bind-mounted as a file. Not EXDEV,
        // which cannot occur on this path at all, since `part_path` builds the
        // part file with `with_file_name` and it is therefore always in the
        // destination's own directory.
        let dest = dir.path().join("m.onnx");
        std::fs::create_dir(&dest).unwrap();

        let part = part_path(&dest).unwrap();
        std::fs::write(&part, b"a hundred megabytes, pretend").unwrap();

        let err = finalize_download(&part, &dest).unwrap_err();

        // The payload is asserted, not just the variant. `finalize_download` has
        // one Err construction site, so `matches!` alone cannot fail, and the
        // reason this variant exists at all is to name the path a bare
        // `Error::Io` omitted: pointing it at the part file instead of the
        // destination would restore the unhelpful message it replaced.
        match &err {
            Error::DownloadInstallFailed { dest: named, .. } => assert_eq!(
                named, &dest,
                "the error must name the destination it failed to publish onto"
            ),
            other => panic!("the caller must be told the publish failed, got: {other}"),
        }
        assert!(
            !part.exists(),
            "a failed publish must consume its part file; nothing else ever \
             reclaims it"
        );
    }

    #[test]
    fn test_roll_back_tolerates_a_file_that_is_already_gone() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("never-existed.onnx");

        // A download that failed before creating its destination leaves nothing
        // to remove. That is the desired state, not an error to report.
        roll_back(&[&missing]);
    }

    #[test]
    fn test_roll_back_removes_a_file_this_install_created() {
        let dir = tempfile::tempdir().unwrap();
        let created = dir.path().join("m.onnx");
        std::fs::write(&created, b"right").unwrap();

        roll_back(&[&created]);

        assert!(!created.exists());
    }

    #[test]
    fn test_verify_or_remove_leaves_files_intact_on_a_read_error() {
        // The #348 data-loss regression guard: a read error (here EISDIR, a
        // portable stand-in for the EACCES/EIO seen on a failing disk) must not
        // be mistaken for a checksum mismatch and delete a possibly-correct
        // model. The model is a valid regular file and the labels are the
        // unreadable one, so the model assertion is load-bearing: reverting the
        // `is_checksum_mismatch` gate deletes the real model file and turns this
        // red, which the predicate-only tests do not catch.
        let dir = tempfile::tempdir().unwrap();

        // The model reads and verifies fine; the read error is on the labels.
        let model = dir.path().join("model.onnx");
        std::fs::write(&model, b"right").unwrap(); // matches RIGHT_SHA256
        // A directory cannot be read as a file, so `verify` fails with Error::Io.
        let labels = dir.path().join("labels");
        std::fs::create_dir(&labels).unwrap();
        let paths = InstalledRangeFilter { model, labels };

        let err = verify_or_remove(&paths, &test_asset()).expect_err("a read error must surface");
        assert!(
            !crate::update::checksum::is_checksum_mismatch(&err),
            "a read error is not a checksum mismatch"
        );
        assert!(
            paths.model.exists(),
            "the readable model must not be deleted"
        );
        assert!(paths.labels.exists(), "the labels must not be deleted");
    }

    #[test]
    fn test_verify_or_remove_deletes_both_files_on_a_mismatch() {
        // The complement: a genuine mismatch is the one case that legitimately
        // removes the files, so the caller can download a fresh copy.
        let dir = tempfile::tempdir().unwrap();

        let model = dir.path().join("model.onnx");
        std::fs::write(&model, b"wrong").unwrap(); // does not match RIGHT_SHA256
        let labels = dir.path().join("labels.txt");
        std::fs::write(&labels, b"right").unwrap();
        let paths = InstalledRangeFilter { model, labels };

        let err = verify_or_remove(&paths, &test_asset()).expect_err("a mismatch must surface");
        assert!(crate::update::checksum::is_checksum_mismatch(&err));
        assert!(!paths.model.exists(), "a corrupt model must be removed");
        assert!(!paths.labels.exists(), "its labels must be removed with it");
    }

    // HF_ENDPOINT is process-global, so these run serially against every other
    // test that reads the environment.

    #[test]
    #[serial]
    fn test_resolve_url_is_identity_without_an_endpoint_override() {
        temp_env::with_var_unset("HF_ENDPOINT", || {
            let url = "https://huggingface.co/tphakala/X/resolve/main/m.onnx";
            assert_eq!(resolve_url(url), url);
        });
    }

    #[test]
    #[serial]
    fn test_resolve_url_rewrites_the_hugging_face_prefix() {
        temp_env::with_var("HF_ENDPOINT", Some("https://hf-mirror.com"), || {
            assert_eq!(
                resolve_url("https://huggingface.co/tphakala/X/resolve/main/m.onnx"),
                "https://hf-mirror.com/tphakala/X/resolve/main/m.onnx"
            );
        });
    }

    #[test]
    #[serial]
    fn test_resolve_url_tolerates_a_trailing_slash_on_the_endpoint() {
        // Without trimming, the rewrite produces a double slash after the host,
        // which some mirrors serve and others 404.
        temp_env::with_var("HF_ENDPOINT", Some("https://hf-mirror.com/"), || {
            assert_eq!(
                resolve_url("https://huggingface.co/a/b"),
                "https://hf-mirror.com/a/b"
            );
        });
    }

    #[test]
    #[serial]
    fn test_resolve_url_leaves_non_hugging_face_urls_alone() {
        temp_env::with_var("HF_ENDPOINT", Some("https://hf-mirror.com"), || {
            let url = "https://zenodo.org/records/1/files/m.onnx";
            assert_eq!(resolve_url(url), url);
        });
    }

    #[test]
    #[serial]
    fn test_resolve_url_ignores_a_blank_endpoint() {
        // An exported-but-empty HF_ENDPOINT is a common shell accident.
        // Treating it as an endpoint would rewrite every URL to a bare path.
        temp_env::with_var("HF_ENDPOINT", Some("   "), || {
            let url = "https://huggingface.co/a/b";
            assert_eq!(resolve_url(url), url);
        });
    }

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
    fn test_find_stale_part_files_detects_a_leftover_download() {
        let dir = tempfile::tempdir().unwrap();
        // A completed model and its labels must be ignored.
        std::fs::write(dir.path().join("birdnet-v30.onnx"), b"x").unwrap();
        std::fs::write(dir.path().join("birdnet-v30-labels.txt"), b"x").unwrap();
        // A download interrupted mid-transfer leaves a pid-qualified part file.
        // u32::MAX is above any Linux pid_max, so the liveness check treats its
        // writer as dead and the file as a genuine leftover.
        let leftover = format!(
            "birdnet-v30.onnx.{}.{}",
            u32::MAX,
            crate::constants::download::PARTIAL_SUFFIX
        );
        std::fs::write(dir.path().join(&leftover), b"partial").unwrap();
        // A directory matching the <name>.<pid>.part shape must still be
        // excluded: only regular files are leftover downloads.
        std::fs::create_dir(dir.path().join("interrupted.99999.part")).unwrap();
        // A .part file with no numeric pid segment is not a birda download.
        std::fs::write(dir.path().join("manual-notes.part"), b"x").unwrap();

        let found = find_stale_part_files(dir.path()).unwrap();

        assert_eq!(found.len(), 1);
        assert!(found[0].ends_with(&leftover));
    }

    #[test]
    fn test_find_stale_part_files_returns_all_leftovers_sorted() {
        let dir = tempfile::tempdir().unwrap();
        let suffix = crate::constants::download::PARTIAL_SUFFIX;
        // Pids above any Linux pid_max, so both are treated as dead writers.
        let b = format!("b-model.onnx.{}.{suffix}", u32::MAX);
        let a = format!("a-model.onnx.{}.{suffix}", u32::MAX - 1);
        std::fs::write(dir.path().join(&b), b"x").unwrap();
        std::fs::write(dir.path().join(&a), b"x").unwrap();

        let found = find_stale_part_files(dir.path()).unwrap();

        assert_eq!(found.len(), 2);
        // read_dir order is unspecified; the result is sorted, so "a-" precedes "b-".
        assert!(found[0].ends_with(&a));
        assert!(found[1].ends_with(&b));
    }

    #[test]
    fn test_find_stale_part_files_reports_nothing_on_a_clean_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("birdnet-v30.onnx"), b"x").unwrap();

        assert!(find_stale_part_files(dir.path()).unwrap().is_empty());
    }

    #[test]
    fn test_find_stale_part_files_is_empty_for_a_missing_directory() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does-not-exist");

        assert!(find_stale_part_files(&missing).unwrap().is_empty());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_find_stale_part_files_skips_a_download_whose_writer_is_alive() {
        // A part file tagged with this test process's own pid: its writer is
        // provably alive, so it is a download in progress, not a leftover.
        let dir = tempfile::tempdir().unwrap();
        let suffix = crate::constants::download::PARTIAL_SUFFIX;
        let live = format!("birdnet-v30.onnx.{}.{suffix}", std::process::id());
        std::fs::write(dir.path().join(&live), b"partial").unwrap();

        assert!(
            find_stale_part_files(dir.path()).unwrap().is_empty(),
            "a part file whose writer pid is still running must not be reported as a leftover"
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_find_stale_part_files_ignores_a_symlink_named_like_a_download() {
        use std::os::unix::fs::symlink;

        // A symlink whose name matches the <name>.<pid>.part shape must not be
        // followed or reported: only a real regular file is a leftover download.
        let dir = tempfile::tempdir().unwrap();
        let suffix = crate::constants::download::PARTIAL_SUFFIX;
        let target = dir.path().join("real-file");
        std::fs::write(&target, b"x").unwrap();
        // u32::MAX so that even if the symlink were wrongly followed the liveness
        // check would not apply, isolating the regular-file check as the reason
        // it is excluded.
        let link = dir.path().join(format!("model.onnx.{}.{suffix}", u32::MAX));
        symlink(&target, &link).unwrap();

        assert!(
            find_stale_part_files(dir.path()).unwrap().is_empty(),
            "a symlink named like a part file must not be reported as a leftover"
        );
    }

    #[test]
    fn test_part_download_pid_parses_the_pid_and_rejects_non_downloads() {
        // A well-formed part name yields its pid, taken from the last dot-segment
        // of the multi-dot stem.
        assert_eq!(
            part_download_pid(std::path::Path::new("birdnet-v30.onnx.4321.part")),
            Some(4321)
        );
        // Not a `.part` file at all.
        assert_eq!(
            part_download_pid(std::path::Path::new("birdnet-v30.onnx")),
            None
        );
        // `.part` but no numeric pid segment.
        assert_eq!(
            part_download_pid(std::path::Path::new("manual-notes.part")),
            None
        );
        // `.part` with an empty pid segment.
        assert_eq!(
            part_download_pid(std::path::Path::new("model.onnx..part")),
            None
        );
        // A numeric segment that overflows u32 is not a real pid (std::process::id
        // is a u32), so it is not recognised as a birda download.
        assert_eq!(
            part_download_pid(std::path::Path::new("model.onnx.99999999999999999999.part")),
            None
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_find_stale_part_files_reports_a_pid_1_leftover() {
        // A `<name>.1.part` only arises when birda ran as a container entrypoint
        // (pid 1). /proc/1 always exists, so a naive liveness check would hide it,
        // but it is the common crashed-container leftover and must be reported.
        let dir = tempfile::tempdir().unwrap();
        let suffix = crate::constants::download::PARTIAL_SUFFIX;
        let leftover = format!("birdnet-v30.onnx.1.{suffix}");
        std::fs::write(dir.path().join(&leftover), b"partial").unwrap();

        let found = find_stale_part_files(dir.path()).unwrap();

        assert_eq!(
            found.len(),
            1,
            "a pid-1 (container-entrypoint) part file must be reported, not hidden by /proc/1"
        );
        assert!(found[0].ends_with(&leftover));
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
        assert_eq!(
            Path::new(&name).extension(),
            Some(std::ffi::OsStr::new("part")),
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
            provenance: None,
        };

        assert_eq!(
            installed.labels.to_string_lossy(),
            "/models/birdnet-v24-en.txt"
        );
    }
}
