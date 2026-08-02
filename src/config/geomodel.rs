//! Resolving and acquiring the `BirdNET` Geomodel v3.0.2 range filter.
//!
//! The geomodel is a shared asset rather than part of any classifier, so it can
//! be absent when a user upgrades from a birda that used the `BirdNET` v2.4
//! meta model. This module finds it, and offers to fetch it when it is missing.

use crate::config::types::Config;
use crate::error::{Error, Result};
use crate::registry::{InstalledRangeFilter, Registry};
use std::io::{IsTerminal, Write};
use std::path::Path;
use tracing::warn;

/// What geomodel resolution needs from the invoking command.
///
/// This exists so the resolver takes exactly what it reads. It previously took
/// `&AnalyzeArgs` while reading three of its fields, which let `birda species`
/// build a throwaway `AnalyzeArgs` that set two fields the resolver never reads
/// and left unset the three it does. That compiled, and shipped as #286.
///
/// Deliberately does NOT derive `Default`. `..Default::default()` is exactly
/// what let the under-specified call typecheck; requiring every field to be
/// named is the point of this type.
#[derive(Debug, Clone, Copy)]
pub struct GeomodelRequest<'a> {
    /// `--geomodel-path`, if given.
    pub model_path: Option<&'a Path>,
    /// `--geomodel-labels-path`, if given.
    pub labels_path: Option<&'a Path>,
    /// `--yes`: download without prompting.
    pub assume_yes: bool,
}

/// Outcome of looking for the geomodel.
#[derive(Debug, Clone)]
pub enum GeomodelResolution {
    /// The geomodel is available at these paths.
    Ready(InstalledRangeFilter),
    /// The geomodel is unavailable; range filtering must be skipped.
    ///
    /// Carries a message explaining what the user can do about it.
    Unavailable(String),
}

/// Paths explicitly configured by the user, if any.
///
/// CLI flags win over the config file. Both paths must be supplied together:
/// pairing a new model with the previous version's labels would fail the label
/// count check with a confusing message, so it is rejected up front.
pub fn configured_paths(
    request: GeomodelRequest<'_>,
    config: &Config,
) -> Result<Option<InstalledRangeFilter>> {
    match (request.model_path, request.labels_path) {
        (Some(model), Some(labels)) => {
            return Ok(Some(InstalledRangeFilter {
                model: model.to_path_buf(),
                labels: labels.to_path_buf(),
            }));
        }
        (Some(_), None) => {
            return Err(Error::GeomodelPathsIncomplete {
                given: "--geomodel-path".to_string(),
            });
        }
        (None, Some(_)) => {
            return Err(Error::GeomodelPathsIncomplete {
                given: "--geomodel-labels-path".to_string(),
            });
        }
        (None, None) => {}
    }

    match (
        config.defaults.geomodel.as_ref(),
        config.defaults.geomodel_labels.as_ref(),
    ) {
        (Some(model), Some(labels)) => Ok(Some(InstalledRangeFilter {
            model: model.clone(),
            labels: labels.clone(),
        })),
        (Some(_), None) => Err(Error::GeomodelPathsIncomplete {
            given: "defaults.geomodel".to_string(),
        }),
        (None, Some(_)) => Err(Error::GeomodelPathsIncomplete {
            given: "defaults.geomodel_labels".to_string(),
        }),
        (None, None) => Ok(None),
    }
}

/// Locate the geomodel, offering to download it when it is missing.
///
/// Resolution order is CLI flags, then the config file, then the path the
/// registry says the asset installs to.
pub fn resolve_geomodel(
    request: GeomodelRequest<'_>,
    config: &Config,
    registry: &Registry,
    output_mode: crate::config::OutputMode,
) -> Result<GeomodelResolution> {
    let asset = registry
        .range_filter
        .as_ref()
        .ok_or(Error::RangeFilterAssetMissing)?;

    let registry_paths = crate::registry::geomodel_paths(asset)?;
    let paths = configured_paths(request, config)?.unwrap_or_else(|| registry_paths.clone());

    // "Ours to verify" is decided by which file this is, not by how the path
    // was reached: `birda models install geomodel` records its own install path
    // in config, so keying on "did it come from config" would exempt exactly
    // the copy birda manages, which is the copy that needs checking.
    let birda_managed = paths == registry_paths;

    if paths.is_installed() {
        // Presence is not enough for the copy birda manages: a download
        // interrupted by an older version, a half-copied models directory, or
        // two processes racing all leave a file that exists but is corrupt.
        // Without this check the bad file is loaded on every later run and
        // never re-verified, because only the install path verifies.
        //
        // A path pointing somewhere else is taken on trust: it may legitimately
        // be a different build of the geomodel, and its checksum is not ours to
        // police. The label-count check still guards the shape.
        if !birda_managed {
            return Ok(GeomodelResolution::Ready(paths));
        }

        match paths.verify(asset) {
            Ok(()) => return Ok(GeomodelResolution::Ready(paths)),
            // A genuine mismatch: the cached bytes are wrong, re-download them.
            Err(e) if crate::update::checksum::is_checksum_mismatch(&e) => {
                warn!(
                    "Installed {} failed checksum verification and will be downloaded again",
                    asset.name
                );
            }
            // A read error (permission/IO) is not a checksum failure. Naming it
            // one is wrong, and re-downloading a file that is fine will not fix
            // a failing disk. Surface the real error instead.
            Err(e) => return Err(e),
        }
    }

    // A configured path pointing outside the models directory is a
    // configuration error when it is missing, not something to paper over by
    // downloading to a different location and rewriting the user's setting.
    if !birda_managed {
        return Ok(GeomodelResolution::Unavailable(format!(
            "configured geomodel path {} does not exist; correct defaults.geomodel \
             or run 'birda models install {}'",
            paths.model.display(),
            crate::registry::GEOMODEL_INSTALL_ID
        )));
    }

    let interactive = std::io::stdin().is_terminal() && !output_mode.is_structured();
    acquire(asset, interactive, request.assume_yes)
}

/// Acquire a geomodel that is not on disk.
///
/// A missing geomodel is never a hard error. birda enables range filtering
/// implicitly whenever coordinates are configured, so failing here would break
/// every automated pipeline the moment it upgrades. Non-interactive runs warn
/// and continue unfiltered; `--yes` is how such a run opts into the download.
fn acquire(
    asset: &crate::registry::RangeFilterAsset,
    interactive: bool,
    assume_yes: bool,
) -> Result<GeomodelResolution> {
    let install_hint = format!(
        "run 'birda models install {}' to enable range filtering",
        crate::registry::GEOMODEL_INSTALL_ID
    );

    if !assume_yes {
        if !interactive {
            return Ok(GeomodelResolution::Unavailable(format!(
                "{} is not installed; {install_hint}",
                asset.name
            )));
        }

        if !prompt_for_download(asset)? {
            return Ok(GeomodelResolution::Unavailable(format!(
                "download declined; {install_hint}"
            )));
        }
    }

    let runtime = tokio::runtime::Runtime::new().map_err(|e| Error::Internal {
        message: format!("Failed to create async runtime: {e}"),
    })?;

    // Distinguish "offline" from "the download failed" before spending five
    // minutes in a request timeout.
    runtime.block_on(check_connectivity(&asset.model.url))?;

    let installed = runtime.block_on(crate::registry::install_range_filter(asset))?;

    // Deliberately NOT written back to config.toml here. The destructive half
    // of this reasoning is gone: save_config now writes to a temporary beside
    // the target and renames over it, so an interrupted write can no longer
    // leave an empty config that parses as all-defaults and silently loses
    // every model the user had configured (#307).
    //
    // What remains is enough to keep the decision. save_config re-serialises
    // the whole file from the struct, so it still drops comments and any
    // unrecognised keys, and an analyze run rewriting the user's config as a
    // side effect is surprising in its own right. The paths resolve from the
    // registry on the next run anyway, and 'birda models install geomodel'
    // records them explicitly.
    Ok(GeomodelResolution::Ready(installed))
}

/// Ask the user whether to download the geomodel.
#[allow(clippy::print_stderr)]
fn prompt_for_download(asset: &crate::registry::RangeFilterAsset) -> Result<bool> {
    // Written to stderr, not stdout. Interactivity is decided by stdin being a
    // terminal, so `birda ... > results.txt` from a terminal still prompts; on
    // stdout the prompt would land in the redirected file and the user would
    // see an unexplained hang on the read below.
    eprintln!(
        "Range filtering needs the {}, which is not installed.",
        asset.name
    );
    eprintln!(
        "  Model: {}    Labels: {}    Licence: {}",
        human_size(asset.model.size_bytes),
        human_size(asset.labels.size_bytes),
        asset.license.r#type
    );
    eprint!("Download it now? [y/N]: ");
    std::io::stderr().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    Ok(matches!(input.trim().to_lowercase().as_str(), "y" | "yes"))
}

/// Probe the download host so an offline machine gets a clear message.
///
/// Without this, being offline surfaces as a raw reqwest error after the full
/// request timeout, which reads like a birda bug rather than a missing network.
async fn check_connectivity(url: &str) -> Result<()> {
    let host = host_of(url);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(
            crate::constants::download::CONNECTIVITY_PROBE_TIMEOUT_SECS,
        ))
        .build()
        .map_err(|e| Error::Internal {
            message: format!("Failed to create HTTP client: {e}"),
        })?;

    client
        .head(url)
        .send()
        .await
        .map_err(|_| Error::NoNetworkConnectivity { host })?;

    Ok(())
}

/// Host portion of a URL, for error messages.
fn host_of(url: &str) -> String {
    url.split("://")
        .nth(1)
        .and_then(|rest| rest.split('/').next())
        .unwrap_or(url)
        .to_string()
}

/// Render a file size for the download prompt.
///
/// Returns "unknown size" when the registry does not declare one, rather than
/// printing a misleading zero.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn human_size(bytes: Option<u64>) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;

    match bytes {
        None => "unknown size".to_string(),
        Some(bytes) if (bytes as f64) >= MIB => format!("{:.1} MB", bytes as f64 / MIB),
        Some(bytes) => format!("{:.0} KB", bytes as f64 / KIB),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Build a request the way a command line would.
    ///
    /// The explicit lifetime is required: with two borrowed parameters, elision
    /// cannot pick which one the returned `GeomodelRequest` borrows from.
    fn request_with_paths<'a>(
        model: Option<&'a str>,
        labels: Option<&'a str>,
    ) -> GeomodelRequest<'a> {
        GeomodelRequest {
            model_path: model.map(Path::new),
            labels_path: labels.map(Path::new),
            assume_yes: false,
        }
    }

    #[test]
    fn test_cli_paths_take_precedence_over_config() {
        let args = request_with_paths(Some("/cli/m.onnx"), Some("/cli/l.txt"));
        let mut config = Config::default();
        config.defaults.geomodel = Some(PathBuf::from("/cfg/m.onnx"));
        config.defaults.geomodel_labels = Some(PathBuf::from("/cfg/l.txt"));

        let paths = configured_paths(args, &config).unwrap().unwrap();

        assert_eq!(paths.model, PathBuf::from("/cli/m.onnx"));
        assert_eq!(paths.labels, PathBuf::from("/cli/l.txt"));
    }

    #[test]
    fn test_config_paths_used_when_cli_absent() {
        let args = request_with_paths(None, None);
        let mut config = Config::default();
        config.defaults.geomodel = Some(PathBuf::from("/cfg/m.onnx"));
        config.defaults.geomodel_labels = Some(PathBuf::from("/cfg/l.txt"));

        let paths = configured_paths(args, &config).unwrap().unwrap();

        assert_eq!(paths.model, PathBuf::from("/cfg/m.onnx"));
    }

    #[test]
    fn test_no_configured_paths_yields_none() {
        let paths = configured_paths(request_with_paths(None, None), &Config::default()).unwrap();

        assert!(paths.is_none(), "falls back to the registry install path");
    }

    #[test]
    fn test_partial_cli_paths_are_rejected() {
        // Pairing a new model with the previous version's labels would fail the
        // label count check with a confusing message, so reject it up front.
        let args = request_with_paths(Some("/cli/m.onnx"), None);

        let err = configured_paths(args, &Config::default()).unwrap_err();

        assert!(matches!(err, Error::GeomodelPathsIncomplete { .. }));
    }

    #[test]
    fn test_partial_cli_labels_path_is_rejected() {
        let args = request_with_paths(None, Some("/cli/l.txt"));

        let err = configured_paths(args, &Config::default()).unwrap_err();

        assert!(matches!(err, Error::GeomodelPathsIncomplete { .. }));
    }

    #[test]
    fn test_partial_config_paths_are_rejected() {
        let mut config = Config::default();
        config.defaults.geomodel = Some(PathBuf::from("/cfg/m.onnx"));

        let err = configured_paths(request_with_paths(None, None), &config).unwrap_err();

        assert!(matches!(err, Error::GeomodelPathsIncomplete { .. }));
    }

    #[test]
    fn test_non_interactive_missing_geomodel_is_unavailable_not_an_error() {
        // No on-disk setup: acquire() resolves the destination from the asset,
        // and test_asset()'s filenames are not installed in this environment.
        let resolution = acquire(&test_asset(), false, false).unwrap();

        match resolution {
            GeomodelResolution::Unavailable(message) => {
                assert!(
                    message.contains("birda models install geomodel"),
                    "the message must name the command that fixes it, got: {message}"
                );
            }
            GeomodelResolution::Ready(_) => panic!("must not report ready"),
        }
    }

    #[test]
    fn test_human_size_renders_megabytes_and_kilobytes() {
        assert_eq!(human_size(Some(14_700_313)), "14.0 MB");
        assert_eq!(human_size(Some(479_350)), "468 KB");
    }

    #[test]
    fn test_human_size_reports_unknown_rather_than_zero() {
        assert_eq!(human_size(None), "unknown size");
    }

    #[test]
    fn test_host_of_extracts_the_download_host() {
        assert_eq!(
            host_of("https://huggingface.co/tphakala/BirdNET-Geomodel/resolve/main/m.onnx"),
            "huggingface.co"
        );
    }

    #[test]
    fn test_host_of_falls_back_to_the_whole_string() {
        assert_eq!(host_of("not-a-url"), "not-a-url");
    }

    fn test_asset() -> crate::registry::RangeFilterAsset {
        crate::registry::RangeFilterAsset {
            id: "birdnet-geomodel-v3".into(),
            name: "BirdNET Geomodel v3.0.2".into(),
            version: "3.0.2".into(),
            vendor: "Cornell Lab".into(),
            license: crate::registry::LicenseInfo {
                r#type: "CC-BY-SA-4.0".into(),
                url: "https://creativecommons.org/licenses/by-sa/4.0/".into(),
                commercial_use: true,
                attribution_required: true,
                share_alike: true,
            },
            species_count: 12012,
            model: crate::registry::FileInfo {
                url: "https://example.invalid/m.onnx".into(),
                filename: "birdnet-geomodel-v3.0.2.onnx".into(),
                sha256: None,
                size_bytes: Some(14_700_313),
            },
            labels: crate::registry::FileInfo {
                url: "https://example.invalid/l.txt".into(),
                filename: "birdnet-geomodel-v3.0.2-labels.txt".into(),
                sha256: None,
                size_bytes: Some(479_350),
            },
        }
    }
}
