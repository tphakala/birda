//! Release manifest fetching and parsing.

use crate::error::{Error, Result};
use serde::Deserialize;

/// A binary asset entry in the release manifest.
#[derive(Debug, Deserialize)]
pub struct BinAsset {
    /// Filename of the archive (e.g., `birda-linux-x64-bin-v1.9.0.tar.gz`).
    pub file: String,
    /// SHA256 hex digest of the archive.
    pub sha256: String,
}

/// Dependencies section of the release manifest.
#[derive(Debug, Deserialize)]
pub struct Dependencies {
    /// Required ONNX Runtime version.
    pub onnxruntime: String,
}

/// CUDA-specific version requirements.
#[derive(Debug, Deserialize)]
pub struct CudaVersions {
    /// Required CUDA toolkit version.
    pub cuda_toolkit: String,
    /// Required cuDNN version.
    pub cudnn: String,
}

/// Asset collections in the manifest.
#[derive(Debug, Deserialize)]
pub struct Assets {
    /// Binary-only archives keyed by platform (e.g., "linux-x64", "linux-x64-cuda").
    pub bin: std::collections::HashMap<String, BinAsset>,
}

/// Release manifest fetched from GitHub.
#[derive(Debug, Deserialize)]
pub struct Manifest {
    /// Release version (semver, e.g., "1.9.0").
    pub version: String,
    /// Available assets.
    pub assets: Assets,
    /// Library dependency versions.
    pub dependencies: Dependencies,
    /// CUDA-specific versions (absent for CPU-only manifests).
    pub cuda: Option<CudaVersions>,
}

impl Manifest {
    /// Parse a manifest from JSON bytes.
    pub fn from_json(json: &[u8]) -> Result<Self> {
        serde_json::from_slice(json).map_err(|source| Error::UpdateManifestParse { source })
    }
}

/// Fetch the release manifest from GitHub.
///
/// Downloads `manifest.json` from the latest release using the direct
/// download URL (no GitHub API needed, avoids rate limits).
pub async fn fetch_manifest(client: &reqwest::Client) -> Result<Manifest> {
    let url = super::constants::RELEASE_DOWNLOAD_URL
        .replace("{repo}", super::constants::GITHUB_REPO)
        .replace("{file}", super::constants::MANIFEST_FILENAME);

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| Error::UpdateFetchFailed {
            reason: format!("HTTP request failed: {e}"),
        })?;

    if !response.status().is_success() {
        return Err(Error::UpdateFetchFailed {
            reason: format!("HTTP {}", response.status()),
        });
    }

    // Guard against oversized responses (manifest should be < 1 MiB)
    if let Some(len) = response.content_length()
        && len > super::constants::MANIFEST_MAX_BYTES
    {
        return Err(Error::UpdateFetchFailed {
            reason: format!("manifest too large: {len} bytes"),
        });
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| Error::UpdateFetchFailed {
            reason: format!("failed to read response body: {e}"),
        })?;

    // Also check actual body size (Content-Length can be omitted or spoofed)
    if bytes.len() as u64 > super::constants::MANIFEST_MAX_BYTES {
        return Err(Error::UpdateFetchFailed {
            reason: format!("manifest too large: {} bytes", bytes.len()),
        });
    }

    Manifest::from_json(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_manifest() {
        let json = br#"{
            "version": "1.9.0",
            "min_gui_version": "1.1.0",
            "assets": {
                "bin": {
                    "linux-x64": {
                        "file": "birda-linux-x64-bin-v1.9.0.tar.gz",
                        "sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
                    },
                    "linux-x64-cuda": {
                        "file": "birda-linux-x64-cuda-bin-v1.9.0.tar.gz",
                        "sha256": "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
                    }
                },
                "embed": {},
                "cuda_libs": {}
            },
            "dependencies": {
                "onnxruntime": "1.24.2"
            },
            "cuda": {
                "cuda_toolkit": "12.9",
                "cudnn": "9.17.1.4"
            }
        }"#;

        let manifest = Manifest::from_json(json).expect("test manifest should parse");
        assert_eq!(manifest.version, "1.9.0");
        assert_eq!(manifest.dependencies.onnxruntime, "1.24.2");
        let cuda = manifest.cuda.expect("cuda should be present");
        assert_eq!(cuda.cuda_toolkit, "12.9");
        assert_eq!(cuda.cudnn, "9.17.1.4");
        assert_eq!(manifest.assets.bin.len(), 2);

        let linux = &manifest.assets.bin["linux-x64"];
        assert_eq!(linux.file, "birda-linux-x64-bin-v1.9.0.tar.gz");
        assert_eq!(linux.sha256.len(), 64);
    }

    #[test]
    fn test_parse_invalid_json() {
        let result = Manifest::from_json(b"not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_missing_required_fields() {
        let json = br#"{"version": "1.0.0"}"#;
        let result = Manifest::from_json(json);
        assert!(result.is_err());
    }
}
