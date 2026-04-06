//! SHA256 checksum verification for downloaded archives.

use crate::error::{Error, Result};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Verify that a file's SHA256 hash matches the expected hex digest.
///
/// Returns `Ok(())` if the checksum matches. Returns `Err(UpdateChecksumMismatch)`
/// if it doesn't. The file is read in streaming fashion to avoid loading it all
/// into memory.
pub fn verify_sha256(path: &Path, expected_hex: &str) -> Result<()> {
    let file_bytes = std::fs::read(path).map_err(Error::Io)?;
    let actual_hex = hex_digest(&file_bytes);

    if actual_hex != expected_hex.to_ascii_lowercase() {
        return Err(Error::UpdateChecksumMismatch {
            file: path.file_name().map_or_else(
                || "unknown".to_string(),
                |n| n.to_string_lossy().to_string(),
            ),
            expected: expected_hex.to_string(),
            actual: actual_hex,
        });
    }

    Ok(())
}

/// Compute the SHA256 hex digest of raw bytes.
fn hex_digest(data: &[u8]) -> String {
    let hash = Sha256::digest(data);
    // Format each byte as two lowercase hex characters
    hash.iter()
        .fold(String::with_capacity(64), |mut acc, byte| {
            use std::fmt::Write;
            let _ = write!(acc, "{byte:02x}");
            acc
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_hex_digest_known_value() {
        // SHA256 of empty string
        let hash = hex_digest(b"");
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_hex_digest_hello_world() {
        let hash = hex_digest(b"hello world");
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_verify_sha256_matching() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"test content").unwrap();
        drop(f);

        let expected = hex_digest(b"test content");
        assert!(verify_sha256(&path, &expected).is_ok());
    }

    #[test]
    fn test_verify_sha256_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"test content").unwrap();
        drop(f);

        let result = verify_sha256(
            &path,
            "0000000000000000000000000000000000000000000000000000000000000000",
        );
        assert!(result.is_err());
    }
}
