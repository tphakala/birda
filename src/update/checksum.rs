//! SHA256 checksum verification for downloaded archives.

use crate::error::{Error, Result};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Verify that a file's SHA256 hash matches the expected hex digest.
///
/// Returns `Ok(())` if the checksum matches. Returns `Err(UpdateChecksumMismatch)`
/// if it doesn't.
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

/// Report whether a verification error is a genuine content mismatch.
///
/// `verify_sha256` fails two ways that must not be treated alike. A
/// [`Error::UpdateChecksumMismatch`] proves the bytes on disk are wrong, so a
/// caller may safely delete them and download again. A [`Error::Io`] means the
/// file could not even be read (EACCES, EIO on a failing disk or SD card): that
/// is not proof the bytes are bad, and deleting a possibly-correct model to
/// re-download hundreds of MB is destructive and, on failing hardware, loops.
///
/// Call sites that react to a failed verification by removing files or
/// re-downloading must gate that action on this returning `true`.
pub fn is_checksum_mismatch(err: &Error) -> bool {
    matches!(err, Error::UpdateChecksumMismatch { .. })
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

    /// A hash no real content produces, used to force a deterministic mismatch.
    const ALL_ZERO_SHA256: &str =
        "0000000000000000000000000000000000000000000000000000000000000000";

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
        let dir = tempfile::tempdir().expect("test setup failed");
        let path = dir.path().join("test.bin");
        let mut f = std::fs::File::create(&path).expect("test setup failed");
        f.write_all(b"test content").expect("test setup failed");
        drop(f);

        let expected = hex_digest(b"test content");
        assert!(verify_sha256(&path, &expected).is_ok());
    }

    #[test]
    fn test_verify_sha256_mismatch() {
        let dir = tempfile::tempdir().expect("test setup failed");
        let path = dir.path().join("test.bin");
        let mut f = std::fs::File::create(&path).expect("test setup failed");
        f.write_all(b"test content").expect("test setup failed");
        drop(f);

        let result = verify_sha256(&path, ALL_ZERO_SHA256);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_sha256_read_error_is_io_not_mismatch() {
        // A directory cannot be read as a file: `std::fs::read` returns EISDIR.
        // This stands in for the EACCES/EIO read failures on failing hardware,
        // and locks the contract that a read failure surfaces as `Error::Io`,
        // never as a spurious checksum mismatch.
        let dir = tempfile::tempdir().expect("test setup failed");

        let result = verify_sha256(dir.path(), ALL_ZERO_SHA256);
        assert!(matches!(result, Err(Error::Io(_))));
    }

    #[test]
    fn test_is_checksum_mismatch_true_for_mismatch() {
        let dir = tempfile::tempdir().expect("test setup failed");
        let path = dir.path().join("test.bin");
        let mut f = std::fs::File::create(&path).expect("test setup failed");
        f.write_all(b"test content").expect("test setup failed");
        drop(f);

        let err =
            verify_sha256(&path, ALL_ZERO_SHA256).expect_err("mismatched checksum should fail");
        assert!(is_checksum_mismatch(&err));
    }

    #[test]
    fn test_is_checksum_mismatch_false_for_read_error() {
        // The destructive delete-and-redownload must not fire on a read error.
        let dir = tempfile::tempdir().expect("test setup failed");

        let err = verify_sha256(dir.path(), ALL_ZERO_SHA256)
            .expect_err("reading a directory as a file should fail");
        assert!(!is_checksum_mismatch(&err));
    }
}
