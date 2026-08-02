//! SHA256 checksum verification for downloaded archives.

use crate::error::{Error, Result};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

/// Verify that a file's SHA256 hash matches the expected hex digest.
///
/// Returns `Ok(())` if the checksum matches, `Err(UpdateChecksumMismatch)` if it
/// does not, and `Err(Error::Io)` if the file cannot be read. Callers gate a
/// destructive re-download on [`is_checksum_mismatch`], so the read-error case
/// must stay distinct from a content mismatch.
pub fn verify_sha256(path: &Path, expected_hex: &str) -> Result<()> {
    let actual_hex = hash_file(path)?;

    // Case-insensitive compare, so an uppercase expected digest still matches
    // without allocating a lowercased copy of it.
    if !actual_hex.eq_ignore_ascii_case(expected_hex) {
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

/// Compute the SHA256 hex digest of a file, streaming it through a fixed-size
/// buffer so peak memory stays bounded to
/// [`HASH_CHUNK_BYTES`](super::constants::HASH_CHUNK_BYTES) regardless of the
/// file's length. Hashing a several-hundred-MB model therefore never
/// materialises the whole file in memory.
///
/// A read failure surfaces as [`Error::Io`], never as a spurious mismatch, which
/// the delete-and-redownload callers rely on (see [`is_checksum_mismatch`]).
fn hash_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path).map_err(Error::Io)?;
    let mut hasher = Sha256::new();
    // Heap-allocated so the buffer does not sit on the stack; a single 64 KiB
    // allocation per call is negligible next to the file read it serves.
    let mut buf = vec![0u8; super::constants::HASH_CHUNK_BYTES];
    loop {
        // Retry on EINTR, matching std::fs::read's read_to_end: a signal
        // interrupting the read must not abort verification.
        let read = match file.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(Error::Io(e)),
        };
        hasher.update(&buf[..read]);
    }
    Ok(format_hex(hasher.finalize().as_slice()))
}

/// Format a hash digest as a lowercase hex string.
fn format_hex(hash: &[u8]) -> String {
    hash.iter().fold(
        String::with_capacity(super::constants::SHA256_HEX_LEN),
        |mut acc, byte| {
            use std::fmt::Write;
            let _ = write!(acc, "{byte:02x}");
            acc
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A hash no real content produces, used to force a deterministic mismatch.
    const ALL_ZERO_SHA256: &str =
        "0000000000000000000000000000000000000000000000000000000000000000";

    /// Hash raw bytes to a lowercase hex SHA256, for computing expected values.
    fn sha256_hex(data: &[u8]) -> String {
        format_hex(Sha256::digest(data).as_slice())
    }

    #[test]
    fn test_sha256_hex_empty_string() {
        // SHA256 of the empty string
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_sha256_hex_hello_world() {
        assert_eq!(
            sha256_hex(b"hello world"),
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

        let expected = sha256_hex(b"test content");
        assert!(verify_sha256(&path, &expected).is_ok());
    }

    #[test]
    fn test_verify_sha256_accepts_uppercase_expected() {
        // The expected digest is compared case-insensitively, so an uppercase
        // hex string must still verify.
        let dir = tempfile::tempdir().expect("test setup failed");
        let path = dir.path().join("test.bin");
        std::fs::write(&path, b"test content").expect("test setup failed");

        let expected = sha256_hex(b"test content").to_ascii_uppercase();
        assert!(verify_sha256(&path, &expected).is_ok());
    }

    #[test]
    fn test_verify_sha256_streams_file_larger_than_chunk() {
        use crate::update::constants::HASH_CHUNK_BYTES;
        // A file spanning several read buffers, ending on a partial chunk. This
        // pins that the hasher is fed only the bytes actually read (buf[..n])
        // and accumulates across chunk boundaries: a whole-buffer bug would
        // corrupt the digest of any file whose size is not a chunk multiple.
        let dir = tempfile::tempdir().expect("test setup failed");
        let path = dir.path().join("multi_chunk.bin");
        let size = HASH_CHUNK_BYTES * 2 + HASH_CHUNK_BYTES / 2 + 7;
        let pattern = b"BIRDA";
        let content: Vec<u8> = (0..size).map(|i| pattern[i % pattern.len()]).collect();
        std::fs::write(&path, &content).expect("test setup failed");

        let expected = sha256_hex(&content);
        assert!(verify_sha256(&path, &expected).is_ok());
    }

    #[test]
    fn test_verify_sha256_empty_file() {
        // The streaming loop's immediate-break path: the first read returns 0,
        // so the digest is SHA256 of no input.
        let dir = tempfile::tempdir().expect("test setup failed");
        let path = dir.path().join("empty.bin");
        std::fs::File::create(&path).expect("test setup failed");

        let expected = sha256_hex(b"");
        assert!(verify_sha256(&path, &expected).is_ok());
    }

    #[test]
    fn test_verify_sha256_missing_file_is_io_not_mismatch() {
        // Covers the `File::open` error arm on every platform (a directory path
        // only fails at open on Windows), and locks that a missing file is
        // `Error::Io`, never a spurious mismatch that would delete a model.
        let dir = tempfile::tempdir().expect("test setup failed");
        let path = dir.path().join("does-not-exist.bin");

        let err = verify_sha256(&path, ALL_ZERO_SHA256)
            .expect_err("a missing file should fail to verify");
        assert!(matches!(err, Error::Io(_)));
        assert!(!is_checksum_mismatch(&err));
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
        // A directory cannot be read as a file: on Linux `File::open` succeeds
        // and the first `read` returns EISDIR; on Windows the open itself fails.
        // Either way it stands in for the EACCES/EIO read failures on failing
        // hardware, and locks the contract that a read failure surfaces as
        // `Error::Io`, never as a spurious checksum mismatch.
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
