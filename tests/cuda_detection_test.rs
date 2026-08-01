//! Integration tests for CUDA library detection.
// Integration test crate. `unwrap`, `expect` and `panic` are how a test reports
// failure, not unhandled error paths, so rewriting them into propagated errors
// would only hide which assertion fired. Every exact float assertion in these
// tests is on a passed-through value (a literal parsed from a file, a
// coordinate round-tripped through JSON, a clip boundary clamped to a whole
// number) rather than a computed one, so exact equality is the assertion the
// test wants. The crate-level deny still governs everything birda ships.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

use birda::inference::{get_cuda_library_patterns, is_cuda_available};

#[test]
fn test_cuda_detection_doesnt_panic() {
    // Should not panic regardless of whether CUDA is installed
    let _available = is_cuda_available();
    // If this completes, the detection logic is safe
}

#[test]
fn test_cuda_library_patterns_platform_specific() {
    let patterns = get_cuda_library_patterns();

    // Should have at least one pattern
    assert!(!patterns.is_empty());

    #[cfg(target_os = "windows")]
    {
        assert_eq!(patterns, &["cudart64_*.dll"]);
        // `contains`, not `ends_with`: see the macOS arm below. The lint is
        // extension-agnostic, so this arm trips it too, and nothing in CI
        // compiles either one.
        assert!(patterns[0].contains(".dll"));
        assert!(patterns[0].contains('*'));
    }

    #[cfg(target_os = "linux")]
    {
        assert_eq!(patterns, &["libcudart.so.*"]);
        assert!(patterns[0].starts_with("lib"));
        assert!(patterns[0].contains(".so"));
        assert!(patterns[0].contains('*'));
    }

    #[cfg(target_os = "macos")]
    {
        assert_eq!(patterns, &["libcudart.*.dylib"]);
        assert!(patterns[0].starts_with("lib"));
        // `contains` rather than `ends_with`, matching the Linux arm above.
        // `case_sensitive_file_extension_comparisons` fires on `ends_with` with
        // an extension literal and suggests `Path::extension`, which is the
        // wrong question to ask of a glob pattern. Every job in
        // .github/workflows/ci.yml runs on ubuntu-latest, so neither this arm
        // nor the Windows one is ever linted there, which is how both survived
        // the `--all-targets` sweep in #338; they surface for a developer
        // running the gate on that platform.
        assert!(patterns[0].contains(".dylib"));
        assert!(patterns[0].contains('*'));
    }
}
