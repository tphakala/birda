//! Integration tests for CUDA library detection.

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
        assert!(patterns[0].ends_with(".dll"));
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
        assert!(patterns[0].ends_with(".dylib"));
        assert!(patterns[0].contains('*'));
    }
}
