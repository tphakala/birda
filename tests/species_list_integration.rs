//! Integration tests for species list feature.
//!
//! Note: These tests require actual model files to run.
//! They will be skipped if model environment variables are not set.
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    fn get_test_model_path() -> Option<(PathBuf, PathBuf, PathBuf)> {
        // Return (model_path, labels_path, meta_model_path) if available
        // Otherwise return None to skip tests
        std::env::var("BIRDA_TEST_MODEL").ok().map(|model| {
            let labels = std::env::var("BIRDA_TEST_LABELS")
                .expect("BIRDA_TEST_LABELS required if BIRDA_TEST_MODEL is set");
            let meta = std::env::var("BIRDA_TEST_META_MODEL")
                .expect("BIRDA_TEST_META_MODEL required if BIRDA_TEST_MODEL is set");
            (
                PathBuf::from(model),
                PathBuf::from(labels),
                PathBuf::from(meta),
            )
        })
    }

    #[test]
    fn test_species_list_generation_integration() {
        if get_test_model_path().is_none() {
            eprintln!("Skipping integration test - model files not configured");
            eprintln!("Set BIRDA_TEST_MODEL, BIRDA_TEST_LABELS, and BIRDA_TEST_META_MODEL to run");
        }

        // Test will run when models are available
        // This is a placeholder for actual integration tests
        // In the future, this could test:
        // - Species list generation from range filter
        // - Species list file filtering
        // - End-to-end analysis with species lists
    }
}
