//! Integration tests for CSV UTF-8 BOM functionality.
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

#[test]
#[ignore = "Manual test: requires audio files and models. See comment for instructions."]
fn test_csv_with_bom_default() {
    // This test would require actual model files and audio
    // For now, we verify the unit tests cover the functionality
    // Integration testing should be done manually with:
    // birda test.wav -f csv
    // xxd output.csv | head -1
    // Expected: "00000000: efbb bf53 7065 6369 6573 2c53 7461 7274"
}

#[test]
#[ignore = "Manual test: requires audio files and models. See comment for instructions."]
fn test_csv_without_bom_flag() {
    // Manual test:
    // birda test.wav -f csv --no-csv-bom
    // xxd output.csv | head -1
    // Expected: "00000000: 5370 6563 6965 732c 5374 6172 742c 456e"
}
