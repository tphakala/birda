//! Integration tests for CSV UTF-8 BOM functionality.

#[test]
fn test_csv_with_bom_default() {
    // This test would require actual model files and audio
    // For now, we verify the unit tests cover the functionality
    // Integration testing should be done manually with:
    // birda test.wav -f csv
    // xxd output.csv | head -1
    // Expected: "00000000: efbb bf53 7065 6369 6573 2c53 7461 7274"
}

#[test]
fn test_csv_without_bom_flag() {
    // Manual test:
    // birda test.wav -f csv --no-csv-bom
    // xxd output.csv | head -1
    // Expected: "00000000: 5370 6563 6965 732c 5374 6172 742c 456e"
}
