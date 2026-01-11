//! Tests for detection file parser.

use std::io::Write;

use birda::clipper::parse_detection_file;
use tempfile::NamedTempFile;

#[test]
fn test_parse_birda_csv_format() {
    let csv_content = r#"Start (s),End (s),Scientific name,Common name,Confidence
0.0,3.0,Parus major,Great Tit,0.8542
3.0,6.0,Cyanistes caeruleus,Eurasian Blue Tit,0.7123
6.0,9.0,Parus major,Great Tit,0.9001
"#;

    let mut file = NamedTempFile::with_suffix(".BirdNET.results.csv").unwrap();
    file.write_all(csv_content.as_bytes()).unwrap();
    file.flush().unwrap();

    let detections = parse_detection_file(file.path()).unwrap();

    assert_eq!(detections.len(), 3);

    assert_eq!(detections[0].start, 0.0);
    assert_eq!(detections[0].end, 3.0);
    assert_eq!(detections[0].scientific_name, "Parus major");
    assert_eq!(detections[0].common_name, "Great Tit");
    assert!((detections[0].confidence - 0.8542).abs() < 0.0001);

    assert_eq!(detections[1].scientific_name, "Cyanistes caeruleus");
    assert_eq!(detections[2].scientific_name, "Parus major");
}

#[test]
fn test_parse_csv_with_utf8_bom() {
    let csv_content = "\u{FEFF}Start (s),End (s),Scientific name,Common name,Confidence
0.0,3.0,Parus major,Great Tit,0.8542
";

    let mut file = NamedTempFile::with_suffix(".BirdNET.results.csv").unwrap();
    file.write_all(csv_content.as_bytes()).unwrap();
    file.flush().unwrap();

    let detections = parse_detection_file(file.path()).unwrap();
    assert_eq!(detections.len(), 1);
    assert_eq!(detections[0].scientific_name, "Parus major");
}

#[test]
fn test_parse_empty_file_returns_error() {
    let mut file = NamedTempFile::with_suffix(".csv").unwrap();
    file.write_all(b"Start (s),End (s),Scientific name,Common name,Confidence\n")
        .unwrap();
    file.flush().unwrap();

    let result = parse_detection_file(file.path());
    assert!(result.is_err());
}

#[test]
fn test_parse_missing_column_returns_error() {
    let csv_content = "Start (s),End (s),Scientific name,Confidence
0.0,3.0,Parus major,0.85
";

    let mut file = NamedTempFile::with_suffix(".csv").unwrap();
    file.write_all(csv_content.as_bytes()).unwrap();
    file.flush().unwrap();

    let result = parse_detection_file(file.path());
    assert!(result.is_err());
}

#[test]
fn test_parse_csv_with_optional_columns() {
    // birda CSV output may include optional metadata columns
    let csv_content = r#"Start (s),End (s),Scientific name,Common name,Confidence,Lat,Lon,Week,Model
0.0,3.0,Parus major,Great Tit,0.8542,60.1699,24.9384,15,birdnet-v24
"#;

    let mut file = NamedTempFile::with_suffix(".csv").unwrap();
    file.write_all(csv_content.as_bytes()).unwrap();
    file.flush().unwrap();

    let detections = parse_detection_file(file.path()).unwrap();
    assert_eq!(detections.len(), 1);
    assert_eq!(detections[0].scientific_name, "Parus major");
}

#[test]
fn test_parse_invalid_time_range_returns_error() {
    // End time before start time should be rejected
    let csv_content = r#"Start (s),End (s),Scientific name,Common name,Confidence
5.0,3.0,Parus major,Great Tit,0.85
"#;

    let mut file = NamedTempFile::with_suffix(".csv").unwrap();
    file.write_all(csv_content.as_bytes()).unwrap();
    file.flush().unwrap();

    let result = parse_detection_file(file.path());
    assert!(result.is_err());
}

#[test]
fn test_parse_csv_with_quoted_fields() {
    // birda CSV output may quote fields containing commas
    let csv_content = r#"Start (s),End (s),Scientific name,Common name,Confidence
0.0,3.0,Tyto alba,"Owl, Barn",0.8542
"#;

    let mut file = NamedTempFile::with_suffix(".csv").unwrap();
    file.write_all(csv_content.as_bytes()).unwrap();
    file.flush().unwrap();

    let detections = parse_detection_file(file.path()).unwrap();
    assert_eq!(detections.len(), 1);
    assert_eq!(detections[0].scientific_name, "Tyto alba");
    assert_eq!(detections[0].common_name, "Owl, Barn");
}
