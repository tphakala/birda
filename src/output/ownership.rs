//! Which input an existing result file was written for.
//!
//! A result file's name says which input it was planned for, but a name is
//! not a record: an earlier run with other inputs, other arguments or an older
//! naming scheme can have left a file at the path a different input plans today.
//! These helpers read the source a result file records about itself, so the
//! caller can tell its own output from someone else's.
//!
//! What each format records:
//!
//! | Format | Evidence |
//! |---|---|
//! | JSON | `source_path` (absolute), or for older files `source_file` (name only) |
//! | CSV | `File` column of the first data row |
//! | Raven | `Begin Path` column of the first data row |
//! | Audacity, Kaleidoscope, Parquet | none |
//!
//! Kaleidoscope is left out because its writer does not escape commas, so a
//! path cannot be read back reliably. Parquet would need the Arrow reader for a
//! file name only.

use crate::config::OutputFormat;
use crate::constants::output_columns;
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

/// Path segment that stands for the current folder.
const CURRENT_DIR: &str = ".";
/// Path segment that stands for the parent folder.
const PARENT_DIR: &str = "..";
/// Segment of a Windows verbatim path (`\\?\C:\x`) that is not a folder.
const VERBATIM_MARKER: &str = "?";
/// Prefix of a Windows verbatim path.
const VERBATIM_PREFIX: &str = r"\\?\";
/// Last character of a drive segment (`C:`).
const DRIVE_SUFFIX: char = ':';

/// The two fields of a JSON result file that name its source.
#[derive(Deserialize)]
struct RecordedJson {
    source_file: String,
    #[serde(default)]
    source_path: Option<String>,
}

/// The source a result file records for itself, as written in it.
///
/// `None` when the format records none, the file holds no detection (CSV and
/// Raven write a header only), or it cannot be read or parsed. A caller treats
/// that as "no evidence", not as a mismatch.
#[must_use]
pub fn recorded_source(path: &Path, format: OutputFormat) -> Option<String> {
    match format {
        OutputFormat::Json => recorded_in_json(path),
        OutputFormat::Csv => first_value(path, b',', true, output_columns::CSV_SOURCE),
        OutputFormat::Raven => first_value(path, b'\t', false, output_columns::RAVEN_SOURCE),
        OutputFormat::Audacity | OutputFormat::Kaleidoscope | OutputFormat::Parquet => None,
    }
}

fn recorded_in_json(path: &Path) -> Option<String> {
    let file = File::open(path).ok()?;
    let recorded: RecordedJson = serde_json::from_reader(BufReader::new(file)).ok()?;
    recorded
        .source_path
        .filter(|p| !p.is_empty())
        .or_else(|| Some(recorded.source_file).filter(|n| !n.is_empty()))
}

/// Value of `column` in the first data row of a delimited file, found by its
/// header name.
fn first_value(path: &Path, delimiter: u8, quoting: bool, column: &str) -> Option<String> {
    let file = File::open(path).ok()?;
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .quoting(quoting)
        .flexible(true)
        .from_reader(file);
    let index = reader
        .headers()
        .ok()?
        .iter()
        .position(|name| name.trim_start_matches('\u{feff}').trim() == column)?;
    let record = reader.records().next()?.ok()?;
    record
        .get(index)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

/// Text of an input's identity, as written into result files.
///
/// A Windows verbatim drive path (`\\?\C:\x`) is written as `C:\x`, so the
/// value reads the way the user would type it. UNC verbatim paths are left
/// alone.
#[must_use]
pub fn identity_text(identity: &Path) -> String {
    let text = identity.to_string_lossy();
    if let Some(rest) = text.strip_prefix(VERBATIM_PREFIX) {
        let mut chars = rest.chars();
        if let (Some(letter), Some(DRIVE_SUFFIX)) = (chars.next(), chars.next())
            && letter.is_ascii_alphabetic()
        {
            return rest.to_string();
        }
    }
    text.into_owned()
}

/// Split a path written on any platform into its segments.
fn segments(text: &str) -> Vec<&str> {
    text.split(['/', '\\']).filter(|s| !s.is_empty()).collect()
}

/// Whether a segment says nothing about which folder holds the file.
fn is_marker(segment: &str) -> bool {
    segment == CURRENT_DIR
        || segment == PARENT_DIR
        || segment == VERBATIM_MARKER
        || segment.ends_with(DRIVE_SUFFIX)
}

/// Compare two path segments, ignoring case where the filesystem does.
fn same_segment(a: &str, b: &str) -> bool {
    if cfg!(any(windows, target_os = "macos")) {
        a.to_lowercase() == b.to_lowercase()
    } else {
        a == b
    }
}

/// Whether `text` is an absolute path on Unix or Windows.
fn is_absolute_text(text: &str) -> bool {
    let mut chars = text.chars();
    match (chars.next(), chars.next()) {
        (Some('/' | '\\'), _) => true,
        (Some(letter), Some(DRIVE_SUFFIX)) => letter.is_ascii_alphabetic(),
        _ => false,
    }
}

/// Whether a recorded source is compatible with an input.
///
/// A recorded path may be absolute, relative to wherever the earlier run
/// started (`in/a/x.wav`, `../in/a/x.wav`, `./x.wav`), or only a file name
/// (older JSON). The folders a relative path names are the ones it needs below
/// its starting point, so the check keeps the trailing segments that name real
/// folders and the file, dropping the leading `.`, `..`, `?` and drive
/// segments, and requires them to be the tail of one of the `identities`.
///
/// A recorded value with nothing left after that has no evidence, which counts
/// as compatible.
#[must_use]
pub fn recorded_matches(recorded: &str, identities: &[&Path]) -> bool {
    let all = segments(recorded);
    let kept = all.iter().rev().take_while(|s| !is_marker(s)).count();
    if kept == 0 {
        return true;
    }
    let tail = &all[all.len() - kept..];
    identities.iter().any(|identity| {
        let text = identity.to_string_lossy();
        let own = segments(&text);
        own.len() >= tail.len()
            && own[own.len() - tail.len()..]
                .iter()
                .zip(tail)
                .all(|(a, b)| same_segment(a, b))
    })
}

/// The last segment of a recorded source: the file name, whichever separator
/// the value was written with.
#[must_use]
pub fn recorded_file_name(recorded: &str) -> &str {
    segments(recorded).last().copied().unwrap_or(recorded)
}

/// Resolve `..` and `.` segments lexically, the way a shell does before it
/// touches the filesystem.
fn collapse_parent_segments(segments: Vec<&str>) -> Vec<&str> {
    let mut kept: Vec<&str> = Vec::with_capacity(segments.len());
    for segment in segments {
        match segment {
            PARENT_DIR => {
                kept.pop();
            }
            CURRENT_DIR => {}
            other => kept.push(other),
        }
    }
    kept
}

/// Whether a recorded source is, in full, the identity of one of `sources`.
///
/// A relative value (CSV and Raven hold the input path as it was typed) is
/// resolved against the current directory first, with `..` segments collapsed
/// lexically, so it counts only when it names that exact input from here. The
/// directory the earlier run started in is not recorded: a value typed in
/// another directory can resolve to an unrelated input of this run. A path
/// that goes through a symlinked folder does not match the resolved identity.
///
/// This is deliberately stricter than [`recorded_matches`]: a relative value
/// such as `audio.wav` would otherwise match any unrelated `.../audio.wav`.
#[must_use]
pub fn recorded_is_exactly_one_of(recorded: &str, sources: &[PathBuf]) -> bool {
    let absolute = if is_absolute_text(recorded) {
        recorded.to_string()
    } else {
        match std::path::absolute(recorded) {
            Ok(path) => path.to_string_lossy().into_owned(),
            Err(_) => return false,
        }
    };
    let wanted = collapse_parent_segments(segments(&absolute));
    sources.iter().any(|source| {
        let text = identity_text(source);
        let own = segments(&text);
        own.len() == wanted.len() && own.iter().zip(&wanted).all(|(a, b)| same_segment(a, b))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::{CsvWriter, Detection, JsonResultWriter, OutputWriter, RavenWriter};

    fn detection(file: &str) -> Detection {
        Detection::from_label(
            "Passer domesticus_House Sparrow",
            0.95,
            0.0,
            3.0,
            PathBuf::from(file),
        )
    }

    #[test]
    fn test_recorded_matches_table() {
        let identity = Path::new("/data/in/a/x.wav");
        let cases: &[(&str, bool)] = &[
            ("/data/in/a/x.wav", true),
            ("in/a/x.wav", true),
            ("../in/a/x.wav", true),
            ("./x.wav", true),
            ("x.wav", true),
            ("a/x.wav", true),
            (r"C:\data\in\a\x.wav", true),
            (r"\\?\C:\data\in\a\x.wav", true),
            ("", true),
            (".", true),
            ("..", true),
            ("/data/in/b/x.wav", false),
            ("/data/in/a/y.wav", false),
            ("b/x.wav", false),
            ("x.wav.wav", false),
            ("./y.wav", false),
        ];

        for (recorded, expected) in cases {
            assert_eq!(
                recorded_matches(recorded, &[identity]),
                *expected,
                "recorded {recorded:?}"
            );
        }
    }

    #[test]
    fn test_recorded_matches_any_of_the_identities() {
        let canonical = Path::new("/data/real/x.wav");
        let typed = Path::new("/home/me/link/x.wav");

        assert!(recorded_matches("/home/me/link/x.wav", &[canonical, typed]));
        assert!(!recorded_matches("/home/me/link/x.wav", &[canonical]));
    }

    #[test]
    fn test_recorded_is_exactly_one_of_needs_the_whole_path() {
        let sources = [PathBuf::from("/data/in/a/x.wav")];

        assert!(recorded_is_exactly_one_of("/data/in/a/x.wav", &sources));
        assert!(!recorded_is_exactly_one_of("a/x.wav", &sources));
        assert!(!recorded_is_exactly_one_of("x.wav", &sources));
        assert!(!recorded_is_exactly_one_of("/other/in/a/x.wav", &sources));
        assert!(!recorded_is_exactly_one_of("/in/a/x.wav", &sources));
        assert!(!recorded_is_exactly_one_of("/data/in/a/x.wav", &[]));
    }

    #[test]
    fn test_recorded_is_exactly_one_of_resolves_a_relative_value_against_the_current_directory() {
        // CSV and Raven hold the input path as it was typed. The run that wrote
        // `in/x.wav` started in this directory unless the user moved, and a
        // value that does not resolve to an input of this run claims nothing.
        let cwd = std::env::current_dir().unwrap();
        let sources = [cwd.join("in").join("x.wav")];

        assert!(recorded_is_exactly_one_of("in/x.wav", &sources));
        assert!(recorded_is_exactly_one_of("./in/x.wav", &sources));
        assert!(recorded_is_exactly_one_of("sub/../in/x.wav", &sources));
        assert!(!recorded_is_exactly_one_of("x.wav", &sources));
        assert!(!recorded_is_exactly_one_of("../in/x.wav", &sources));
    }

    #[cfg(any(windows, target_os = "macos"))]
    #[test]
    fn test_segments_differing_only_in_case_match_where_the_filesystem_ignores_case() {
        assert!(recorded_matches(
            r"C:\In\X.wav",
            &[Path::new(r"c:\in\x.wav")]
        ));
        assert!(recorded_is_exactly_one_of(
            "/Data/In/X.wav",
            &[PathBuf::from("/data/in/x.wav")]
        ));
    }

    #[cfg(not(any(windows, target_os = "macos")))]
    #[test]
    fn test_segments_differing_in_case_are_different_names_where_the_filesystem_keeps_case() {
        assert!(!recorded_matches(
            "/Data/In/X.wav",
            &[Path::new("/data/in/x.wav")]
        ));
        assert!(!recorded_is_exactly_one_of(
            "/Data/In/X.wav",
            &[PathBuf::from("/data/in/x.wav")]
        ));
    }

    #[test]
    fn test_recorded_file_name_is_the_last_segment_with_either_separator() {
        assert_eq!(recorded_file_name("/data/in/x.wav"), "x.wav");
        assert_eq!(recorded_file_name(r"C:\data\in\x.wav"), "x.wav");
        assert_eq!(recorded_file_name("x.wav"), "x.wav");
        assert_eq!(recorded_file_name("in/x.wav/"), "x.wav");
    }

    #[test]
    fn test_recorded_is_exactly_one_of_reads_a_windows_verbatim_identity() {
        let sources = [PathBuf::from(r"\\?\C:\in\x.wav")];

        assert!(recorded_is_exactly_one_of(r"C:\in\x.wav", &sources));
    }

    #[test]
    fn test_identity_text_strips_a_verbatim_drive_prefix_only() {
        assert_eq!(identity_text(Path::new(r"\\?\C:\in\x.wav")), r"C:\in\x.wav");
        assert_eq!(
            identity_text(Path::new(r"\\?\UNC\host\share\x.wav")),
            r"\\?\UNC\host\share\x.wav"
        );
        assert_eq!(identity_text(Path::new("/in/x.wav")), "/in/x.wav");
    }

    #[test]
    fn test_json_round_trip_gives_the_source_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.BirdNET.json");
        let mut writer =
            JsonResultWriter::new(&path, "x.wav", 10.0, "m", 0.1, 0.0, None, None, None)
                .unwrap()
                .with_source_path("/data/in/a/x.wav");
        writer.write_header().unwrap();
        writer.finalize().unwrap();

        assert_eq!(
            recorded_source(&path, OutputFormat::Json),
            Some("/data/in/a/x.wav".to_string())
        );
    }

    #[test]
    fn test_json_without_source_path_gives_the_file_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.BirdNET.json");
        let mut writer =
            JsonResultWriter::new(&path, "x.wav", 10.0, "m", 0.1, 0.0, None, None, None).unwrap();
        writer.write_header().unwrap();
        writer.finalize().unwrap();

        assert_eq!(
            recorded_source(&path, OutputFormat::Json),
            Some("x.wav".to_string())
        );
    }

    #[test]
    fn test_unreadable_or_empty_files_have_no_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let empty = dir.path().join("empty");
        std::fs::write(&empty, "").unwrap();
        let junk = dir.path().join("junk");
        std::fs::write(&junk, "not a result file").unwrap();
        let missing = dir.path().join("missing");

        for format in [OutputFormat::Json, OutputFormat::Csv, OutputFormat::Raven] {
            for path in [&empty, &junk, &missing] {
                assert_eq!(
                    recorded_source(path, format),
                    None,
                    "{format:?} {}",
                    path.display()
                );
            }
        }
    }

    #[test]
    fn test_formats_without_a_source_column_have_no_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.txt");
        std::fs::write(&path, "0.0\t3.0\tHouse Sparrow\t0.9\n").unwrap();

        for format in [
            OutputFormat::Audacity,
            OutputFormat::Kaleidoscope,
            OutputFormat::Parquet,
        ] {
            assert_eq!(recorded_source(&path, format), None);
        }
    }

    #[test]
    fn test_csv_round_trip_with_bom_and_extra_columns() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.BirdNET.results.csv");
        let mut writer =
            CsvWriter::new(&path, vec!["lat".to_string(), "model".to_string()], true).unwrap();
        writer.write_header().unwrap();
        writer.write_detection(&detection("in/a/x.wav")).unwrap();
        writer.finalize().unwrap();

        assert_eq!(
            recorded_source(&path, OutputFormat::Csv),
            Some("in/a/x.wav".to_string())
        );
    }

    #[test]
    fn test_csv_round_trip_with_a_comma_in_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.BirdNET.results.csv");
        let mut writer = CsvWriter::new(&path, Vec::new(), false).unwrap();
        writer.write_header().unwrap();
        writer
            .write_detection(&detection("recordings, 2024/x.wav"))
            .unwrap();
        writer.finalize().unwrap();

        assert_eq!(
            recorded_source(&path, OutputFormat::Csv),
            Some("recordings, 2024/x.wav".to_string())
        );
    }

    #[test]
    fn test_csv_with_only_a_header_has_no_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.BirdNET.results.csv");
        let mut writer = CsvWriter::new(&path, Vec::new(), true).unwrap();
        writer.write_header().unwrap();
        writer.finalize().unwrap();

        assert_eq!(recorded_source(&path, OutputFormat::Csv), None);
    }

    #[test]
    fn test_raven_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.BirdNET.selection.table.txt");
        let mut writer = RavenWriter::new(&path).unwrap();
        writer.write_header().unwrap();
        writer.write_detection(&detection("in/a/x.wav")).unwrap();
        writer.finalize().unwrap();

        assert_eq!(
            recorded_source(&path, OutputFormat::Raven),
            Some("in/a/x.wav".to_string())
        );
    }

    #[test]
    fn test_source_columns_are_the_names_the_writers_put_in_their_headers() {
        // The reader finds its column by name. If a writer renamed the header,
        // every round trip above would also fail, but this names the cause.
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("a.csv");
        let mut csv_writer = CsvWriter::new(&csv_path, Vec::new(), false).unwrap();
        csv_writer.write_header().unwrap();
        csv_writer.finalize().unwrap();
        let raven_path = dir.path().join("a.txt");
        let mut raven_writer = RavenWriter::new(&raven_path).unwrap();
        raven_writer.write_header().unwrap();
        raven_writer.finalize().unwrap();

        let csv_header = std::fs::read_to_string(&csv_path).unwrap();
        let raven_header = std::fs::read_to_string(&raven_path).unwrap();

        assert!(
            csv_header
                .trim_end()
                .split(',')
                .any(|c| c == output_columns::CSV_SOURCE)
        );
        assert!(
            raven_header
                .trim_end()
                .split('\t')
                .any(|c| c == output_columns::RAVEN_SOURCE)
        );
    }
}
