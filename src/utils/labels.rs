//! Labels file reading.

use crate::error::{Error, Result};
use std::path::Path;

/// Read a labels file: one label per line, surrounding whitespace trimmed,
/// blank lines skipped.
///
/// The one reader for every file that is a list of model labels (a classifier's
/// or the geomodel's). Contract: an unreadable file is an error, and so is a
/// file with no labels in it, because a model with no labels cannot be run and
/// an `Ok` empty list only moves the failure to a later, less specific error.
///
/// # Errors
/// - [`Error::LabelsFileNotFound`] if the file does not exist
/// - [`Error::LabelLoad`] if it cannot be read as UTF-8 text, or holds no labels
pub fn read_label_lines(path: &Path) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            Error::LabelsFileNotFound {
                path: path.to_path_buf(),
            }
        } else {
            Error::LabelLoad {
                path: path.display().to_string(),
                reason: e.to_string(),
            }
        }
    })?;

    let labels: Vec<String> = content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect();

    if labels.is_empty() {
        return Err(Error::LabelLoad {
            path: path.display().to_string(),
            reason: "file contains no labels".to_string(),
        });
    }

    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels_file(content: &[u8]) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(content).unwrap();
        file
    }

    #[test]
    fn test_trims_and_skips_blank_lines() {
        let file =
            labels_file(b"  Parus major_Great Tit  \n\n\t\nCyanistes caeruleus_Blue Tit\r\n");

        assert_eq!(
            read_label_lines(file.path()).unwrap(),
            vec!["Parus major_Great Tit", "Cyanistes caeruleus_Blue Tit"]
        );
    }

    #[test]
    fn test_a_missing_file_is_labels_file_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.txt");

        let err = read_label_lines(&path).unwrap_err();

        assert_eq!(
            err.to_string(),
            format!("labels file does not exist: {}", path.display())
        );
    }

    #[test]
    fn test_an_empty_file_is_an_error_not_an_empty_list() {
        // Two readers disagreed on this: one errored, the other returned
        // `Ok(vec![])`, so the `species` command carried on with no labels.
        let file = labels_file(b"");

        let err = read_label_lines(file.path()).unwrap_err();

        assert_eq!(
            err.to_string(),
            format!(
                "failed to load labels from {}: file contains no labels",
                file.path().display()
            )
        );
    }

    #[test]
    fn test_a_file_of_only_blank_lines_is_an_error() {
        let file = labels_file(b"\n  \n\t\n");

        assert!(matches!(
            read_label_lines(file.path()),
            Err(Error::LabelLoad { .. })
        ));
    }

    #[test]
    fn test_invalid_utf8_is_a_label_load_error_naming_the_file() {
        let file = labels_file(&[0xff, 0xfe, b'\n']);

        let err = read_label_lines(file.path()).unwrap_err();

        assert!(
            err.to_string().starts_with(&format!(
                "failed to load labels from {}: ",
                file.path().display()
            )),
            "got: {err}"
        );
    }
}
