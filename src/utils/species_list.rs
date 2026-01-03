//! Species list file reading utilities.

use crate::error::{Error, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Read species list from file.
///
/// # File Format
/// - One species per line
/// - Format: `Genus species_Common Name` (e.g., `Parus major_Great Tit`)
/// - Blank lines are ignored
/// - Compatible with BirdNET-Analyzer species lists
///
/// # Errors
/// - Returns error if file cannot be read
/// - Returns error if file contains invalid UTF-8
pub fn read_species_list(path: &Path) -> Result<Vec<String>> {
    let file = File::open(path).map_err(|e| Error::SpeciesListRead {
        path: path.to_path_buf(),
        source: e,
    })?;

    let reader = BufReader::new(file);
    let mut species = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| Error::SpeciesListRead {
            path: path.to_path_buf(),
            source: e,
        })?;

        let trimmed = line.trim();
        if !trimmed.is_empty() {
            species.push(trimmed.to_string());
        }
    }

    Ok(species)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // Test setup code - panics are acceptable
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_species_list_valid_file() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Parus major_Great Tit").unwrap();
        writeln!(file, "Cyanistes caeruleus_Blue Tit").unwrap();
        writeln!(file).unwrap(); // blank line should be ignored
        writeln!(file, "Sturnus vulgaris_European Starling").unwrap();

        let species = read_species_list(file.path()).unwrap();
        assert_eq!(species.len(), 3);
        assert!(species.contains(&"Parus major_Great Tit".to_string()));
        assert!(species.contains(&"Cyanistes caeruleus_Blue Tit".to_string()));
    }

    #[test]
    fn test_read_species_list_file_not_found() {
        let result = read_species_list(std::path::Path::new("nonexistent.txt"));
        assert!(result.is_err());
    }
}
