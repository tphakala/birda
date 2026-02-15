//! Parquet output format writer.
//!
//! This module provides functionality to write bird detection results in Apache Parquet format,
//! offering better compression, type safety, and integration with data science tooling compared to CSV.

use arrow::array::{ArrayRef, Float32Array, Float64Array, StringArray, UInt8Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use crate::error::Result;
use crate::output::OutputWriter;
use crate::output::types::Detection;

/// Parquet writer for detection results.
///
/// Buffers detections and writes them in batches to a Parquet file for efficient
/// columnar compression.
pub struct ParquetWriter {
    writer: ArrowWriter<File>,
    schema: Arc<Schema>,
    detections: Vec<Detection>,
    batch_size: usize,
}

impl ParquetWriter {
    /// Create a new Parquet writer.
    ///
    /// # Arguments
    ///
    /// * `output_path` - Path where the Parquet file will be written
    /// * `include_additional_columns` - Additional metadata columns to include in schema
    ///
    /// # Errors
    ///
    /// Returns error if file creation fails or Parquet writer initialization fails.
    pub fn new(output_path: &Path, include_additional_columns: &[String]) -> Result<Self> {
        let schema = build_schema(include_additional_columns)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_writer_version(parquet::file::properties::WriterVersion::PARQUET_2_0)
            .build();

        let file =
            File::create(output_path).map_err(|e| crate::error::Error::ParquetFileCreate {
                path: output_path.to_path_buf(),
                source: e,
            })?;

        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).map_err(|e| {
            crate::error::Error::ParquetWrite {
                context: "Failed to initialize Parquet writer".to_string(),
                source: e,
            }
        })?;

        Ok(Self {
            writer,
            schema,
            detections: Vec::new(),
            batch_size: 1000,
        })
    }

    /// Add a detection to the buffer.
    ///
    /// Automatically flushes the batch when `batch_size` is reached.
    ///
    /// # Errors
    ///
    /// Returns error if batch flush fails.
    pub fn write_detection(&mut self, detection: Detection) -> Result<()> {
        self.detections.push(detection);

        if self.detections.len() >= self.batch_size {
            self.flush_batch()?;
        }

        Ok(())
    }

    /// Flush buffered detections to file.
    ///
    /// # Errors
    ///
    /// Returns error if record batch building or writing fails.
    fn flush_batch(&mut self) -> Result<()> {
        if self.detections.is_empty() {
            return Ok(());
        }

        let batch = build_record_batch(&self.detections, &self.schema)?;
        self.writer
            .write(&batch)
            .map_err(|e| crate::error::Error::ParquetWrite {
                context: "Failed to write Parquet record batch".to_string(),
                source: e,
            })?;
        self.detections.clear();

        Ok(())
    }

    /// Finalize and close the writer.
    ///
    /// Flushes any remaining buffered detections and closes the file.
    ///
    /// # Errors
    ///
    /// Returns error if final flush or file close fails.
    pub fn close(mut self) -> Result<()> {
        self.flush_batch()?;
        self.writer
            .close()
            .map_err(|e| crate::error::Error::ParquetWrite {
                context: "Failed to close Parquet writer".to_string(),
                source: e,
            })?;
        Ok(())
    }
}

impl OutputWriter for ParquetWriter {
    fn write_header(&mut self) -> Result<()> {
        // Parquet doesn't need a separate header - schema is embedded in file format
        Ok(())
    }

    fn write_detection(&mut self, detection: &Detection) -> Result<()> {
        self.write_detection(detection.clone())
    }

    fn finalize(&mut self) -> Result<()> {
        // Can't consume self in trait method, so we flush manually
        self.flush_batch()
    }
}

/// Build Arrow schema based on included columns.
///
/// Creates a schema with core detection columns plus any additional metadata columns.
///
/// # Arguments
///
/// * `include_additional_columns` - Names of additional metadata columns to include
///
/// # Errors
///
/// Currently infallible, but returns Result for future extensibility.
fn build_schema(include_additional_columns: &[String]) -> Result<Arc<Schema>> {
    let mut fields = vec![
        Field::new("start_s", DataType::Float32, false),
        Field::new("end_s", DataType::Float32, false),
        Field::new("scientific_name", DataType::Utf8, false),
        Field::new("common_name", DataType::Utf8, false),
        Field::new("confidence", DataType::Float32, false),
        Field::new("file", DataType::Utf8, false),
    ];

    // Add optional metadata columns
    for col in include_additional_columns {
        let field = match col.as_str() {
            "lat" => Field::new("lat", DataType::Float64, true),
            "lon" => Field::new("lon", DataType::Float64, true),
            "week" => Field::new("week", DataType::UInt8, true),
            "model" => Field::new("model", DataType::Utf8, true),
            "overlap" => Field::new("overlap", DataType::Float32, true),
            "sensitivity" => Field::new("sensitivity", DataType::Float32, true),
            "min_conf" => Field::new("min_conf", DataType::Float32, true),
            "species_list" => Field::new("species_list", DataType::Utf8, true),
            _ => continue, // Skip unknown columns
        };
        fields.push(field);
    }

    Ok(Arc::new(Schema::new(fields)))
}

/// Build Arrow RecordBatch from detections.
///
/// Converts a slice of detections into a columnar format suitable for Parquet.
///
/// # Arguments
///
/// * `detections` - Slice of detections to convert
/// * `schema` - Arrow schema defining the table structure
///
/// # Errors
///
/// Returns error if record batch creation fails.
fn build_record_batch(detections: &[Detection], schema: &Arc<Schema>) -> Result<RecordBatch> {
    let n = detections.len();

    // Build core columns
    let start_times: Float32Array = detections.iter().map(|d| d.start_time).collect();
    let end_times: Float32Array = detections.iter().map(|d| d.end_time).collect();
    let scientific_names: StringArray = detections
        .iter()
        .map(|d| d.scientific_name.as_str())
        .collect();
    let common_names: StringArray = detections.iter().map(|d| d.common_name.as_str()).collect();
    let confidences: Float32Array = detections.iter().map(|d| d.confidence).collect();

    // Extract filenames from file_path
    let files: StringArray = detections
        .iter()
        .map(|d| {
            d.file_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
        })
        .collect();

    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(start_times),
        Arc::new(end_times),
        Arc::new(scientific_names),
        Arc::new(common_names),
        Arc::new(confidences),
        Arc::new(files),
    ];

    // Add optional metadata columns based on schema
    for field in schema.fields().iter().skip(6) {
        let array = build_metadata_column(field, detections)?;
        columns.push(array);
    }

    RecordBatch::try_new(schema.clone(), columns).map_err(|e| crate::error::Error::ParquetWrite {
        context: format!("Failed to build record batch: {e}"),
        source: parquet::errors::ParquetError::General(e.to_string()),
    })
}

/// Build metadata column from detections.
///
/// Creates an Arrow array for a specific metadata column based on the field type.
///
/// # Arguments
///
/// * `field` - Arrow field defining the column type
/// * `detections` - Slice of detections to extract metadata from
///
/// # Errors
///
/// Returns error if the field name is not recognized.
fn build_metadata_column(field: &Field, detections: &[Detection]) -> Result<ArrayRef> {
    match field.name().as_str() {
        "lat" => {
            let values: Vec<Option<f64>> = detections.iter().map(|d| d.metadata.lat).collect();
            Ok(Arc::new(Float64Array::from(values)))
        }
        "lon" => {
            let values: Vec<Option<f64>> = detections.iter().map(|d| d.metadata.lon).collect();
            Ok(Arc::new(Float64Array::from(values)))
        }
        "week" => {
            let values: Vec<Option<u8>> = detections.iter().map(|d| d.metadata.week).collect();
            Ok(Arc::new(UInt8Array::from(values)))
        }
        "model" => {
            let values: Vec<Option<&str>> = detections
                .iter()
                .map(|d| d.metadata.model.as_deref())
                .collect();
            Ok(Arc::new(StringArray::from(values)))
        }
        "overlap" => {
            let values: Vec<Option<f32>> = detections.iter().map(|d| d.metadata.overlap).collect();
            Ok(Arc::new(Float32Array::from(values)))
        }
        "sensitivity" => {
            let values: Vec<Option<f32>> =
                detections.iter().map(|d| d.metadata.sensitivity).collect();
            Ok(Arc::new(Float32Array::from(values)))
        }
        "min_conf" => {
            let values: Vec<Option<f32>> = detections.iter().map(|d| d.metadata.min_conf).collect();
            Ok(Arc::new(Float32Array::from(values)))
        }
        "species_list" => {
            let values: Vec<Option<&str>> = detections
                .iter()
                .map(|d| d.metadata.species_list.as_deref())
                .collect();
            Ok(Arc::new(StringArray::from(values)))
        }
        name => Err(crate::error::Error::InvalidColumnName {
            name: name.to_string(),
        }),
    }
}

/// Combine multiple Parquet files into one.
///
/// Reads all input Parquet files and writes their contents into a single output file.
/// All input files must have compatible schemas.
///
/// # Arguments
///
/// * `input_files` - Paths to Parquet files to combine
/// * `output_path` - Path where the combined Parquet file will be written
///
/// # Errors
///
/// Returns error if:
/// - Any input file cannot be read
/// - Input files have incompatible schemas
/// - Output file cannot be written
/// - No input files are provided
pub fn combine_parquet_files(input_files: &[std::path::PathBuf], output_path: &Path) -> Result<()> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::file::reader::SerializedFileReader;

    if input_files.is_empty() {
        return Err(crate::error::Error::NoValidAudioFiles);
    }

    let mut all_batches = Vec::new();
    let mut schema = None;

    // Read all input files
    for file_path in input_files {
        let file = File::open(file_path).map_err(|e| crate::error::Error::ParquetFileCreate {
            path: file_path.clone(),
            source: e,
        })?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
            crate::error::Error::ParquetWrite {
                context: format!("Failed to read Parquet file: {}", file_path.display()),
                source: e,
            }
        })?;

        // Get schema from first file
        if schema.is_none() {
            schema = Some(builder.schema().clone());
        }

        let mut reader = builder
            .build()
            .map_err(|e| crate::error::Error::ParquetWrite {
                context: format!("Failed to create Parquet reader: {}", file_path.display()),
                source: e,
            })?;

        // Read all record batches
        while let Some(batch_result) = reader.next() {
            let batch = batch_result.map_err(|e| crate::error::Error::ParquetWrite {
                context: format!("Failed to read record batch from: {}", file_path.display()),
                source: e,
            })?;
            all_batches.push(batch);
        }
    }

    let schema = schema.ok_or(crate::error::Error::NoValidAudioFiles)?;

    // Write combined file
    let output_file =
        File::create(output_path).map_err(|e| crate::error::Error::ParquetFileCreate {
            path: output_path.to_path_buf(),
            source: e,
        })?;

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_writer_version(parquet::file::properties::WriterVersion::PARQUET_2_0)
        .build();

    let mut writer = ArrowWriter::try_new(output_file, schema, Some(props)).map_err(|e| {
        crate::error::Error::ParquetWrite {
            context: "Failed to create combined Parquet writer".to_string(),
            source: e,
        }
    })?;

    for batch in all_batches {
        writer
            .write(&batch)
            .map_err(|e| crate::error::Error::ParquetWrite {
                context: "Failed to write batch to combined file".to_string(),
                source: e,
            })?;
    }

    writer
        .close()
        .map_err(|e| crate::error::Error::ParquetWrite {
            context: "Failed to close combined Parquet writer".to_string(),
            source: e,
        })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_schema_basic() {
        let schema = build_schema(&[]).ok().unwrap();
        assert_eq!(schema.fields().len(), 6);
        assert_eq!(schema.field(0).name(), "start_s");
        assert_eq!(schema.field(1).name(), "end_s");
        assert_eq!(schema.field(2).name(), "scientific_name");
        assert_eq!(schema.field(3).name(), "common_name");
        assert_eq!(schema.field(4).name(), "confidence");
        assert_eq!(schema.field(5).name(), "file");
    }

    #[test]
    fn test_schema_with_metadata() {
        let schema = build_schema(&["lat".to_string(), "lon".to_string()])
            .ok()
            .unwrap();
        assert_eq!(schema.fields().len(), 8);
        assert!(schema.field_with_name("lat").is_ok());
        assert!(schema.field_with_name("lon").is_ok());
    }

    #[test]
    fn test_record_batch_building() {
        let detections = vec![Detection {
            file_path: PathBuf::from("/path/to/test.wav"),
            start_time: 0.0,
            end_time: 3.0,
            scientific_name: "Poecile atricapillus".to_string(),
            common_name: "Black-capped Chickadee".to_string(),
            confidence: 0.95,
            metadata: Default::default(),
        }];

        let schema = build_schema(&[]).ok().unwrap();
        let batch = build_record_batch(&detections, &schema).ok().unwrap();

        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 6);
    }

    #[test]
    fn test_empty_detections() {
        let schema = build_schema(&[]).ok().unwrap();
        let batch = build_record_batch(&[], &schema).ok().unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 6);
    }
}
