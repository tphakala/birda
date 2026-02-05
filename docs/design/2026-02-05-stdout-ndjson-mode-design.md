# Stdout NDJSON Stream Mode Design

**Date:** 2026-02-05
**Status:** Implemented
**Feature:** Add `--stdout` flag for GUI integration

## Overview

Add `--stdout` flag that outputs a single NDJSON stream to stdout containing all events: progress updates, detection results, and status information. This enables GUI applications to monitor progress and collect results in real-time.

## Requirements

- Single NDJSON stream to stdout with all events
- Detection results in structured format
- Real-time progress updates
- Single file processing only (batch not supported)
- Works with all existing analysis options (models, confidence, range filtering)

## CLI Interface

### New Flag

```rust
/// Write results to stdout as NDJSON stream (single file only).
#[arg(long)]
pub stdout: bool,
```

### Usage Examples

```bash
# Basic usage - outputs NDJSON stream
birda --stdout audio.wav

# With model and confidence options
birda --stdout -m birdnet-v24 -c 0.3 audio.wav

# With range filtering
birda --stdout --lat 60.17 --lon 24.94 --week 24 audio.wav
```

### Validation Rules

1. `--stdout` requires exactly **one input file**
   - Error: "error: --stdout requires exactly one input file"
2. `--stdout` conflicts with `--output-dir`
   - Error: "error: --stdout cannot be used with --output-dir"
3. `--stdout` conflicts with `--combine`
   - Error: "error: --stdout cannot be used with --combine"
4. `--stdout` conflicts with `--format`
   - Error: "error: --stdout cannot be used with --format (detections are output in JSON format automatically)"
5. When `--stdout` is set, automatically enable `OutputMode::Ndjson`

## Event Stream Structure

### Event Sequence

For successful processing:

1. `pipeline_started` - Total files, model, confidence
2. `file_started` - File path, estimated segments, duration
3. `progress` (multiple) - Batch/file progress percentages
4. `detections` - **NEW EVENT** - All detection results for the file
5. `file_completed` - Status, detection count, duration
6. `pipeline_completed` - Summary statistics

### New Event Type: Detections

```rust
pub enum EventType {
    // ... existing events ...
    PipelineStarted,
    FileStarted,
    Progress,
    FileCompleted,
    PipelineCompleted,
    Error,
    Cancelled,
    Result,
    // NEW:
    Detections,  // Contains detection results for a file
}
```

### Detections Payload Structure

```rust
#[derive(Debug, Clone, Serialize)]
pub struct DetectionsPayload {
    /// Path to source file
    pub file: PathBuf,
    /// All detections found
    pub detections: Vec<DetectionInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DetectionInfo {
    /// Full label (e.g., "Parus major_Great Tit")
    pub species: String,
    /// Common name
    pub common_name: String,
    /// Scientific name
    pub scientific_name: String,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
}
```

### Example NDJSON Output

```json
{"event":"pipeline_started","timestamp":"2024-01-15T10:30:00Z","payload":{"total_files":1,"model":"birdnet-v24","min_confidence":0.1}}
{"event":"file_started","timestamp":"2024-01-15T10:30:00Z","payload":{"file":"audio.wav","index":0,"estimated_segments":100,"duration_seconds":60.0}}
{"event":"progress","timestamp":"2024-01-15T10:30:01Z","payload":{"file":{"percent":10.0,"segments_processed":10,"segments_total":100}}}
{"event":"progress","timestamp":"2024-01-15T10:30:02Z","payload":{"file":{"percent":20.0,"segments_processed":20,"segments_total":100}}}
{"event":"detections","timestamp":"2024-01-15T10:30:05Z","payload":{"file":"audio.wav","detections":[{"species":"Parus major_Great Tit","common_name":"Great Tit","scientific_name":"Parus major","confidence":0.95,"start_time":0.0,"end_time":3.0}]}}
{"event":"file_completed","timestamp":"2024-01-15T10:30:05Z","payload":{"file":"audio.wav","status":"processed","detections":5,"duration_ms":5000}}
{"event":"pipeline_completed","timestamp":"2024-01-15T10:30:05Z","payload":{"status":"success","files_processed":1,"files_failed":0,"total_detections":5}}
```

## Implementation Details

### Output Routing

When `--stdout` is active:
- **stdout**: NDJSON event stream (all events including detections)
- **stderr**: Tracing logs (controlled by `RUST_LOG` env var)
- **Progress bars**: Automatically disabled (OutputMode::Ndjson disables indicatif)

### Processing Flow

1. Validate CLI arguments (single file, no conflicts)
2. Auto-enable `OutputMode::Ndjson`
3. Process file normally (decode → inference → collect detections)
4. **Skip file writing** - No `write_output()` call
5. Emit `detections` event via `reporter.detections()`
6. Complete normally with `file_completed` and `pipeline_completed` events

### Files to Modify

1. **src/cli/args.rs** - Add `stdout` flag to `AnalyzeArgs`
2. **src/output/json_envelope.rs** - Add `Detections` event type and payload structs
3. **src/output/reporter.rs** - Add `detections()` method to `ProgressReporter` trait
4. **src/pipeline/processor.rs** - Emit detections event, skip file writing when stdout mode
5. **src/pipeline/coordinator.rs** - Validate stdout constraints, auto-enable NDJSON mode
6. **src/config/validate.rs** - Add validation for stdout conflicts (if applicable)

## Design Decisions

### Why NDJSON instead of buffered JSON?

- Real-time progress updates for GUI responsiveness
- Stream processing allows parsing events as they arrive
- Consistent with existing `--output-mode ndjson` behavior
- Each line is independent JSON - simpler parsing

### Why single file only?

- Keeps stdout output clean and predictable
- Batch processing typically uses file output
- GUIs typically process one file at a time
- Users can run multiple commands for batch jobs

### Why conflict with --format?

- stdout mode has a fixed output format (NDJSON with detections event)
- Prevents confusion about what format to expect
- Detections are always JSON in the event payload
- File formats (CSV, Raven, etc.) are for file output, not stdout

### Why auto-enable OutputMode::Ndjson?

- Ensures consistent behavior
- Automatically disables progress bars (they interfere with stdout parsing)
- User doesn't need to remember to specify both flags
- Clear single flag for GUI mode

## Testing Strategy

### Unit Tests

- Validate CLI argument conflicts
- Test DetectionInfo serialization
- Test DetectionsPayload construction

### Integration Tests

- Process single file with `--stdout`
- Verify NDJSON stream structure
- Verify all events are emitted
- Verify detections payload contains expected data
- Verify error cases (multiple files, conflicts)

## Future Enhancements

Potential future improvements (not in this implementation):

- Support batch processing with separate detections events per file
- Add `--stdout-format` to choose output format (CSV, JSON, etc.)
- Add metadata to detections payload (model version, system info)
- Support streaming detections (emit as found, not at end)
