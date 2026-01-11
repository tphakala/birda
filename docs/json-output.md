# JSON Output Guide

Birda supports structured JSON output for programmatic integration with GUIs, web applications, and automation scripts.

## Overview

There are two JSON-related features:

1. **CLI Output Mode** (`--output-mode json|ndjson`) - Controls how birda communicates progress and results to stdout
2. **JSON File Format** (`-f json`) - Writes detection results to `.BirdNET.json` files

## CLI Output Mode

Use `--output-mode` to get structured JSON output instead of human-readable text.

### Output Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `human` | Default. Progress bars, colors, human-readable text | Interactive CLI use |
| `json` | Buffered JSON array at completion | Simple integrations, single result parsing |
| `ndjson` | Newline-delimited JSON, one event per line | Streaming, real-time progress, GUI apps |

### Basic Usage

```bash
# Get JSON output for any command
birda --output-mode json config show
birda --output-mode json models list
birda --output-mode json providers

# NDJSON for real-time streaming
birda --output-mode ndjson recording.wav
```

### Environment Variable

Set the default output mode:

```bash
export BIRDA_OUTPUT_MODE=json
birda config show  # Now outputs JSON by default
```

### Configuration File

Set in `config.toml`:

```toml
[output]
default_format = "json"  # or "ndjson" or "human"
```

## JSON Envelope Format

All JSON output follows a consistent envelope structure:

```json
{
  "spec_version": "1.0",
  "timestamp": "2025-01-11T12:34:56.789Z",
  "event": "result",
  "payload": { ... }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `spec_version` | string | API version for compatibility checking |
| `timestamp` | string | ISO 8601 UTC timestamp |
| `event` | string | Event type (see below) |
| `payload` | object | Event-specific data |

## Event Types

### Pipeline Events (Analysis)

| Event | Description |
|-------|-------------|
| `pipeline_started` | Analysis beginning, includes total files and model info |
| `file_started` | Starting to process a file |
| `progress` | Periodic progress update |
| `file_completed` | File finished (success, failed, or skipped) |
| `pipeline_completed` | All files processed, includes summary |

### Result Events (Commands)

| Event | Description |
|-------|-------------|
| `result` | Command result with `result_type` discriminator |
| `error` | Error occurred |
| `cancelled` | Operation was cancelled |

### Result Types

The `result` event includes a `result_type` field:

| Result Type | Command |
|-------------|---------|
| `config` | `birda config show` |
| `model_list` | `birda models list` |
| `model_info` | `birda models info <id>` |
| `providers` | `birda providers` |
| `species_list` | `birda species` |
| `clip_extraction` | `birda clip` |

## Example: Real-Time Progress with NDJSON

For GUI applications that need real-time progress:

```bash
birda --output-mode ndjson recording.wav 2>/dev/null
```

Output (one JSON object per line):

```json
{"spec_version":"1.0","timestamp":"...","event":"pipeline_started","payload":{"total_files":1,"model":"birdnet-v24","min_confidence":0.1}}
{"spec_version":"1.0","timestamp":"...","event":"file_started","payload":{"file":"recording.wav","index":0,"estimated_segments":100}}
{"spec_version":"1.0","timestamp":"...","event":"progress","payload":{"file":{"path":"recording.wav","segments_done":50,"segments_total":100,"percent":50.0}}}
{"spec_version":"1.0","timestamp":"...","event":"file_completed","payload":{"file":"recording.wav","status":"processed","detections":42,"duration_ms":1234}}
{"spec_version":"1.0","timestamp":"...","event":"pipeline_completed","payload":{"status":"success","files_processed":1,"files_failed":0,"total_detections":42,"duration_ms":1234,"realtime_factor":85.2}}
```

## Example: Command Results

### Config Show

```bash
birda --output-mode json config show
```

```json
{
  "spec_version": "1.0",
  "timestamp": "2025-01-11T12:34:56.789Z",
  "event": "result",
  "payload": {
    "result_type": "config",
    "config_path": "/home/user/.config/birda/config.toml",
    "config": {
      "defaults": {
        "model": "birdnet-v24",
        "min_confidence": 0.1
      },
      "models": { ... }
    }
  }
}
```

### Models List

```bash
birda --output-mode json models list
```

```json
{
  "spec_version": "1.0",
  "timestamp": "2025-01-11T12:34:56.789Z",
  "event": "result",
  "payload": {
    "result_type": "model_list",
    "models": [
      {
        "id": "birdnet-v24",
        "model_type": "birdnet-v24",
        "is_default": true,
        "path": "/home/user/.local/share/birda/models/birdnet.onnx",
        "labels_path": "/home/user/.local/share/birda/models/labels.txt",
        "has_meta_model": true
      }
    ]
  }
}
```

### Providers

```bash
birda --output-mode json providers
```

```json
{
  "spec_version": "1.0",
  "timestamp": "2025-01-11T12:34:56.789Z",
  "event": "result",
  "payload": {
    "result_type": "providers",
    "providers": [
      {"id": "cpu", "name": "CPU", "description": "CPU (always available)"},
      {"id": "cuda", "name": "CUDA", "description": "CUDA (NVIDIA GPU acceleration)"}
    ]
  }
}
```

### Species List

```bash
birda --output-mode json species --lat 60.17 --lon 24.94 --week 24
```

```json
{
  "spec_version": "1.0",
  "timestamp": "2025-01-11T12:34:56.789Z",
  "event": "result",
  "payload": {
    "result_type": "species_list",
    "lat": 60.17,
    "lon": 24.94,
    "week": 24,
    "threshold": 0.03,
    "species_count": 150,
    "species": [
      {"scientific_name": "Turdus merula", "common_name": "Eurasian Blackbird", "frequency": 0.92},
      {"scientific_name": "Parus major", "common_name": "Great Tit", "frequency": 0.89}
    ]
  }
}
```

### Clip Extraction

```bash
birda --output-mode json clip results.csv -c 0.7
```

```json
{
  "spec_version": "1.0",
  "timestamp": "2025-01-11T12:34:56.789Z",
  "event": "result",
  "payload": {
    "result_type": "clip_extraction",
    "output_dir": "clips",
    "total_clips": 15,
    "total_files": 1,
    "clips": [
      {
        "source_audio": "recording.wav",
        "scientific_name": "Turdus merula",
        "confidence": 0.95,
        "start_time": 12.0,
        "end_time": 18.0,
        "output_file": "clips/Turdus merula/recording_12.0-18.0.wav"
      }
    ]
  }
}
```

## JSON Detection File Format

Use `-f json` to write detection results to JSON files:

```bash
birda -f json recording.wav
# Creates: recording.BirdNET.json
```

### File Structure

```json
{
  "source_file": "recording.wav",
  "analysis_date": "2025-01-11T12:34:56.789Z",
  "model": "birdnet-v24",
  "settings": {
    "min_confidence": 0.1,
    "overlap": 0.0,
    "lat": 60.17,
    "lon": 24.94,
    "week": 24
  },
  "detections": [
    {
      "start_time": 0.0,
      "end_time": 3.0,
      "scientific_name": "Turdus merula",
      "common_name": "Eurasian Blackbird",
      "confidence": 0.95
    }
  ],
  "summary": {
    "total_detections": 42,
    "unique_species": 8,
    "audio_duration_seconds": 3600.0
  }
}
```

## Integration Examples

### Python

```python
import subprocess
import json

# Get models list
result = subprocess.run(
    ["birda", "--output-mode", "json", "models", "list"],
    capture_output=True, text=True
)
data = json.loads(result.stdout)
models = data["payload"]["models"]

for model in models:
    print(f"{model['id']}: {model['model_type']}")
```

### Node.js (Streaming NDJSON)

```javascript
const { spawn } = require('child_process');
const readline = require('readline');

const birda = spawn('birda', ['--output-mode', 'ndjson', 'recording.wav']);

const rl = readline.createInterface({ input: birda.stdout });

rl.on('line', (line) => {
  const event = JSON.parse(line);

  switch (event.event) {
    case 'pipeline_started':
      console.log(`Processing ${event.payload.total_files} files...`);
      break;
    case 'progress':
      if (event.payload.file) {
        console.log(`Progress: ${event.payload.file.percent}%`);
      }
      break;
    case 'file_completed':
      console.log(`Found ${event.payload.detections} detections`);
      break;
  }
});
```

### Shell Script

```bash
#!/bin/bash

# Parse species list JSON with jq
birda --output-mode json species --lat 60.17 --lon 24.94 --week 24 | \
  jq -r '.payload.species[] | "\(.scientific_name): \(.frequency * 100 | floor)%"'
```

## Error Handling

Errors are reported as JSON events:

```json
{
  "spec_version": "1.0",
  "timestamp": "2025-01-11T12:34:56.789Z",
  "event": "error",
  "payload": {
    "code": "file_not_found",
    "severity": "fatal",
    "message": "Audio file not found: recording.wav",
    "suggestion": "Check that the file path is correct"
  }
}
```

Error severities:
- `fatal` - Operation cannot continue
- `warning` - Operation continues with issues

## Notes

- Logs are written to stderr, JSON output to stdout - use `2>/dev/null` to suppress logs
- The `spec_version` field enables backwards-compatible API evolution
- All timestamps are UTC in ISO 8601 format
- File paths in output are absolute paths
