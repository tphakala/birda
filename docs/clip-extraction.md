# Clip Extraction Guide

The `birda clip` command extracts audio segments from recordings based on BirdNET detection results. It automatically organizes clips by species and intelligently merges overlapping detections.

## Quick Start

```bash
# Extract all detections from a results file
birda clip recording.BirdNET.results.csv

# Filter by confidence threshold (70%)
birda clip recording.BirdNET.results.csv -c 0.7

# Process multiple files
birda clip *.BirdNET.results.csv -c 0.8
```

## How It Works

1. **Parse detection files** - Reads BirdNET CSV results with columns: Start (s), End (s), Scientific name, Common name, Confidence
2. **Filter by confidence** - Only processes detections above the threshold
3. **Group by species** - Clusters detections by scientific name
4. **Merge overlapping clips** - Combines adjacent detections (with padding) into single clips
5. **Extract audio** - Seeks to each time range and writes 16-bit WAV files
6. **Organize output** - Creates species subdirectories with descriptive filenames

## Command Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `clips` | Output directory for extracted clips |
| `-c, --confidence` | `0.0` | Minimum confidence threshold (0.0-1.0) |
| `--pre` | `5.0` | Seconds of audio before each detection |
| `--post` | `5.0` | Seconds of audio after each detection |
| `-a, --audio` | auto | Explicit source audio file path |
| `--base-dir` | - | Base directory for resolving audio paths |

## Audio File Resolution

The clipper automatically finds the source audio file based on the detection filename:

1. **Direct match**: `recording.wav.BirdNET.results.csv` → `recording.wav`
2. **Extension fallback**: If original not found, tries `.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`
3. **Explicit override**: Use `-a path/to/audio.wav` to specify manually

### Base Directory

When audio files are in a different location than detection files:

```bash
# Detection files in ./results, audio in ./recordings
birda clip results/*.csv --base-dir ./recordings
```

## Output Format

### Directory Structure

```
clips/
├── Parus major/
│   ├── Parus major_92p_10.5-18.5.wav
│   └── Parus major_85p_45.0-53.0.wav
├── Turdus merula/
│   └── Turdus merula_88p_120.0-128.0.wav
└── Dendrocopos major/
    └── Dendrocopos major_76p_200.0-211.0.wav
```

### Filename Format

```
{Scientific name}_{confidence}p_{start}-{end}.wav
```

- **Scientific name**: Species identifier (filesystem-safe)
- **confidence**: Detection confidence as percentage (e.g., `92p` = 92%)
- **start-end**: Time range in seconds from source audio

## Detection Merging

Adjacent or overlapping detections for the same species are merged into single clips. This prevents duplicate extractions and creates more natural listening segments.

### Example

Given these detections for "Parus major":
- Detection 1: 10.0s - 13.0s (85%)
- Detection 2: 15.0s - 18.0s (92%)

With default padding (5s pre, 5s post):
- Range 1: 5.0s - 18.0s
- Range 2: 10.0s - 23.0s

These overlap, so they merge into a single clip:
- **Merged**: 5.0s - 23.0s (max confidence: 92%)

## Audio Format

Output files are:
- **Format**: WAV (RIFF)
- **Channels**: Mono
- **Bit depth**: 16-bit signed integer
- **Sample rate**: Same as source audio

## Examples

### Basic Extraction

```bash
# Extract all detections
birda clip recording.BirdNET.results.csv
```

### High-Confidence Only

```bash
# Only extract detections with 80%+ confidence
birda clip recording.BirdNET.results.csv -c 0.8
```

### Custom Padding

```bash
# Shorter clips: 2s before, 3s after each detection
birda clip recording.BirdNET.results.csv --pre 2 --post 3
```

### Batch Processing

```bash
# Process all CSV files in a directory
birda clip /path/to/results/*.csv -o /path/to/clips -c 0.7
```

### Explicit Audio Source

```bash
# When audio file is in a different location
birda clip results.csv -a /recordings/2024-06-15_dawn.flac
```

### Organized Workflow

```bash
# Analyze recordings, then extract best clips
birda analyze recordings/ -o detections/
birda clip detections/*.csv -c 0.85 -o best_clips/
```

## Performance

- **Streaming extraction**: Audio is read and written in chunks, not loaded entirely into memory
- **Efficient seeking**: Uses format-native seeking when available (WAV, FLAC)
- **Progress indication**: Shows extraction progress with time estimates

## Troubleshooting

### "Source audio file not found"

The clipper couldn't locate the audio file. Solutions:
1. Ensure the audio file exists in the same directory as the CSV
2. Use `-a` to specify the audio file path explicitly
3. Use `--base-dir` if audio files are in a different directory

### "No detections above confidence threshold"

All detections in the file are below your threshold. Try:
1. Lower the confidence threshold with `-c 0.5`
2. Check the CSV file to see what confidence values are present

### Empty clips directory

If no clips are extracted:
1. Verify the CSV file contains valid detections
2. Check that confidence values are in the expected 0.0-1.0 range
3. Ensure the audio file is readable and not corrupted

## Supported Audio Formats

Input (source audio):
- WAV (recommended for fastest seeking)
- FLAC
- MP3
- M4A/AAC

Output: WAV only (for maximum compatibility)
