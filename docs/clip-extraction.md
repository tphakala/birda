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

Each clip is written to a temporary file in its species directory and renamed into place, so a clip appears at its final path complete or not at all. Without that, interrupting an extraction leaves a WAV whose header says it holds no audio, because the length fields are only filled in once the clip has been written.

Four consequences, none of which affect an ordinary extraction into a fresh directory:

- **Watching for new clips must watch for a move, not a write.** A clip now arrives via `rename`, so inotify reports `IN_MOVED_TO` on the clip name; the `IN_CLOSE_WRITE` happens on the temporary, whose name has no `.wav` suffix. A post-processing hook filtering on `close_write` and `*.wav` will stop firing. Use `-e moved_to` (or both).
- **A clip file that is a hardlink stops tracking** after a re-extraction, since a rename gives the path a new inode.
- **A clip path that is a dangling symlink is replaced** by a regular file rather than written through. An existing symlink is followed, as before.
- **Interrupting an extraction with Ctrl+C can leave one temporary behind** in the species directory, named `.tmp` followed by random characters and with no extension. birda releases its lock files on Ctrl+C but exits without unwinding, so an in-progress temporary is not removed, and nothing sweeps stale ones later; they are safe to delete.

A clip keeps the permissions your umask asks for, exactly as before, so a clip directory served by a web server or read by another account keeps working. Per-file ACLs set on a specific clip are not carried across, because the replacement is a new inode; a default ACL on the directory is inherited as it would be for any new file.

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
birda recordings/ -o detections/
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
