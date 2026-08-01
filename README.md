# Birda

[![CI](https://github.com/tphakala/birda/actions/workflows/ci.yml/badge.svg)](https://github.com/tphakala/birda/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.92%2B-blue.svg)](https://www.rust-lang.org/)
[![Sponsor](https://img.shields.io/badge/sponsor-GitHub-pink.svg)](https://github.com/sponsors/tphakala)

A fast, cross-platform CLI tool for bird species detection using [BirdNET](https://github.com/birdnet-team/BirdNET-Analyzer) and Google Perch AI models.

## Features

- **Multiple AI Models**: Support for BirdNET v2.4, BirdNET v3.0, Google Perch v2, BSG Finnish Birds, and BattyBirdNET bat classifiers
- **GPU Acceleration**: Optional CUDA support for faster inference on NVIDIA GPUs
- **Species Filtering**: Dynamic range filtering by location/date or static species list files
- **Multiple Output Formats**: CSV, Parquet, JSON, Raven selection tables, Audacity labels, Kaleidoscope CSV
- **JSON Output Mode**: Structured JSON/NDJSON output for GUI integration and automation
- **Graphical User Interface**: Optional cross-platform GUI available separately
- **Batch Processing**: Process entire directories of audio files
- **Flexible Configuration**: TOML-based config with CLI overrides
- **Cross-Platform**: Works on Linux, Windows, and macOS

## Installation

### Pre-built Binaries

Download the latest release from the [Releases](https://github.com/tphakala/birda/releases) page.

> **Windows users**: See the [Windows User Guide](docs/windows-guide.md) for detailed installation and GPU setup instructions.

### From Source

Requires Rust 1.92 or later.

```bash
# Clone the repository
git clone https://github.com/tphakala/birda.git
cd birda

# Build with CUDA support (default)
cargo build --release

# Build CPU-only version
cargo build --release --no-default-features

# Install to ~/.cargo/bin
cargo install --path .
```

### GPU Support (CUDA)

Download the CUDA package from [Releases](https://github.com/tphakala/birda/releases) (e.g., `birda-windows-x64-cuda.zip`). These bundles include all required ONNX Runtime and CUDA libraries - no separate CUDA installation needed.

Requirements:
- NVIDIA GPU with CUDA support
- Up-to-date NVIDIA GPU drivers

### TensorRT Support (Optional)

For maximum GPU performance, TensorRT provides ~2x speedup over CUDA. TensorRT is **not bundled** due to size constraints but can be installed separately:

1. Download TensorRT for CUDA 12.9 from [NVIDIA TensorRT](https://github.com/NVIDIA/TensorRT#downloading-tensorrt-build)
2. Copy the TensorRT DLLs/libs to the birda installation directory:
   - Windows: `nvinfer_10.dll`, `nvinfer_plugin_10.dll`, `nvonnxparser_10.dll`
   - Linux: `libnvinfer.so.10`, `libnvinfer_plugin.so.10`, `libnvonnxparser.so.10`
3. Run with `--tensorrt` flag: `birda --tensorrt recording.wav`

TensorRT requires an NVIDIA GPU with compute capability 5.0+ (GTX 10-series and newer). See [Performance Tips](#performance-tips) for benchmark comparisons.

### Checking Available Execution Providers

To see which execution providers are available on your system:

```bash
birda providers
```

This shows which backends (CPU, CUDA, TensorRT, etc.) are available at compile-time.

**For programmatic use (GUI integration, scripts):**

```bash
birda providers --output-mode json
```

Returns structured JSON output:

```json
{
  "spec_version": "1.0",
  "timestamp": "2026-02-15T12:34:56Z",
  "event": "result",
  "payload": {
    "result_type": "providers",
    "providers": [
      {
        "id": "cpu",
        "name": "CPU",
        "description": "CPU (always available)"
      },
      {
        "id": "cuda",
        "name": "CUDA",
        "description": "CUDA (NVIDIA GPU acceleration)"
      }
    ]
  }
}
```

**Important Notes:**

- **Compile-time vs Runtime**: The `providers` command shows what was available when the binary was built. Actual runtime availability may differ based on drivers and hardware.
- **Provider Selection**: Use `--gpu` for CUDA, `--cpu` for CPU-only, or omit both for auto-selection (GPU if available with CPU fallback).
- **Verification**: During analysis, birda logs which provider was requested and whether it's available. However, due to ONNX Runtime limitations, it cannot detect if a runtime fallback from GPU to CPU occurred.
- **Best Practice**: Check the log output during analysis to verify device selection. If using GPU mode, the logs will indicate whether CUDA is available at compile-time.

Example output:

```text
INFO birda::inference::classifier: Requested device: GPU (CUDA)
DEBUG birda::inference::classifier: Available execution providers: Cpu
WARN birda::inference::classifier: CUDA not available at compile-time, but GPU was requested
WARN birda::inference::classifier: Build will proceed, but may fall back to CPU at runtime
INFO birda::inference::classifier: Loaded model: BirdNetV24, sample_rate: 48000, segment_duration: 3s, device: GPU (CUDA requested, may fallback to CPU)
```

## Graphical User Interface

For users who prefer a graphical interface, [Birda GUI](https://github.com/tphakala/birda-gui) provides a cross-platform desktop application built with Electron.

**Features:**

- Visual file selection and drag-and-drop support
- Real-time progress monitoring
- Interactive detection results with spectrograms
- Model management through GUI
- Runs inference using the Birda CLI tool

**Requirements:**

The GUI requires Birda CLI to be installed and accessible in your system PATH.

**Installation:**

Visit the [Birda GUI releases](https://github.com/tphakala/birda-gui/releases) page for platform-specific installers (Windows, macOS, Linux).

## Quick Start

### 1. Install a Model

```bash
# List available models
birda models list-available

# Install BirdNET (recommended for most users)
birda models install birdnet-v24
```

This downloads the model, its labels, and the shared BirdNET Geomodel v3.0.2 range filter automatically.

### 2. Analyze Audio Files

```bash
# Analyze a single file
birda recording.wav

# Analyze multiple files
birda *.wav

# Analyze a directory
birda /path/to/recordings/

# Analyze with GPU acceleration
birda --gpu -b 64 recording.wav
```

## Clip Extraction

Extract audio clips from detection results, organized by species:

```bash
# Extract clips with 70% confidence threshold
birda clip results.BirdNET.results.csv -c 0.7

# Custom output directory and padding
birda clip *.csv -o my_clips --pre 3 --post 5
```

Clips are saved to species directories (e.g., `clips/Dendrocopos major/`).

**See [Clip Extraction Guide](docs/clip-extraction.md) for detailed documentation.**

## Species Filtering

Birda supports filtering detections by species using two complementary approaches:

### Dynamic Range Filtering

Filter species based on location and date using the BirdNET Geomodel v3.0.2, which covers 12,012 species and is shared by every classifier:

```bash
# Filter by location and week
birda recording.wav --lat 60.17 --lon 24.94 --week 24

# Filter by location and month/day
birda recording.wav --lat 42.36 --lon -71.06 --month 6 --day 15
```

### Static Species Lists

Use pre-generated species list files compatible with BirdNET-Analyzer:

```bash
# Generate a species list for your location
birda species --lat 60.17 --lon 24.94 --week 24 --output my_species.txt

# Use the species list during analysis
birda recording.wav --slist my_species.txt
```

**See [Species List Usage Guide](docs/species-list-usage.md) for detailed documentation.**

## Usage

```
birda [OPTIONS] [INPUTS]... [COMMAND]

Commands:
  clip       Extract audio clips from detection results
  config     Manage configuration
  models     Manage models (install, list, add, check, info)
  providers  Show available execution providers (CPU, CUDA, etc.)
  species    Generate species list from range filter
  help       Print help information

Arguments:
  [INPUTS]...  Input files or directories to analyze

Options:
  -m, --model <MODEL>           Model name from configuration
  -f, --format <FORMAT>         Output formats (csv,raven,audacity,kaleidoscope,json,parquet)
      --output-mode <MODE>      CLI output mode (human,json,ndjson)
  -o, --output-dir <DIR>        Output directory (default: same as input)
  -c, --min-confidence <VALUE>  Minimum confidence threshold (0.0-1.0)
  -b, --batch-size <SIZE>       Inference batch size
      --overlap <SECONDS>       Segment overlap in seconds (finite, non-negative)
      --bat <REGION>            Enable bat detection with a regional classifier
      --gpu                     Enable CUDA GPU acceleration
      --cpu                     Force CPU inference
      --force                   Reprocess files even if output exists
      --fail-fast               Stop on first error
  -q, --quiet                   Suppress progress output
      --no-progress             Disable progress bars (useful for scripting/logging)
      --no-csv-bom              Disable UTF-8 BOM in CSV output
  -v, --verbose                 Increase verbosity (-v, -vv, -vvv)
  -h, --help                    Print help
  -V, --version                 Print version
```

### Performance and Progress

The CLI displays detailed timing and performance metrics:

- Total processing time for batch operations
- Per-file processing time
- Performance metrics (segments/sec per file and overall)
- Clear indication of which device (CPU/GPU) is being used
- Optional progress bars showing file and segment processing status

**Progress Bar Control:**

- `--no-progress` - Disable progress bars (useful for scripting/logging)

Progress bars are enabled by default for interactive use but automatically disabled in quiet mode (`--quiet`).

**Example output:**

```text
INFO birda: Found 1 audio file(s) to process
INFO birda: Loading model: perch-v2
INFO birda::inference::classifier: Auto mode: using CPU (use --gpu to force CUDA)
INFO birda::inference::classifier: Loaded model: PerchV2, sample_rate: 32000, segment_duration: 5s, device: CPU
INFO birda::pipeline::processor: Processing: recording.wav
INFO birda::pipeline::processor: Found 10800 detections above 80.0% confidence
INFO birda::pipeline::processor: Processed 2160 segments in 12.35s (174.9 segments/sec)
INFO birda: Complete: 1 processed, 0 skipped, 0 errors, 10800 total detections in 12.48s
INFO birda: Performance: 173.1 segments/sec overall
```

**For headless/scripted usage:**

```bash
birda --no-progress --quiet recording.wav
```

### Model Management

```bash
# List models available for download
birda models list-available

# Install a model (downloads automatically)
birda models install birdnet-v24
birda models install perch-v2

# List configured models
birda models list

# Show model details
birda models info <name>

# Verify model files exist
birda models check

# Add a model manually (advanced)
birda models add <name> --path <model.onnx> --labels <labels.txt> --type <type> [--default]
# Supported types: birdnet-v24, birdnet-v30, perch-v2
```

### Configuration Management

```bash
# Create default config file
birda config init

# Show current configuration
birda config show

# Print config file path
birda config path
```

## Configuration

Configuration file location:
- **Linux**: `~/.config/birda/config.toml`
- **macOS**: `~/Library/Application Support/birda/config.toml`
- **Windows**: `%APPDATA%\birda\config\config.toml`

### Example Configuration

```toml
[models.birdnet]
path = "/path/to/birdnet.onnx"
labels = "/path/to/BirdNET_GLOBAL_6K_V2.4_Labels.txt"
type = "birdnet-v24"

[defaults]
model = "birdnet"
min_confidence = 0.1
overlap = 0.0
formats = ["csv"]
batch_size = 1

[defaults.csv_columns]
include = []

[inference]
device = "auto"  # auto, gpu, or cpu

[output]
combined_prefix = "BirdNET"
```

### Configuration Validation

An analysis run validates the configuration file before it starts, so a bad value is reported once, up front, instead of turning into odd results later in the run. The rules cover `min_confidence` and `range_threshold` (both 0.0 to 1.0), `overlap` (finite and non-negative), `batch_size` (1 to 512), `day_of_year` (1 to 366), `latitude` (-90.0 to 90.0), `longitude` (-180.0 to 180.0), `formats` (at least one output format), `csv_columns.include` (only names birda recognises), and `defaults.model`, which must name a model that exists in the file.

For `overlap` the rule also applies to the command-line flag and the environment variable, not just to the file: `--overlap`, `BIRDA_OVERLAP` and `defaults.overlap` are three routes to one setting, and all three reject a negative, NaN or infinite value. Previously only the file did, and the other two silently became zero overlap. `min_confidence`, `range_threshold`, `latitude`, `longitude`, `batch_size` and `day_of_year` agree across their routes too.

`batch_size` and `day_of_year` were the last two to disagree, and in both cases the config file was the route without the rule. The 512 cap on `batch_size` exists to keep a run from exhausting GPU memory, and it applied to `--batch-size` and `BIRDA_BATCH_SIZE` but not to the file, so a larger value hand-edited into `config.toml` went straight to the inference path. `day_of_year` was bounded on the flag and the environment variable, and `birda config set defaults.day_of_year` did not exist, so editing the file was the only way to set the stored default and the only route with nothing checking it.

`formats` and `csv_columns.include` are the two keys where the config file is the only route that can carry a bad value. `--format` and `BIRDA_FORMAT` do reach `formats`, and win over the file, but they go through a fixed list of names, so neither can produce an empty or unknown one; `csv_columns.include` has no command-line route at all. Neither key has a `birda config set` arm either, so a hand edit was the only way to set either one and the only route with nothing checking it. An empty `formats` was the more expensive of the two: an analysis run asked whether the outputs already existed, got "yes" for a list with nothing in it, and so treated every input file as already done. It logged `Skipping (output exists)` against files that had no output, finished with `0 processed`, and exited 0 having written nothing. An unrecognised name in `csv_columns.include` was quieter and inconsistent between formats: CSV grew a column that was empty in every row, Parquet dropped it.

If you already have a `config.toml` with `batch_size` above 512 or a `day_of_year` outside 1 to 366, this release starts refusing it: an analysis run stops with an error naming the key, and so does any command that saves the configuration. Bring the value into range to get going again:

```bash
birda config set defaults.batch_size ""    # or a size in 1-512; empty restores the smart default
birda config set defaults.day_of_year ""   # or a day in 1-366; empty restores auto-detection
```

An empty `formats` and an unrecognised CSV column are refused the same way, and neither has a `config set` arm, so both are fixed by editing `config.toml`: give `formats` at least one of `csv`, `raven`, `audacity`, `kaleidoscope`, `json` or `parquet`, and leave `csv_columns.include` holding only names from `lat`, `lon`, `week`, `model`, `overlap`, `sensitivity`, `min_conf` and `species_list`. The error names the offending key, lists the accepted values where there is a list, and points at `birda config path` for the file.

Validation reads the file as a document, so a stored value is checked whether or not the run would have used it. `--format csv` does not get you past an empty `formats`, `--stdout` does not get you past it either even though that mode writes no files, and a `csv_columns.include` typo stops a `--format json` run that would never have opened a CSV writer. That is how every rule here has always behaved: a `batch_size` of 9999 in the file stops `birda --batch-size 32` too. The upside is that you learn the file is broken on the next run rather than on the day you leave the flag off; the cost is that the repair is not optional. For a `--stdout` user the repair is free, since naming a format writes no file in that mode anyway.

If more than one value is out of range, each write is rejected by the other fault and you will need to edit `config.toml` directly; that limitation is described below.

```text
$ birda recording.wav
error: invalid latitude: 200 (must be -90.0 to 90.0)
```

Only analysis is gated, because it is the only command that turns the whole of `[defaults]` into a result. Everything else keeps working, which matters because those are the commands you need in order to fix the file. `config show` prints a config that fails validation so you can see which value is wrong, `config set` rewrites it, and `models list`, `models check` and `models install` all stay reachable.

```bash
birda config show                          # works, shows the bad value
birda config set defaults.latitude 60.17   # rewrites it
birda config set defaults.model ""         # clears a default naming a model you no longer have
```

That last one is worth knowing. If `defaults.model` names a model that is not in the file, clearing it is the fix; installing some *other* model will not help, because saving validates the whole configuration and the dangling name is still there.

One limitation worth knowing: `config set` validates the whole file before saving, so it cannot repair a config with two independent bad values one key at a time. Each write is rejected by the other fault. Fix those by editing `config.toml` directly.

Writing is validated too. `config init`, `config set` and the `models` commands all refuse to save a configuration that would not load, and they refuse before touching the file, so a rejected write leaves the existing configuration intact.

Writes are also atomic. The new configuration goes to a temporary file beside `config.toml` and is renamed over it, so an interrupted write cannot leave a truncated file. That matters because a truncated `config.toml` still parses: an empty file is valid TOML and loads as all-defaults, which would silently drop every model you had configured. Three consequences:

- A `config.toml` that is a **symlink** is followed, and the file it points at is the one rewritten, so keeping it in a dotfiles repository works. This holds whether or not the target exists yet, so you can create the link first and let birda create the file.
- A `config.toml` that is a **hardlink** is not, because a rename gives the path a new inode. The other name keeps the old contents and stops tracking.
- If you run birda in a container, bind-mount the config **directory** rather than the `config.toml` file itself. Renaming over a bind-mounted file fails with `EBUSY`.

A configuration file birda creates for the first time is readable only by you (mode 0600 on Unix). An existing file keeps whatever mode you gave it.

`registry.json`, in the same directory, is written the same way, so the hardlink note above applies to it too, with three differences worth knowing. It is **not** created private: it holds the model catalogue, which ships inside the binary and is not secret, so it keeps whatever your umask gives it, exactly as before. A `registry.json` that is a **dangling** symlink is replaced by a regular file rather than written through, because only `config.toml` resolves a link whose target does not exist yet (an existing link is followed for both). And if it cannot be written at all, because the directory is read-only or the file itself is bind-mounted, birda warns and carries on with the built-in registry rather than failing the command; it will try again on the next run.

It is rewritten whenever an upgrade ships a newer bundled registry, so any local edits to it are replaced at that point rather than merged.

### Environment Variables

All options can be set via environment variables:

| Variable | Description |
|----------|-------------|
| `BIRDA_MODEL` | Default model name |
| `BIRDA_MODEL_PATH` | Path to ONNX model file |
| `BIRDA_LABELS_PATH` | Path to labels file |
| `BIRDA_FORMAT` | Output formats (comma-separated) |
| `BIRDA_OUTPUT_DIR` | Output directory |
| `BIRDA_MIN_CONFIDENCE` | Minimum confidence threshold |
| `BIRDA_OVERLAP` | Segment overlap in seconds (finite, non-negative) |
| `BIRDA_BATCH_SIZE` | Inference batch size |
| `BIRDA_OUTPUT_MODE` | CLI output mode (human, json, ndjson) |

## Output Formats

### CSV (default)

Standard CSV with columns: `Start (s)`, `End (s)`, `Scientific name`, `Common name`, `Confidence`, `File`.

CSV files include a UTF-8 BOM (Byte Order Mark) by default for proper encoding detection in Excel on Windows. Use `--no-csv-bom` to disable for compatibility with applications that don't handle BOM.

```csv
Start (s),End (s),Scientific name,Common name,Confidence,File
0.0,3.0,Glaucidium passerinum,Eurasian Pygmy Owl,0.9237,recording.wav
3.0,6.0,Glaucidium passerinum,Eurasian Pygmy Owl,0.9849,recording.wav
```

### Parquet

Apache Parquet columnar format for efficient data storage and analysis. Provides 50-80% file size reduction compared to CSV with native support in data science tools (Pandas, Polars, DuckDB).

**Benefits:**

- **Compact**: 50-80% smaller than CSV for large datasets
- **Type-safe**: Native typed columns (Float32, String) eliminate parsing errors
- **Fast queries**: Columnar format enables efficient filtering without loading entire dataset
- **Ecosystem**: First-class support in Pandas, Polars, DuckDB, Arrow, Spark
- **Self-documenting**: Schema and column types embedded in file format

```bash
# Single format
birda -f parquet recording.wav

# Multiple formats
birda -f csv,parquet recording.wav

# With metadata columns
birda -f parquet --lat 45.0 --lon -73.0 --week 24 recording.wav
```

**Column Schema:**

- Core: `start_s`, `end_s`, `scientific_name`, `common_name`, `confidence`, `file`
- Optional metadata: `lat`, `lon`, `week`, `model`, `overlap`, `sensitivity`, `min_conf`, `species_list`

**Reading Parquet files:**

```python
import pandas as pd
df = pd.read_parquet('recording.BirdNET.results.parquet')
print(df.head())
```

```python
import polars as pl
df = pl.read_parquet('recording.BirdNET.results.parquet')
print(df.describe())
```

```sql
-- DuckDB
SELECT species, COUNT(*) FROM 'recording.BirdNET.results.parquet'
GROUP BY species ORDER BY COUNT(*) DESC;
```

### Raven Selection Table

Compatible with [Raven Pro](https://ravensoundsoftware.com/) audio analysis software.

### Audacity Labels

Tab-separated format for import into [Audacity](https://www.audacityteam.org/).

### Kaleidoscope CSV

Compatible with [Wildlife Acoustics Kaleidoscope](https://www.wildlifeacoustics.com/products/kaleidoscope) software.

### JSON

Structured JSON output with metadata and summary statistics. Use `-f json` to generate `.BirdNET.json` files:

```bash
birda -f json recording.wav
```

## JSON Output for Programmatic Use

Birda supports structured JSON output for integration with GUIs, web applications, and automation scripts.

### CLI Output Mode

Use `--output-mode` to get machine-readable output:

```bash
# Buffered JSON (single object at completion)
birda --output-mode json config show
birda --output-mode json models list

# Streaming NDJSON (one event per line, for real-time progress)
birda --output-mode ndjson recording.wav
```

### Stdout Mode (GUI Integration)

Output detection results as NDJSON stream for integration with GUI applications:

```bash
birda --stdout audio.wav
```

Output format: Each line is a JSON object with event type and payload. Progress updates and detection results are streamed in real-time.

**Constraints:**

- Only one input file allowed
- Cannot combine with `--output-dir`, `--combine`, or `--format`
- Progress bars automatically disabled (NDJSON mode)

### Example: Real-Time Progress

```bash
birda --output-mode ndjson recording.wav 2>/dev/null
```

Outputs events like `pipeline_started`, `file_started`, `progress`, `file_completed`, and `pipeline_completed` - ideal for progress bars in GUI applications.

### Environment Variable

```bash
export BIRDA_OUTPUT_MODE=json
```

**See [JSON Output Guide](docs/json-output.md) for complete documentation including payload schemas, integration examples, and error handling.**

## Bat Detection

Birda supports bat species detection using [BattyBirdNET](https://github.com/rdz-oss/BattyBirdNET-Analyzer) regional classifiers. This uses a two-stage pipeline: BirdNET v2.4 extracts audio embeddings, then a regional bat classifier identifies bat species from those embeddings.

### How It Works

Bat echolocation calls are ultrasonic (20-120 kHz). BattyBirdNET exploits a "slow-down trick": 256 kHz bat recordings are fed directly to BirdNET without resampling. BirdNET's spectrogram pipeline (trained on 48 kHz bird audio) treats the samples as 48 kHz, shifting ultrasonic frequencies into the audible range where its learned features can extract useful embeddings. Regional bat classifiers then map these 1024-dim embeddings to bat species.

### Prerequisites

1. **BirdNET v2.4 with embeddings**: A patched model that exposes the embedding layer. Create it with [birdnet-onnx-converter](https://github.com/tphakala/birdnet-onnx-converter):

   ```bash
   python expose_embeddings.py --input birdnet-v24.onnx --output birdnet-v24-embeddings.onnx
   ```

2. **Regional bat classifier models**: ONNX models converted from BattyBirdNET. Place them in the birda models directory:
   - Linux: `~/.local/share/birda/models/bat/`
   - macOS: `~/Library/Application Support/birda/models/bat/`
   - Windows: `%APPDATA%\birda\models\bat\`

### Usage

```bash
# Analyze bat recordings with the Bavaria classifier
birda -m birdnet-v24-embeddings --bat bavaria bat_recording.wav

# Other available regions
birda -m birdnet-v24-embeddings --bat uk bat_recording.wav
birda -m birdnet-v24-embeddings --bat eu bat_recording.wav
```

### Available Regions

| Region | Flag | Species | Coverage |
|--------|------|---------|----------|
| Bavaria | `--bat bavaria` | 32 | Germany, Central Europe |
| Bavaria (high confidence) | `--bat bavaria-high` | 24 | Germany, stricter thresholds |
| EU | `--bat eu` | 30 | Broad European coverage |
| Scotland | `--bat scotland` | 11 | Scotland |
| South Wales | `--bat south-wales` | 29 | South Wales |
| Sweden | `--bat sweden` | 23 | Sweden, Nordic |
| UK | `--bat uk` | 20 | United Kingdom |
| USA | `--bat usa` | 38 | United States (full) |
| USA East | `--bat usa-east` | 23 | Eastern United States |
| USA East (high confidence) | `--bat usa-east-high` | 17 | Eastern US, stricter thresholds |
| USA West | `--bat usa-west` | 28 | Western United States |

### Audio Requirements

- **Sample rate**: 256 kHz (standard for bat recording devices like AudioMoth)
- **Format**: WAV, FLAC, or MP3
- Birda will warn if the source audio is not 256 kHz but will still attempt analysis

### Notes

- Bat mode overrides segment duration to 0.5625s (144,000 samples at 256 kHz) with 25% overlap
- The backbone model must be BirdNET v2.4 with the embedding output exposed
- All standard output formats are supported (CSV, Raven, Audacity, JSON, Parquet, Kaleidoscope)

## Performance Tips

### GPU vs CPU

- **TensorRT**: Fastest option when available; optimal batch size 16-32
- **CUDA**: Good performance with batch sizes 128-256
- **CPU inference**: Uses AVX2/AVX-512 acceleration automatically; batch size 8 recommended

### Batch Size Guidelines

| Scenario | Recommended Batch Size |
|----------|------------------------|
| CPU inference | 8 |
| CUDA | 256 |
| TensorRT | 32 |

### Example Performance (BirdNET v2.4)

**Test system:** Intel Core i7-13700K, NVIDIA RTX 5080 (16GB VRAM), Windows 11 Pro

**Test file:** 12+ hours of audio (44739s, 14913 segments)

| Device | Batch Size | Time | Segments/sec | Realtime | Speedup |
|--------|------------|------|--------------|----------|---------|
| CPU | 8 | 81.7s | 183 | 547x | 1x |
| CUDA | 64 | 11.3s | 1323 | 3970x | 7.2x |
| CUDA | 128 | 9.7s | 1537 | 4610x | 8.4x |
| CUDA | 256 | 9.1s | 1636 | 4906x | 9.0x |
| TensorRT | 32 | 4.2s | 3589 | 10767x | **19.6x** |
| TensorRT | 64 | 5.0s | 3000 | 9000x | 16.4x |
| TensorRT | 128 | 5.4s | 2765 | 8295x | 15.1x |

**Key findings:**

- TensorRT batch 32 is optimal: **~20x faster** than CPU, over 10000x realtime
- CUDA batch 256 is optimal for CUDA: 9x faster than CPU
- TensorRT is ~2.2x faster than CUDA at optimal settings
- TensorRT engine caches after first run (~120ms load time)
- **Batch size behavior:** TensorRT performs best with small batches (16-32) while CUDA needs large batches (256) for peak performance
- **VRAM considerations:** TensorRT's small batch efficiency makes it ideal for GPUs with limited VRAM
- **Note:** TensorRT requires an NVIDIA GPU with compute capability 5.0+ (GTX 10-series and newer); optimal batch sizes may vary by GPU model

### Example Performance (Perch V2)

**Test system:** Intel Core i7-13700K, NVIDIA RTX 5080 (16GB VRAM), Windows 11 Pro

**Test file:** 12+ hours of audio (44739s, 8948 segments at 5s each)

| Device | Batch Size | Time | Segments/sec | Realtime | Speedup |
|--------|------------|------|--------------|----------|---------|
| CPU | 8 | 215.4s | 42 | 208x | 1x |
| CUDA | 32 | 17.4s | 515 | 2550x | **12.4x** |

**Key findings:**

- Perch V2 requires more VRAM; batch size 32 recommended for GPU
- CUDA provides **12x speedup** over CPU
- CPU inference is ~4x slower than BirdNET due to larger model
- **Note:** TensorRT is not supported for Perch V2 at this time

## Supported Audio Formats

- WAV (PCM)
- MP3
- FLAC
- AAC

Audio is automatically resampled to the model's required sample rate (48kHz for BirdNET).

## Building from Source

### Development

```bash
# Run all checks
task check

# Format code
task fmt

# Run clippy linter
task clippy

# Run tests
task test

# Build debug version
task build

# Build release version
task build:release
```

### Cross-Compilation

```bash
# Linux ARM64 (CPU-only)
task build:linux-arm64

# Windows x64 (CPU-only)
task build:windows-x64

# macOS ARM64 (CPU-only)
task build:macos-arm64
```

## Models

Models can be installed automatically using `birda models install <model-id>`.

### BirdNET v2.4 (Recommended)

```bash
birda models install birdnet-v24
```

- **License**: CC-BY-NC-SA-4.0 (non-commercial use only)
- **Vendor**: Cornell Lab of Ornithology & Chemnitz University of Technology
- **Sample rate**: 48kHz
- **Segment duration**: 3 seconds
- **Species**: ~6,000 bird species globally
- **Range filtering**: Supported (BirdNET Geomodel v3.0.2, 6,217 of 6,522 species covered)
- **Source**: [BirdNET-onnx on Hugging Face](https://huggingface.co/justinchuby/BirdNET-onnx) (optimized ONNX conversion by Justin Chu)

### BSG Finnish Birds v4.4

```bash
birda models install bsg-fi-v44
```

- **License**: BSG-NC-1.0 (non-commercial use only, no app stores)
- **Vendor**: University of Jyväskylä
- **Sample rate**: 48kHz
- **Segment duration**: 3 seconds
- **Species**: 265 Finnish bird species (breeders, migrants, vagrants)
- **Architecture**: Fine-tuned BirdNET model with custom classification head
- **Post-processing**: Automatic calibration + optional Species Distribution Model (SDM)
- **Range filtering**: Not supported (uses BSG SDM instead)
- **Source**: [BSG on Hugging Face](https://huggingface.co/tphakala/BSG)
- **Citation**: Nokelainen et al. (2024) [doi:10.5334/cstp.710](https://doi.org/10.5334/cstp.710)

The BSG model is optimized for bird sound identification in Finland. It uses a BirdNET-based feature extractor combined with a custom classification head trained on Finnish soundscapes, expert-annotated clips from Xeno-canto, and targeted field recordings.

**Post-processing:**

1. **Calibration (always applied)**: Per-species logistic regression (Platt scaling) to improve probability estimates
2. **Species Distribution Model (optional)**: Filters predictions by seasonal and geographic plausibility using migration curves and distribution maps

**Usage with SDM (recommended for field recordings in Finland):**

```bash
# With location and explicit date
birda recording.wav -m bsg-fi-v44 --lat 60.17 --lon 24.94 --day-of-year 150

# With location only (date auto-detected from file timestamp)
birda recording.wav -m bsg-fi-v44 --lat 60.17 --lon 24.94

# Calibration only (no geographic/seasonal filtering)
birda recording.wav -m bsg-fi-v44
```

**CLI options for BSG:**

- `--lat` - Latitude for SDM filtering
- `--lon` - Longitude for SDM filtering
- `--day-of-year` - Day of year (1-366), auto-detected from file modification time if not provided

**Notes:**

- Geomodel range filtering (`--slist`, `--week`, `--month`) is **not compatible** with BSG models due to different species sets
- SDM filtering improves precision by reducing false positives from non-occurring species
- Day-of-year auto-detection uses file modification timestamp when `--day-of-year` is omitted

### Google Perch v2

```bash
birda models install perch-v2
```

- **License**: Apache-2.0
- **Vendor**: Google Research
- **Sample rate**: 32kHz
- **Segment duration**: 5 seconds
- **Range filtering**: Supported (BirdNET Geomodel v3.0.2, 11,145 of 14,795 species covered)
- **Source**: [Perch-onnx on Hugging Face](https://huggingface.co/justinchuby/Perch-onnx) (ONNX conversion by Justin Chu)

### BirdNET v3.0

```bash
birda models install birdnet-v30
```

> **Note**: these are **developer preview** weights (`3.0-preview3.1`), not a GA release. The repository's `TERMS_OF_USE.txt` adds restrictions beyond the licence.

- **License**: CC BY-SA 4.0 (commercial use permitted, unlike v2.4)
- **Vendor**: Cornell Lab of Ornithology & Chemnitz University of Technology
- **Sample rate**: 32kHz
- **Segment duration**: 5 seconds
- **Species**: 11,560 classes globally, fewer per region
- **Model type**: `birdnet-v30`
- **Source**: [BirdNET-v3.0-Models on Hugging Face](https://huggingface.co/tphakala/BirdNET-v3.0-Models)

### Regional models

BirdNET v3.0 and Perch v2 both publish 39 region-sliced models alongside the global one. A regional model scores only the species of that region, which cuts peak memory by roughly two thirds and latency by 15 to 30 percent. It is numerically identical to the global model on the species it keeps, so accuracy is unchanged for those species.

```bash
birda models regions birdnet-v30              # browse the tiles, grouped by continent
birda models install birdnet-v30 --region nordic
birda -m birdnet-v30-nordic recording.wav
```

A regional install is registered under `<model-id>-<region>`, so a global and a regional model coexist and both stay selectable with `-m`.

### Choosing a variant

Each model publishes several files for different hardware: `fp32` and `fp16` for BirdNET v3.0, `no-dft-fp32` and `int8-arm` for Perch v2. birda picks one at install time from the configured inference device and the GPU libraries present on the system, prints which it chose and why, and accepts an override:

```bash
birda models install birdnet-v30 --variant fp16
```

Auto-selection is deliberately conservative. It picks a narrow variant only on a definite signal, and otherwise installs the family default, which every backend supports. On a memory-constrained CPU device prefer a regional `fp32` over a global `fp16`: fp16 is not a CPU memory saving.

### Downloading through a mirror

Set `HF_ENDPOINT` to use a Hugging Face mirror, for networks where huggingface.co is unreachable:

```bash
HF_ENDPOINT=https://hf-mirror.com birda models install birdnet-v30
```

### BirdNET Geomodel v3.0.2

```bash
birda models install geomodel
```

The shared range filter, used by every classifier. It predicts an occurrence probability for 12,012 species at a given latitude, longitude and week, which birda matches onto your model's labels by scientific name.

- **License**: CC BY-SA 4.0
- **Vendor**: Cornell Lab of Ornithology & Chemnitz University of Technology
- **Species**: 12,012 scored classes, covering birds plus mammals, insects, amphibians and reptiles
- **Size**: 14.7 MB
- **Source**: [BirdNET-Geomodel on Hugging Face](https://huggingface.co/tphakala/BirdNET-Geomodel)

Installed automatically alongside any classifier, or downloaded on first use when you pass `--lat`/`--lon`. See [docs/species-list-usage.md](docs/species-list-usage.md) for coverage details and the `--range-unmatched` option.

Powered by [BirdNET](https://birdnet.cornell.edu/).

### Custom Model Conversion

For converting custom BirdNET classifiers or optimizing models for specific hardware (Raspberry Pi, embedded devices), see [birdnet-onnx-converter](https://github.com/tphakala/birdnet-onnx-converter). This tool supports:

- TFLite to ONNX conversion
- Multiple precision formats: FP32 (GPU/desktop), FP16 (RPi 5, modern GPUs), INT8 (CPU optimization)
- Platform-specific optimizations for ARM devices

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [Birda GUI](https://github.com/tphakala/birda-gui) - Cross-platform graphical interface for Birda

## Acknowledgments

- [BattyBirdNET](https://github.com/rdz-oss/BattyBirdNET-Analyzer) by rdz-oss for bat species detection using BirdNET embeddings
- [BirdNET](https://github.com/birdnet-team/BirdNET-Analyzer) by the K. Lisa Yang Center for Conservation Bioacoustics
- [BSG](https://github.com/luomus/BSG) by the University of Jyväskylä for Finnish bird sound classification
- [birdnet-bsg-fuser](https://github.com/tphakala/birdnet-bsg-fuser) for fusing BirdNET feature extractor with BSG classifier
- [Justin Chu](https://github.com/justinchuby) for converting BirdNET TFLite model to optimized ONNX format
- [birdnet-onnx-converter](https://github.com/tphakala/birdnet-onnx-converter) for custom model conversion and optimization
- [Perch](https://github.com/google-research/perch) by Google Research for bioacoustic analysis
- [ONNX Runtime](https://onnxruntime.ai/) for cross-platform inference
- [Symphonia](https://github.com/pdeljanov/Symphonia) for audio decoding
