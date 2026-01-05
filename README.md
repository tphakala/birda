# Birda

[![CI](https://github.com/tphakala/birda/actions/workflows/ci.yml/badge.svg)](https://github.com/tphakala/birda/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.92%2B-blue.svg)](https://www.rust-lang.org/)
[![Sponsor](https://img.shields.io/badge/sponsor-GitHub-pink.svg)](https://github.com/sponsors/tphakala)

A fast, cross-platform CLI tool for bird species detection using [BirdNET](https://github.com/kahst/BirdNET-Analyzer) and Google Perch AI models.

## Features

- **Multiple AI Models**: Support for BirdNET v2.4, BirdNET v3.0, and Google Perch v2 models
- **GPU Acceleration**: Optional CUDA support for faster inference on NVIDIA GPUs
- **Species Filtering**: Dynamic range filtering by location/date or static species list files
- **Multiple Output Formats**: CSV, Raven selection tables, Audacity labels, Kaleidoscope CSV
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

For GPU acceleration, you need:

1. NVIDIA GPU with CUDA support
2. [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads)
3. [cuDNN 9.x](https://developer.nvidia.com/cudnn) for CUDA 12

Copy the ONNX Runtime CUDA DLLs to your executable directory or add them to PATH:
- `onnxruntime.dll`
- `onnxruntime_providers_cuda.dll`
- `onnxruntime_providers_shared.dll`
- `cudnn*.dll` files

### Checking Available Execution Providers

To see which execution providers are available on your system:

```bash
birda providers
```

This shows which backends (CPU, CUDA, TensorRT, etc.) are available at compile-time.

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

## Quick Start

### 1. Initialize Configuration

```bash
birda config init
```

### 2. Add a Model

Download a BirdNET model and labels file, then add it:

```bash
birda models add birdnet \
  --path /path/to/birdnet.onnx \
  --labels /path/to/BirdNET_GLOBAL_6K_V2.4_Labels.txt \
  --type birdnet-v24 \
  --default
```

### 3. Analyze Audio Files

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

## Species Filtering

Birda supports filtering detections by species using two complementary approaches:

### Dynamic Range Filtering

Filter species based on location and date using BirdNET's meta model:

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
  config     Manage configuration
  models     Manage models
  providers  Show available execution providers (CPU, CUDA, etc.)
  species    Generate species list from range filter
  help       Print help information

Arguments:
  [INPUTS]...  Input files or directories to analyze

Options:
  -m, --model <MODEL>           Model name from configuration
  -f, --format <FORMAT>         Output formats (csv,raven,audacity,kaleidoscope)
  -o, --output-dir <DIR>        Output directory (default: same as input)
  -c, --min-confidence <VALUE>  Minimum confidence threshold (0.0-1.0)
  -b, --batch-size <SIZE>       Inference batch size
      --overlap <SECONDS>       Segment overlap in seconds
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
# List configured models
birda models list

# Add a new model
birda models add <name> --path <model.onnx> --labels <labels.txt> --type <type> [--default]

# Supported types: birdnet-v24, birdnet-v30, perch-v2

# Show model details
birda models info <name>

# Verify model files exist
birda models check
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
| `BIRDA_OVERLAP` | Segment overlap in seconds |
| `BIRDA_BATCH_SIZE` | Inference batch size |

## Output Formats

### CSV (default)

Standard CSV with columns: `Start (s)`, `End (s)`, `Scientific name`, `Common name`, `Confidence`, `File`.

CSV files include a UTF-8 BOM (Byte Order Mark) by default for proper encoding detection in Excel on Windows. Use `--no-csv-bom` to disable for compatibility with applications that don't handle BOM.

```csv
Start (s),End (s),Scientific name,Common name,Confidence,File
0.0,3.0,Glaucidium passerinum,Eurasian Pygmy Owl,0.9237,recording.wav
3.0,6.0,Glaucidium passerinum,Eurasian Pygmy Owl,0.9849,recording.wav
```

### Raven Selection Table

Compatible with [Raven Pro](https://ravensoundsoftware.com/) audio analysis software.

### Audacity Labels

Tab-separated format for import into [Audacity](https://www.audacityteam.org/).

### Kaleidoscope CSV

Compatible with [Wildlife Acoustics Kaleidoscope](https://www.wildlifeacoustics.com/products/kaleidoscope) software.

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

### BirdNET v2.4

- **ONNX Model**: [BirdNET-onnx on Hugging Face](https://huggingface.co/justinchuby/BirdNET-onnx) (required - optimized ONNX conversion by Justin Chu)
- **Labels**: Download from [BirdNET-Analyzer Releases](https://github.com/kahst/BirdNET-Analyzer/releases) (`BirdNET_GLOBAL_6K_V2.4_Labels.txt`)
- **Model type**: `birdnet-v24`
- **Sample rate**: 48kHz
- **Segment duration**: 3 seconds
- **Species**: ~6,000 bird species globally

### BirdNET v3.0

> **Note**: BirdNET v3.0 is currently in **developer preview** and not yet available for production use.

- **Model type**: `birdnet-v30`
- **Status**: Developer preview only
- **Regional variants available** (e.g., EUNA for Europe/North America)

### Google Perch v2

- **ONNX Model**: [Perch-onnx on Hugging Face](https://huggingface.co/justinchuby/Perch-onnx) (ONNX conversion by Justin Chu)
- **Model type**: `perch-v2`
- **Sample rate**: 32kHz

## Roadmap

Planned features for future releases:

- [x] **Progress indicators** - Real-time progress bars for batch processing
- [ ] **Parallel file processing** - Process multiple audio files concurrently

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [BirdNET](https://github.com/kahst/BirdNET-Analyzer) by the K. Lisa Yang Center for Conservation Bioacoustics
- [Justin Chu](https://github.com/justinchuby) for converting BirdNET TFLite model to optimized ONNX format
- [Perch](https://github.com/google-research/perch) by Google Research for bioacoustic analysis
- [ONNX Runtime](https://onnxruntime.ai/) for cross-platform inference
- [Symphonia](https://github.com/pdeljanov/Symphonia) for audio decoding
