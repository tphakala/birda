# Birda

A fast, cross-platform CLI tool for bird species detection using [BirdNET](https://github.com/kahst/BirdNET-Analyzer) and Google Perch AI models.

## Features

- **Multiple AI Models**: Support for BirdNET v2.4, BirdNET v3.0, and Google Perch v2 models
- **GPU Acceleration**: Optional CUDA support for faster inference on NVIDIA GPUs
- **Multiple Output Formats**: CSV, Raven selection tables, Audacity labels, Kaleidoscope CSV
- **Batch Processing**: Process entire directories of audio files
- **Flexible Configuration**: TOML-based config with CLI overrides
- **Cross-Platform**: Works on Linux, Windows, and macOS

## Installation

### Pre-built Binaries

Download the latest release from the [Releases](https://github.com/tphakala/birda/releases) page.

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

## Quick Start

### 1. Initialize Configuration

```bash
birda config init
```

### 2. Add a Model

Download a BirdNET model and labels file, then add it:

```bash
birda models add birdnet \
  --path /path/to/BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx \
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

## Usage

```
birda [OPTIONS] [INPUTS]... [COMMAND]

Commands:
  config  Manage configuration
  models  Manage models
  help    Print help information

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
  -v, --verbose                 Increase verbosity (-v, -vv, -vvv)
  -h, --help                    Print help
  -V, --version                 Print version
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
path = "/path/to/BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx"
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

Standard CSV with columns: `Start (s)`, `End (s)`, `Scientific name`, `Common name`, `Confidence`, `File`

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

- **Small models (BirdNET v2.4)**: CPU is often faster for small files; GPU shines with large batches
- **Optimal GPU batch size**: 64-128 typically works best; larger batches can be slower due to memory overhead
- **CPU inference**: Uses AVX2/AVX-512 acceleration automatically

### Batch Size Guidelines

| Scenario | Recommended Batch Size |
|----------|------------------------|
| CPU inference | 1 (default) |
| GPU with small files | 32-64 |
| GPU with large files | 64-128 |

### Example Performance

**Test system:** Intel Core i7-13700K, NVIDIA RTX 5080 (16GB VRAM), Windows 11 Pro

**Test file:** 1GB WAV (~17 minutes of audio, ~3600 segments)

| Device | Batch Size | Time | Speedup |
|--------|------------|------|---------|
| CPU | 1 | ~36s | 1x |
| GPU | 1 | ~11s | 3.3x |
| GPU | 64 | ~7s | 5.1x |
| GPU | 128 | ~6s | 6x |
| GPU | 256 | ~36s | 1x (overhead) |

**Key findings:**
- GPU with batch 64-128 is optimal (~6x faster than CPU)
- Very large batch sizes (256+) can be slower due to memory allocation overhead
- GPU with batch=1 is still 3x faster than CPU

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

- **Download**: [BirdNET-Analyzer Releases](https://github.com/kahst/BirdNET-Analyzer/releases)
- **Labels**: `BirdNET_GLOBAL_6K_V2.4_Labels.txt`
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

- **Model type**: `perch-v2`
- **Sample rate**: 32kHz

## Roadmap

Planned features for future releases:

- [ ] **Range Filter** - Geographic and temporal species filtering using BirdNET's meta model to eliminate impossible species based on location and time of year
- [ ] **Progress indicators** - Real-time progress bars for batch processing
- [ ] **Parallel file processing** - Process multiple audio files concurrently

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [BirdNET](https://github.com/kahst/BirdNET-Analyzer) by the K. Lisa Yang Center for Conservation Bioacoustics
- [Perch](https://github.com/google-research/perch) by Google Research for bioacoustic analysis
- [ONNX Runtime](https://onnxruntime.ai/) for cross-platform inference
- [Symphonia](https://github.com/pdeljanov/Symphonia) for audio decoding
