# Birda

[![CI](https://github.com/tphakala/birda/actions/workflows/ci.yml/badge.svg)](https://github.com/tphakala/birda/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.92%2B-blue.svg)](https://www.rust-lang.org/)
[![Sponsor](https://img.shields.io/badge/sponsor-GitHub-pink.svg)](https://github.com/sponsors/tphakala)

A fast, cross-platform CLI tool for bird and wildlife sound detection using [BirdNET](https://github.com/birdnet-team/BirdNET-Analyzer), Google Perch, BSG, and BattyBirdNET AI models.

## Features

- **Multiple AI Models**: BirdNET v2.4, BirdNET+ V3.0 (global + 44 regional models), Google Perch v2 (global + 44 calibrated regional models), BSG Finnish Birds v4.4, and BattyBirdNET bat classifiers (11 regions)
- **GPU Acceleration**: NVIDIA CUDA (bundled in GPU releases), optional TensorRT 10.x, and native OpenVINO for Intel GPUs (UHD, Iris Xe, Arc)
- **Species Filtering**: Dynamic range filtering with BirdNET Geomodel v3.0.2 (12,012 species) or static species list files
- **Clip Extraction**: Extract and merge detected audio clips organized by species
- **Multiple Output Formats**: CSV, Apache Parquet, Raven selection tables, Audacity labels, Kaleidoscope CSV, and JSON
- **Programmatic Output**: Structured JSON and streaming NDJSON modes (`--output-mode`, `--stdout`) for automation and GUI integration
- **Batch Processing**: Process directories with collision-safe output naming and resume support
- **Cross-Platform**: Linux, Windows, and macOS

## Installation

### Pre-built Binaries

Download the latest release from the [Releases](https://github.com/tphakala/birda/releases) page.

- **CUDA bundles** include ONNX Runtime, CUDA, and cuDNN libraries (no separate CUDA Toolkit install required).
- **Windows users**: See the [Windows User Guide](docs/windows-guide.md) for installer and GPU setup instructions.
- **Intel GPU users (OpenVINO)**: See the [OpenVINO Guide](docs/openvino.md).
- **TensorRT and tuning**: See the [GPU and Performance Guide](docs/gpu-and-performance.md).

### From Source

Requires Rust 1.92 or later:

```bash
git clone https://github.com/tphakala/birda.git
cd birda

# Build with CUDA support (default)
cargo build --release

# Build CPU-only
cargo build --release --no-default-features

# Install to ~/.cargo/bin
cargo install --path .
```

## Quick Start

### 1. Install a Model

```bash
# List available models
birda models list-available

# Install BirdNET v2.4 (also downloads the shared geomodel)
birda models install birdnet-v24
```

### 2. Analyze Audio

```bash
# Single file, multiple files, or a directory
birda recording.wav
birda *.wav
birda /path/to/recordings/

# Write all outputs into one directory
birda -o results/ /path/to/recordings/

# Dynamic species range filtering by location and date
birda recording.wav --lat 60.17 --lon 24.94 --week 24

# Extract audio clips from detections
birda clip results.BirdNET.results.csv -c 0.7

# Use NVIDIA GPU or Intel GPU (OpenVINO)
birda --gpu recording.wav
birda --openvino recording.wav

# Bat species detection (requires e.g. `birda models install bat-eu`)
birda --bat eu bat_recording.wav
```

## Models

Install models from the built-in registry with `birda models install <id>`. See the [Models Guide](docs/models.md) for details on regional tiles, variants, licensing, mirrors (`HF_ENDPOINT`), and BSG SDM options, or the [Bat Detection Guide](docs/bat-detection.md) for BattyBirdNET.

| Model | Install ID | Classes | Sample Rate / Window | Regional Tiles | License |
|-------|------------|---------|----------------------|----------------|---------|
| BirdNET v2.4 | `birdnet-v24` | 6,522 | 48 kHz / 3 s | No | CC-BY-NC-SA-4.0 |
| BirdNET+ V3.0 | `birdnet-plus-v30` | 11,725 | 32 kHz / 5 s | 44 regions | Apache-2.0 (+ AUP) |
| Google Perch v2 | `perch-v2` | 14,795 | 32 kHz / 5 s | 44 regions | Apache-2.0 |
| BSG Finnish Birds v4.4 | `bsg-fi-v44` | 265 | 48 kHz / 3 s | Finland | BSG-NC-1.0 |
| BattyBirdNET | `bat-<region>` | 11-38 | 256 kHz / 0.5625 s | 11 regions | CC-BY-NC-SA-4.0 |
| BirdNET Geomodel v3.0.2 | `geomodel` | 12,012 | Range filter | Global | CC-BY-SA-4.0 |

## Usage

Run `birda --help` or `birda <command> --help` for the full list of flags and environment variables.

### Subcommands

| Command | Description |
|---------|-------------|
| `birda [OPTIONS] [INPUTS]...` | Analyze audio files or directories |
| `birda models <action>` | Manage models (`list`, `list-available`, `install`, `remove`, `add`, `check`, `info`, `regions`, `manifest`) |
| `birda config <action>` | Manage configuration (`init`, `show`, `path`, `set`) |
| `birda species` | Generate a static species list from the geomodel range filter |
| `birda clip` | Extract audio clips from detection results |
| `birda providers` | Show available ONNX Runtime execution providers |
| `birda openvino devices` | Inspect the OpenVINO runtime and usable Intel GPUs |
| `birda update` | Check for and install updates from GitHub |

### Common Analysis Options

- `-m, --model <MODEL>`: Configured model name
- `-f, --format <FORMAT>`: Output formats, comma-separated (`csv`, `parquet`, `raven`, `audacity`, `kaleidoscope`, `json`)
- `-o, --output-dir <DIR>`: Output directory (defaults to each input's directory)
- `-c, --min-confidence <VALUE>`: Confidence threshold (`0.0-1.0`, default `0.1`)
- `-b, --batch-size <SIZE>`: Inference batch size (`1-512`; smart default per model and provider)
- `--overlap <SECONDS>`: Segment overlap in seconds
- `--lat` / `--lon` + `--week` (or `--month` + `--day`): Dynamic range filtering
- `--slist <FILE>`: Static species list file
- `--bat <REGION>`: Run BattyBirdNET regional bat classifier
- `--gpu` / `--cpu` / `--cuda` / `--tensorrt` / `--openvino`: Device and provider selection
- `--output-mode <human|json|ndjson>` / `--stdout`: Structured output modes
- `--force`, `--fail-fast`, `--no-progress`, `--no-csv-bom`, `-q`, `-v`, `-y`

## Configuration

Configuration file locations (override both config and data directories with `BIRDA_CONFIG_DIR`):

- **Linux**: `~/.config/birda/config.toml`
- **macOS**: `~/Library/Application Support/birda/config.toml`
- **Windows**: `%APPDATA%\birda\config\config.toml`

Create and inspect your configuration with `birda config init`, `birda config show`, `birda config path`, and `birda config set <key> <value>`.

```toml
[defaults]
model = "birdnet-v24"
min_confidence = 0.1
overlap = 0.0
formats = ["csv"]

[inference]
device = "auto"  # auto, cpu, gpu, cuda, tensorrt, openvino, ...

[output]
default_format = "human"  # human, json, or ndjson
```

See the [Configuration Guide](docs/configuration.md) for all keys, validation rules, atomic write/symlink behavior, `registry.json` handling, and the full environment variable reference.

## Output Formats

| Format | Flag | Suffix |
|--------|------|--------|
| CSV (default, UTF-8 BOM) | `-f csv` | `.BirdNET.results.csv` |
| Apache Parquet | `-f parquet` | `.BirdNET.results.parquet` |
| Raven Selection Table | `-f raven` | `.BirdNET.selection.table.txt` |
| Audacity Labels | `-f audacity` | `.BirdNET.results.txt` |
| Kaleidoscope CSV | `-f kaleidoscope` | `.BirdNET.results.kaleidoscope.csv` |
| JSON | `-f json` | `.BirdNET.json` |

See the [Output Formats Guide](docs/output-formats.md) for schemas, Parquet examples, and naming rules, and the [JSON Output Guide](docs/json-output.md) for `--output-mode json`/`ndjson` and `--stdout`.

## Supported Audio Formats

- **WAV** (PCM, including RF64)
- **FLAC**
- **MP3**
- **AAC**

Audio is automatically resampled to the model's required sample rate (48 kHz for BirdNET v2.4 and BSG; 32 kHz for BirdNET+ V3.0 and Perch v2; no resampling for 256 kHz bat recordings).

## Graphical User Interface

[Birda GUI](https://github.com/tphakala/birda-gui) is a cross-platform Electron desktop application with a detection browser, interactive map, audio waveforms, and model management. Release builds bundle the Birda CLI automatically, so no separate CLI installation is required. Installers for Windows, macOS, and Linux are available on the [Birda GUI releases](https://github.com/tphakala/birda-gui/releases) page.

## Documentation

| Guide | Topics |
|-------|--------|
| [Models Guide](docs/models.md) | BirdNET v2.4, BirdNET+ V3.0, Perch v2, BSG Finnish Birds, Geomodel v3.0.2, regional models, variants, mirrors |
| [Bat Detection Guide](docs/bat-detection.md) | BattyBirdNET setup, 11 regional classifiers, 256 kHz audio requirements |
| [Configuration Guide](docs/configuration.md) | `config.toml`, `config set`, validation and repair, atomic writes, `registry.json`, environment variables |
| [Output Formats Guide](docs/output-formats.md) | CSV, Parquet schema and query examples, Raven, Audacity, Kaleidoscope, JSON |
| [JSON Output Guide](docs/json-output.md) | `--output-mode json`/`ndjson`, `--stdout`, envelope schemas, output naming and skipping |
| [GPU and Performance Guide](docs/gpu-and-performance.md) | Execution providers, CUDA, TensorRT 10.x, batch size defaults, benchmarks |
| [OpenVINO Guide](docs/openvino.md) | Intel GPU setup (Linux and Windows), device selection, startup verification, cache |
| [Species List Usage Guide](docs/species-list-usage.md) | Dynamic range filtering, static species lists, `birda species` |
| [Clip Extraction Guide](docs/clip-extraction.md) | Extracting and merging audio clips with `birda clip` |
| [Windows User Guide](docs/windows-guide.md) | Windows installers, CUDA/TensorRT setup, troubleshooting |

## Development

Using [Task](https://taskfile.dev):

```bash
task check          # Run CI checks (fmt, clippy, unit/integration tests, registry)
task build          # Debug build (with CUDA)
task build:release  # Release build (with CUDA)
task build:cpu      # Release build (CPU-only)
```

Cross-compilation tasks (`task build:linux-arm64`, `task build:windows-x64`, `task build:macos-arm64`) are defined in [Taskfile.yml](Taskfile.yml).

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [Birda GUI](https://github.com/tphakala/birda-gui) - Cross-platform desktop application for Birda

## Acknowledgments

- [BirdNET](https://birdnet.cornell.edu/) ([BirdNET-Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer)) by the K. Lisa Yang Center for Conservation Bioacoustics and Chemnitz University of Technology
- [Perch](https://github.com/google-research/perch) by Google Research
- [BSG](https://github.com/luomus/BSG) by the University of Jyväskylä and [birdnet-bsg-fuser](https://github.com/tphakala/birdnet-bsg-fuser)
- [BattyBirdNET](https://github.com/rdz-oss/BattyBirdNET-Analyzer) by rdz-oss
- [Justin Chu](https://github.com/justinchuby) for the BirdNET ONNX conversion
- [birdnet-onnx-converter](https://github.com/tphakala/birdnet-onnx-converter) for model conversion and optimization
- [ONNX Runtime](https://onnxruntime.ai/) and [OpenVINO](https://docs.openvino.ai/) for inference
- [Symphonia](https://github.com/pdeljanov/Symphonia) for audio decoding
