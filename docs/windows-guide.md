# Windows User Guide

This guide covers installing and using Birda on Windows, including GPU acceleration setup.

## Table of Contents

- [Installation](#installation)
- [GPU Support](#gpu-support)
- [TensorRT Setup (Optional)](#tensorrt-setup-optional)
- [Quick Start](#quick-start)
- [Troubleshooting](#troubleshooting)

## Installation

### Option 1: Pre-built Binary (Recommended)

Download from the [Releases](https://github.com/tphakala/birda/releases) page:

| Package | Size | Description |
|---------|------|-------------|
| `birda-windows-x64-cuda-setup.exe` | ~1 GB | **Recommended** - Installer with bundled CUDA |
| `birda-windows-x64-cuda.zip` | ~1.5 GB | ZIP with bundled CUDA |
| `birda-windows-x64.zip` | ~3.4 MB | Small package (uses system CUDA if available) |

**Installation:**
1. Download the CUDA bundle for easy GPU setup, or the small package if you have CUDA installed
2. Run the installer or extract ZIP to a folder (e.g., `C:\Tools\birda`)
3. Add the folder to your PATH (optional but recommended):
   - Press `Win + X` → "System" → "Advanced system settings"
   - Click "Environment Variables"
   - Under "User variables", select "Path" and click "Edit"
   - Click "New" and add `C:\Tools\birda`
   - Click "OK" to save

### Option 2: Build from Source

Requires [Rust 1.92+](https://rustup.rs/) and [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

```powershell
# Clone the repository
git clone https://github.com/tphakala/birda.git
cd birda

# Build CPU-only version
cargo build --release --no-default-features

# Build with CUDA support (requires CUDA Toolkit)
cargo build --release

# The binary will be at target\release\birda.exe
```

## GPU Support

All packages support GPU acceleration:
- **CUDA bundles**: Include all required libraries - just download and run
- **Small package**: Requires CUDA 12.x and cuDNN 9.x installed on your system

### Requirements

- **NVIDIA GPU** with Compute Capability 5.0+ (GTX 10-series or newer)
- **Up-to-date NVIDIA drivers** (check with `nvidia-smi`)
- For small package: [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) and [cuDNN 9.x](https://developer.nvidia.com/cudnn)

### Verify GPU Setup

```powershell
# Check NVIDIA driver
nvidia-smi

# Test GPU inference
birda --gpu -b 256 your_recording.wav
```

If GPU is working, you'll see logs indicating CUDA is being used and inference will be much faster.

## TensorRT Setup (Optional)

TensorRT provides ~2x additional speedup over CUDA. Unlike CUDA (which is bundled), TensorRT requires manual setup.

### Installation

1. Download [TensorRT 10.x](https://developer.nvidia.com/tensorrt) (requires free NVIDIA Developer account)
2. Extract and add TensorRT `lib` folder to your PATH, or copy DLLs to the birda folder:
   - `nvinfer*.dll`
   - `nvonnxparser*.dll`

### Verify TensorRT Setup

```powershell
# Check available providers
birda providers

# Test TensorRT inference
birda --tensorrt -b 32 your_recording.wav
```

> **Note**: TensorRT engines are cached after first run. Initial inference may take longer while the engine is built.

## Quick Start

### 1. Install a Model

```powershell
# List available models
birda models list-available

# Install BirdNET v2.4 (recommended)
birda models install birdnet-v24
```

The installer will download the model, show license terms, and configure it automatically.

### 2. Analyze Audio Files

```powershell
# Analyze a single file (CPU)
birda recording.wav

# Analyze with GPU acceleration (CUDA)
birda --gpu -b 256 recording.wav

# Analyze with TensorRT (fastest, requires setup)
birda --tensorrt -b 32 recording.wav

# Analyze a folder
birda "C:\Recordings\2024\"

# Analyze with custom confidence threshold
birda -c 0.5 recording.wav

# Output to Raven format
birda -f raven recording.wav

# Multiple output formats
birda -f csv,raven,audacity recording.wav
```

### 5. View Results

Results are saved in the same folder as the input file:

- `recording.BirdNET.results.csv` - CSV format
- `recording.BirdNET.selection.table.txt` - Raven format
- `recording.BirdNET.results.txt` - Audacity format

## Performance Tips

### Optimal Batch Sizes

| Scenario | Recommended |
|----------|-------------|
| CPU inference | 8 |
| CUDA | 256 |
| TensorRT | 32 |

### Example Performance (RTX 5080, 16GB VRAM, BirdNET v2.4)

Test file: 12+ hours of audio (14913 segments)

| Device | Batch Size | Time | Speedup |
|--------|------------|------|---------|
| CPU | 8 | 81.7s | 1x |
| CUDA | 256 | 9.1s | 9x |
| TensorRT | 32 | 4.2s | **~20x** |

> **Note**: TensorRT performs best with small batches (16-32) while CUDA needs large batches (256) for peak performance.

## Troubleshooting

### "CUDA provider not available" or falls back to CPU

GPU drivers may be outdated or incompatible.

**Solutions**:
1. Update NVIDIA drivers to the latest version
2. Check driver version: `nvidia-smi`
3. Verify GPU is detected: `nvidia-smi -L`

### TensorRT not working

TensorRT requires separate installation.

**Solutions**:
1. Verify CUDA Toolkit is installed: `nvcc --version`
2. Verify TensorRT DLLs are in PATH or birda folder
3. Check available providers: `birda providers`

### "Model file not found"

The path in your config is incorrect or the file was moved.

**Solutions**:
1. Check your config: `birda config show`
2. Verify the file exists at the configured path
3. Update the path: `birda models add --path "new\path\model.onnx" ...`

### Slow GPU inference

Batch size may not be optimal.

**Solutions**:
1. Try different batch sizes: `-b 32`, `-b 64`, `-b 128`
2. Check GPU utilization with Task Manager or `nvidia-smi`
3. Close other GPU-intensive applications

### Audio format not supported

**Supported formats**: WAV, MP3, FLAC, AAC

**Solution**: Convert your audio to a supported format using tools like FFmpeg:
```powershell
ffmpeg -i input.ogg -c:a pcm_s16le output.wav
```

## Configuration File

The config file is located at `%APPDATA%\birda\config\config.toml`.

Example configuration:

```toml
[models.birdnet]
path = "C:\\Models\\BirdNET\\birdnet.onnx"
labels = "C:\\Models\\BirdNET\\BirdNET_GLOBAL_6K_V2.4_Labels.txt"
type = "birdnet-v24"

[defaults]
model = "birdnet"
min_confidence = 0.1
batch_size = 1
formats = ["csv"]

[inference]
device = "auto"  # auto, gpu, or cpu
```

## Command Reference

```powershell
# Show help
birda --help

# Show version
birda --version

# Check available GPU providers
birda providers

# Configuration commands
birda config init      # Create config file
birda config show      # Show current config
birda config path      # Print config file path

# Model commands
birda models list-available  # List models available for download
birda models install <id>    # Download and install a model
birda models list            # List configured models
birda models check           # Verify model files exist
birda models info <name>     # Show model details

# Generate species list
birda species --lat 60.17 --lon 24.94 --week 24 --output species.txt

# Analysis options
birda [OPTIONS] <FILES>
  -m, --model <NAME>        # Use specific model
  -f, --format <FORMATS>    # Output formats (csv,raven,audacity,kaleidoscope)
  -o, --output-dir <DIR>    # Output directory
  -c, --min-confidence <N>  # Confidence threshold (0.0-1.0)
  -b, --batch-size <N>      # Inference batch size
      --overlap <SEC>       # Segment overlap in seconds
      --combine             # Generate combined results file
      --force               # Reprocess existing files
      --fail-fast           # Stop on first error
  -q, --quiet               # Suppress progress output
      --no-progress         # Disable progress bars
  -v, --verbose             # Increase verbosity

# Device selection
      --gpu                 # Auto-select best GPU (TensorRT → CUDA → ...)
      --cpu                 # Force CPU inference
      --cuda                # Use CUDA explicitly
      --tensorrt            # Use TensorRT explicitly

# Range filtering
      --lat <LAT>           # Latitude (-90.0 to 90.0)
      --lon <LON>           # Longitude (-180.0 to 180.0)
      --week <WEEK>         # Week number (1-48)
      --month <MONTH>       # Month (1-12)
      --day <DAY>           # Day of month (1-31)
      --slist <FILE>        # Path to species list file
```
