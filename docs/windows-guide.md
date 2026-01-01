# Windows User Guide

This guide covers installing and using Birda on Windows, including GPU acceleration setup.

## Table of Contents

- [Installation](#installation)
- [GPU Setup (CUDA)](#gpu-setup-cuda)
- [Quick Start](#quick-start)
- [Troubleshooting](#troubleshooting)

## Installation

### Option 1: Pre-built Binary (Recommended)

1. Download `birda-windows-x64.zip` from the [Releases](https://github.com/tphakala/birda/releases) page
2. Extract to a folder (e.g., `C:\Tools\birda`)
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

## GPU Setup (CUDA)

GPU acceleration provides up to 6x faster inference compared to CPU. This requires an NVIDIA GPU with CUDA support.

### Prerequisites

1. **NVIDIA GPU** with Compute Capability 5.0+ (Maxwell or newer)
2. **NVIDIA Driver** 525.60.13+ (check with `nvidia-smi`)
3. **CUDA Toolkit 12.x** - [Download](https://developer.nvidia.com/cuda-downloads)
4. **cuDNN 9.x** for CUDA 12 - [Download](https://developer.nvidia.com/cudnn) (requires NVIDIA account)

### ONNX Runtime CUDA Setup

Birda uses ONNX Runtime for inference. For GPU support, you need the CUDA execution provider DLLs.

#### Option A: Download from ONNX Runtime

1. Download [ONNX Runtime GPU package](https://github.com/microsoft/onnxruntime/releases) (e.g., `onnxruntime-win-x64-gpu-1.x.x.zip`)
2. Extract and copy these DLLs to the same folder as `birda.exe`:
   - `onnxruntime.dll`
   - `onnxruntime_providers_cuda.dll`
   - `onnxruntime_providers_shared.dll`

#### Option B: Use NuGet Package

```powershell
# Download via NuGet
nuget install Microsoft.ML.OnnxRuntime.Gpu -Version 1.20.0

# Copy DLLs from the package to your birda folder
```

### cuDNN Setup

1. Extract cuDNN archive
2. Copy all DLLs from `bin\` folder to your birda folder:
   - `cudnn64_9.dll`
   - `cudnn_ops64_9.dll`
   - `cudnn_cnn64_9.dll`
   - (and other cudnn*.dll files)

### Verify GPU Setup

```powershell
# Check NVIDIA driver
nvidia-smi

# Test GPU inference
birda --gpu -b 64 your_recording.wav
```

If GPU is working, you'll see inference complete much faster than CPU mode.

## Quick Start

### 1. Initialize Configuration

```powershell
birda config init
```

This creates the config file at `%APPDATA%\birda\config\config.toml`.

### 2. Download BirdNET Model

Download the ONNX model and labels:

- **ONNX Model**: [BirdNET-onnx on Hugging Face](https://huggingface.co/justinchuby/BirdNET-onnx)
  - Download `BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx`
- **Labels**: [BirdNET-Analyzer Releases](https://github.com/kahst/BirdNET-Analyzer/releases)
  - Download `BirdNET_GLOBAL_6K_V2.4_Labels.txt`

Save both files to a permanent location (e.g., `C:\Models\BirdNET\`).

### 3. Add Model to Configuration

```powershell
birda models add birdnet `
  --path "C:\Models\BirdNET\BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx" `
  --labels "C:\Models\BirdNET\BirdNET_GLOBAL_6K_V2.4_Labels.txt" `
  --type birdnet-v24 `
  --default
```

### 4. Analyze Audio Files

```powershell
# Analyze a single file (CPU)
birda recording.wav

# Analyze with GPU acceleration
birda --gpu recording.wav

# Analyze with GPU and optimal batch size
birda --gpu -b 64 recording.wav

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
| CPU inference | 1 (default) |
| GPU with short files | 32-64 |
| GPU with long files | 64-128 |

### Example Performance (RTX 5080, 16GB VRAM)

| Device | Batch Size | Time (1GB WAV) | Speedup |
|--------|------------|----------------|---------|
| CPU | 1 | ~36s | 1x |
| GPU | 1 | ~11s | 3.3x |
| GPU | 64 | ~7s | 5.1x |
| GPU | 128 | ~6s | 6x |

> **Note**: Very large batch sizes (256+) can be slower due to memory overhead.

## Troubleshooting

### "onnxruntime.dll not found"

The ONNX Runtime DLLs are not in your PATH or birda folder.

**Solution**: Copy the ONNX Runtime DLLs to the same folder as `birda.exe`.

### "CUDA provider not available" or falls back to CPU

CUDA dependencies are missing or incompatible.

**Solutions**:
1. Verify CUDA Toolkit is installed: `nvcc --version`
2. Verify cuDNN DLLs are in the birda folder
3. Check NVIDIA driver version: `nvidia-smi`
4. Ensure all ONNX Runtime CUDA DLLs are present

### "Invalid handle" error with GPU

cuDNN version mismatch.

**Solution**: Ensure you have cuDNN 9.x for CUDA 12.x. Download the correct version from NVIDIA.

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
path = "C:\\Models\\BirdNET\\BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx"
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

# Configuration commands
birda config init      # Create config file
birda config show      # Show current config
birda config path      # Print config file path

# Model commands
birda models list      # List configured models
birda models check     # Verify model files exist
birda models info <name>  # Show model details

# Analysis options
birda [OPTIONS] <FILES>
  -m, --model <NAME>        # Use specific model
  -f, --format <FORMATS>    # Output formats (csv,raven,audacity,kaleidoscope)
  -o, --output-dir <DIR>    # Output directory
  -c, --min-confidence <N>  # Confidence threshold (0.0-1.0)
  -b, --batch-size <N>      # Inference batch size
      --overlap <SEC>       # Segment overlap in seconds
      --gpu                 # Enable GPU acceleration
      --cpu                 # Force CPU inference
      --force               # Reprocess existing files
      --fail-fast           # Stop on first error
  -q, --quiet               # Suppress progress output
  -v, --verbose             # Increase verbosity
```
