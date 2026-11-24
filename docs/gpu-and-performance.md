# GPU Acceleration and Performance Guide

Birda runs inference through ONNX Runtime (CPU, CUDA, TensorRT, and other ONNX Runtime execution providers) and through a native OpenVINO backend for Intel GPUs.

## Execution Providers

List the ONNX Runtime providers compiled into your binary:

```bash
birda providers
birda providers --output-mode json
```

Example JSON output:

```json
{
  "spec_version": "1.1",
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

### Selecting a Device

| Flag / `inference.device` | Behavior |
|---------------------------|----------|
| *(default)* / `auto` | Tries GPU providers in priority order (`TensorRT` -> `CUDA` -> `DirectML` -> `CoreML` [non-macOS] -> `ROCm`), falling back silently to CPU if unavailable |
| `--gpu` / `gpu` | Same priority order as `auto`, but logs warnings if GPU libraries or providers are unavailable and falls back to CPU |
| `--cpu` / `cpu` | Forces CPU inference |
| `--cuda` / `cuda` | Explicit NVIDIA CUDA (fails if unavailable) |
| `--tensorrt` / `tensorrt` | Explicit NVIDIA TensorRT (fails if unavailable) |
| `--openvino` / `openvino` | Explicit native OpenVINO on Intel GPU (never falls back; see [OpenVINO Guide](openvino.md)) |
| `--directml`, `--coreml`, `--rocm`, `--onednn`, `--qnn`, `--acl`, `--armnn`, `--xnnpack` | Explicit ONNX Runtime provider (fails if unavailable) |

Notes:
- `birda providers` reports compile-time availability; actual runtime use also requires matching drivers and runtime libraries.
- OpenVINO is never selected automatically by `auto` or `--gpu`; it runs only when explicitly requested with `--openvino` or `inference.device = "openvino"`.
- Check the `INFO` log lines during analysis (or `-v`) to verify which device birda selected.

## NVIDIA CUDA Setup

Download the CUDA release bundle from [Releases](https://github.com/tphakala/birda/releases) (for example `birda-linux-x64-cuda-*.tar.gz` or `birda-windows-x64-cuda-setup.exe`). These bundles ship ONNX Runtime 1.24.2 along with the required CUDA 12.9 and cuDNN 9.17 libraries, so no separate CUDA Toolkit installation is needed.

Requirements:
- NVIDIA GPU with CUDA support
- Up-to-date NVIDIA GPU drivers

For Windows-specific setup and troubleshooting, see the [Windows User Guide](windows-guide.md).

## NVIDIA TensorRT Setup (Optional)

TensorRT provides ~2x speedup over CUDA on supported models. It is not bundled due to release archive size limits, but can be added manually:

1. Download **TensorRT 10.x** for CUDA 12.x from [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) (requires a free NVIDIA Developer account). Birda bundles ONNX Runtime 1.24, which links against TensorRT 10 (`nvinfer_10.dll` / `libnvinfer.so.10`), so TensorRT 11.x builds are not compatible.
2. Copy the TensorRT shared libraries into the birda installation directory (or place them on `PATH` / `LD_LIBRARY_PATH`):
   - **Windows**: `nvinfer_10.dll`, `nvinfer_plugin_10.dll`, `nvonnxparser_10.dll`
   - **Linux**: `libnvinfer.so.10`, `libnvinfer_plugin.so.10`, `libnvonnxparser.so.10`
3. Run with `--tensorrt`:
   ```bash
   birda --tensorrt recording.wav
   ```

TensorRT requires an NVIDIA GPU with compute capability 5.0+ (GTX 10-series and newer). On the first run for a given model and batch size, TensorRT compiles and caches an engine in birda's cache directory (`~/.cache/birda/tensorrt_cache/` on Linux, `%LOCALAPPDATA%\birda\cache\tensorrt_cache\` on Windows).

## Intel GPUs (OpenVINO)

Birda includes a native OpenVINO backend for Intel integrated (UHD, Iris Xe) and discrete (Arc) GPUs on Linux and Windows. OpenVINO is not bundled. See the [OpenVINO Guide](openvino.md) for installation, `PATH`/`ldconfig` setup, and device selection.

## Batch Size Defaults and Tuning

When `--batch-size` and `defaults.batch_size` are not set, birda chooses a smart default based on model type and execution provider:

| Execution Provider | Model | Default Batch Size |
|--------------------|-------|--------------------|
| CPU | All models | 8 |
| OpenVINO (Intel GPU) | All models | 8 |
| CUDA | BirdNET v2.4, BSG Finland | 64 |
| CUDA | BirdNET+ V3.0, Perch v2 | 32 |
| TensorRT | All models | 32 |
| Other GPU providers | All models | 16 |

Valid range for `-b`/`--batch-size` is `1` to `512`.

Tuning guidelines:
- **TensorRT**: Performs best with small batches (16-32), which also keeps VRAM usage low.
- **CUDA**: Benefits from larger batches (64-256 for BirdNET v2.4; 32 for larger models like Perch v2).
- **CPU**: Uses AVX2/AVX-512 automatically; default batch size 8 is a good balance.
- **OpenVINO**: Default batch size 8; batching makes little difference on integrated Intel GPUs.

## Progress and Timing Output

During analysis, birda reports per-file and overall throughput (segments/sec and realtime multiplier) and displays interactive progress bars.

```bash
# Headless / scripted usage
birda --no-progress --quiet recording.wav
```

## Benchmarks

### BirdNET v2.4

- **Test system**: Intel Core i7-13700K, NVIDIA RTX 5080 (16 GB VRAM), Windows 11 Pro
- **Test file**: 12+ hours of audio (44,739 s, 14,913 segments at 3 s each)

| Device | Batch Size | Time | Segments/sec | Realtime | Speedup |
|--------|------------|------|--------------|----------|---------|
| CPU | 8 | 81.7 s | 183 | 547x | 1x |
| CUDA | 64 | 11.3 s | 1,323 | 3,970x | 7.2x |
| CUDA | 128 | 9.7 s | 1,537 | 4,610x | 8.4x |
| CUDA | 256 | 9.1 s | 1,636 | 4,906x | 9.0x |
| TensorRT | 32 | 4.2 s | 3,589 | 10,767x | **19.6x** |
| TensorRT | 64 | 5.0 s | 3,000 | 9,000x | 16.4x |
| TensorRT | 128 | 5.4 s | 2,765 | 8,295x | 15.1x |

Key takeaways:
- TensorRT at batch 32 is ~20x faster than CPU and ~2.2x faster than CUDA.
- TensorRT engine loads from cache in ~120 ms after the first run.
- CUDA peaks at larger batch sizes (128-256).

### Perch v2

- **Test system**: Intel Core i7-13700K, NVIDIA RTX 5080 (16 GB VRAM), Windows 11 Pro
- **Test file**: 12+ hours of audio (44,739 s, 8,948 segments at 5 s each)

| Device | Batch Size | Time | Segments/sec | Realtime | Speedup |
|--------|------------|------|--------------|----------|---------|
| CPU | 8 | 215.4 s | 42 | 208x | 1x |
| CUDA | 32 | 17.4 s | 515 | 2,550x | **12.4x** |

Key takeaways:
- Perch v2 uses more VRAM than BirdNET v2.4; batch size 32 is recommended on GPU.
- TensorRT is not currently supported for Perch v2.
