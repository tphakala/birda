# OpenVINO on Intel GPUs

birda can run BirdNET and Perch models on an Intel GPU (integrated UHD or Iris Xe graphics, or an Arc discrete card) through Intel's OpenVINO runtime. OpenVINO in birda is for Intel GPUs only. CPU inference stays on ONNX Runtime (`--cpu`, or the default when no GPU provider is available).

```bash
birda openvino devices
birda --openvino recording.wav
birda --openvino --openvino-device dgpu recording.wav
birda --openvino --openvino-device igpu recording.wav
```

`--openvino` never falls back. If the runtime, the GPU, or the model cannot be used, birda stops with an error before processing any file. It does not switch to ONNX Runtime or to another device. `--gpu` and the default auto mode do not use OpenVINO.

## What You Need

birda does not bundle OpenVINO. Install these yourself:

- **Supported OS**: Linux or Windows (macOS is not supported; `--openvino` reports an error there).
- **OpenVINO runtime 2025.1 or newer**: birda loads `openvino_c` at run time, so one birda binary works with whichever OpenVINO release you install.
- **Intel GPU driver and compute runtime**:
  - **Linux**: Install `intel-opencl-icd` (GPUs older than Tiger Lake, such as Skylake through Ice Lake, need `intel-opencl-icd-legacy1`). Add your user to the `render` group (`sudo usermod -aG render $USER`, then log out and back in). Intel Arc cards require Linux kernel 6.2 or newer.
  - **Windows**: Install the latest Intel Graphics Driver for your integrated or Arc GPU.
- **ONNX Runtime**: Required for every birda run, including `--openvino`, because birda compares the GPU output against ONNX Runtime on CPU at startup.
- **An `fp32` model file**: `birda models install` downloads `fp32` files (`birdnet-v24`, `birdnet-plus-v30`, and `perch-v2` `no-dft-fp32`). If you still have an older BirdNET v3.0 preview `fp16` model installed, switch to `birda models install birdnet-plus-v30` (with `--region <region>` if desired).

## Setup

### Linux

If you installed OpenVINO from Debian/Ubuntu packages or Intel's APT repository, the libraries are already on the system loader path and birda finds them automatically.

If you extracted an Intel OpenVINO archive (for example under `/opt/intel/openvino_2026`), either source its `setupvars.sh` in your shell before running birda, or register the runtime and TBB directories permanently in `/etc/ld.so.conf.d/openvino.conf`:

```text
/opt/intel/openvino_2026/runtime/lib/intel64
/opt/intel/openvino_2026/runtime/3rdparty/tbb/lib
```

Then run `sudo ldconfig`.

birda searches for `libopenvino_c.so` in this order and uses the first match:

1. `inference.openvino_path` in `config.toml` (a library file, directory holding it, or install root). If set and invalid, birda stops with an error instead of searching further.
2. `INTEL_OPENVINO_DIR` (install root).
3. `LD_LIBRARY_PATH`.
4. Standard system library directories.
5. The newest `/opt/intel/openvino*` directory.
6. The system dynamic loader search (including `ld.so.conf.d`).

### Windows

Windows requires both the OpenVINO runtime DLL directory (`runtime\bin\intel64\Release`) and its TBB dependency directory (`runtime\3rdparty\tbb\bin`) to be on `PATH`. Setting `INTEL_OPENVINO_DIR` or `inference.openvino_path` alone is not enough on Windows, because Windows will fail to resolve `tbb12.dll` and plugin DLLs unless both directories are on `PATH`.

**Option 1: Intel OpenVINO archive**

1. Extract the OpenVINO archive (for example to `C:\Program Files\Intel\openvino_2026`).
2. Either run `setupvars.bat` in your terminal before running birda, or add both directories to your user `PATH`:
   - `C:\Program Files\Intel\openvino_2026\runtime\bin\intel64\Release`
   - `C:\Program Files\Intel\openvino_2026\runtime\3rdparty\tbb\bin`

**Option 2: Microsoft Store / winget (WindowsApps MSIX package)**

OpenVINO installed through the Microsoft Store or `winget` is placed under `C:\Program Files\WindowsApps\Intel.OpenVINOToolkit.*`, which birda does not auto-discover yet. Add its runtime and TBB folders to `PATH` in PowerShell before running birda:

```powershell
$ov = (Get-ChildItem "C:\Program Files\WindowsApps" -Directory -Filter "Intel.OpenVINOToolkit.*" | Sort-Object Name -Descending | Select-Object -First 1).FullName
$env:PATH = "$ov\runtime\bin\intel64\Release;$ov\runtime\3rdparty\tbb\bin;$env:PATH"
birda openvino devices
```

### Verifying Setup

Run `birda openvino devices` to confirm that birda found the runtime and your Intel GPU(s):

```bash
birda openvino devices
```

This prints the OpenVINO version, what your configured device resolves to, and every usable GPU with its class (`integrated` or `discrete`), PCI address, UUID, memory, and the exact `--openvino-device` values that select it. Add `--output-mode json` for scripts.

## Choosing the GPU

Set the target GPU with `--openvino-device <DEVICE>` or `inference.openvino_device` in `config.toml`:

| Value | Meaning |
|-------|---------|
| `auto` (default) | First Intel discrete GPU, otherwise the first Intel integrated GPU |
| `gpu` | Same as `auto` |
| `igpu`, `dgpu` | First integrated or first discrete Intel GPU |
| `igpu.N`, `dgpu.N` | The Nth (0-indexed) GPU of that class |
| `gpu.N` | The device OpenVINO labels `GPU.N` |
| `pci:0000:03:00.0` | GPU at that PCI address |
| `uuid:<32 hex digits>` | GPU with that device UUID |

There is no CPU value, and Intel NPUs are not supported (`npu` is rejected). Non-Intel GPUs are ignored even if OpenVINO lists them.

On a system with both an integrated and a discrete Intel GPU, `auto` selects the discrete GPU. To pin a specific card reliably across driver or firmware changes, use its `pci:` or `uuid:` value from `birda openvino devices`.

To persist OpenVINO as your default backend in `config.toml`:

```toml
[inference]
device = "openvino"
openvino_device = "dgpu"  # optional: auto, igpu, dgpu, pci:..., uuid:...
# openvino_path = "/opt/intel/openvino_2026"  # optional override
```

## Precision, Verification, and Cache

- **Float32 precision**: birda forces 32-bit float precision on the GPU (overriding OpenVINO's 16-bit GPU default) and stops if the device does not honor it.
- **Startup check**: The first time each batch shape is prepared in a run, birda runs deterministic probe segments through both ONNX Runtime (CPU) and OpenVINO (GPU) and stops with an error if any output differs by more than `0.01`.
- **Batch shapes**: birda compiles two static shapes per run: batch 1 and the run's batch size (default `8`, override with `-b`). Keep `-b 8` for most runs; larger batch sizes increase startup verification time on short files.
- **Compiled-model cache**: Compiled blobs are cached per OpenVINO runtime version under `~/.cache/birda/openvino_cache/` (Linux) or `%LOCALAPPDATA%\birda\cache\openvino_cache\` (Windows), capped at 8 GiB with LRU eviction. Delete the directory to clear the cache.

## Common Errors

| Message starts with | What to do |
|---------------------|------------|
| `OpenVINO runtime not found` | Install OpenVINO 2025.1+, add its runtime/TBB folders to `PATH`/`LD_LIBRARY_PATH`, or set `inference.openvino_path`. |
| `inference.openvino_path is set to ...` | Check that the configured path exists and points to `openvino_c`, its directory, or an OpenVINO install root. |
| `could not load the OpenVINO runtime` | Add both the OpenVINO runtime library directory and its `3rdparty/tbb` directory to `PATH` (Windows) or `LD_LIBRARY_PATH` / `ld.so.conf.d` (Linux). |
| `OpenVINO could not list its devices` | Check that the Intel GPU driver and compute runtime (`intel-opencl-icd` on Linux) are installed. |
| `OpenVINO runtime ... is too old` | Upgrade OpenVINO to 2025.1 or newer. |
| `no permission to use the Intel GPU render node` | On Linux, add your user to the `render` group and log in again. |
| `OpenVINO device '...' is not available` | Run `birda openvino devices` to see available GPUs and valid selectors. |
| `OpenVINO backend: ...` | Requested target is unsupported (NPU, non-Intel GPU, or macOS). |
| `OpenVINO could not load the model` | Ensure you are using an `fp32` model (`birdnet-v24`, `birdnet-plus-v30`, or `perch-v2` `no-dft-fp32`). |
| `OpenVINO on ... gives different results from ONNX Runtime` | Update the Intel GPU driver and OpenVINO runtime, or use `--cpu` / another device. |

## Verified Hardware

Tested on:
- **Linux**: Intel Iris Xe (Gen12, OpenVINO 2025.4.1) and Intel Iris Plus 640 (Gen9, OpenVINO 2026.2)
- **Windows 11**: Dual-GPU system with Intel UHD Graphics 730 (iGPU) and Intel Arc A750 (dGPU), OpenVINO 2026.3.0
