# OpenVINO on Intel GPUs

birda can run BirdNET and Perch models on an Intel GPU (integrated Iris Xe or UHD graphics, or an Arc discrete card) through Intel's OpenVINO runtime. OpenVINO in birda is for Intel GPUs only. CPU inference stays on ONNX Runtime (`--cpu`, or the default when no GPU provider is available).

```bash
birda --openvino recording.wav
birda --openvino --openvino-device dgpu recording.wav
birda openvino devices
```

`--openvino` never falls back. If the runtime, the GPU or the model cannot be used, birda stops with an error before it processes any file. It does not switch to ONNX Runtime or to another device. `--gpu` and the default auto mode do not use OpenVINO.

## What you need

birda does not ship OpenVINO. Install it yourself:

- **OpenVINO runtime 2025.1 or newer.** birda loads the `openvino_c` library at run time, so a birda binary works with whichever release you install.
- **The Intel GPU driver and compute runtime.** On Linux that is the `intel-opencl-icd` package. GPUs older than Tiger Lake (Skylake to Ice Lake, which includes Kaby Lake Iris Plus) need `intel-opencl-icd-legacy1` instead, because the current packages dropped them.
- **Access to the GPU.** On Linux your user needs access to the render node, which normally means membership of the `render` group: `sudo usermod -aG render $USER`, then log in again. Arc cards want Linux kernel 6.2 or newer.
- **An ONNX Runtime**, as for every birda run. birda still loads it, because the startup check compares the GPU's output with ONNX Runtime on the CPU.
- **The fp32 file of BirdNET v3.0.** The `fp16` file failed the startup check on both Intel GPUs tested, even though birda compiles it at 32-bit precision. `birda models install` picks the `fp32` file when the configured device is `openvino`, but `--openvino` runs whichever file is installed, and an install made for CUDA or TensorRT gets the `fp16` file. birda warns when it finds one. To switch, run `birda models install birdnet-v30 --variant fp32`, with `--region <region>` for a regional model.

Platforms: Linux (tested) and Windows (**untested**, see below). macOS is not supported: `--openvino` reports an error there.

### Linux

With the Debian or Ubuntu packages, or Intel's APT repository, the libraries are installed on the system library path and birda finds them. With Intel's archive, either run its `setupvars.sh` before birda, or add the runtime and TBB directories to the loader path for good, for example in `/etc/ld.so.conf.d/openvino.conf`:

```text
/opt/intel/openvino_2026/runtime/lib/intel64
/opt/intel/openvino_2026/runtime/3rdparty/tbb/lib
```

followed by `sudo ldconfig`.

birda looks for `libopenvino_c.so` in this order and uses the first one it finds:

1. `inference.openvino_path` in the config file (a library file, the directory holding it, or an install root). If this is set and wrong, birda stops with an error and does not look further.
2. `INTEL_OPENVINO_DIR` (an install root).
3. The directories in `LD_LIBRARY_PATH`.
4. The system library directories.
5. The newest `/opt/intel/openvino*` directory.
6. The system loader's own search (this is how an `ld.so.conf.d` entry is found).

The error message lists every place that was searched.

### Windows (untested)

Install the Intel graphics driver and an OpenVINO runtime archive, then run `setupvars.bat` first, or put the runtime and TBB directories (`runtime\bin\intel64\Release` and `runtime\3rdparty\tbb\bin`) on `PATH`. birda searches `PATH`, `INTEL_OPENVINO_DIR`, `inference.openvino_path` and `C:\Program Files\Intel\openvino*`. This path has not been run on Windows hardware yet. Please report what you see.

## Choosing the GPU

`--openvino-device <DEVICE>` (or `inference.openvino_device` in the config file):

| Value | Meaning |
|-------|---------|
| `auto` (default) | The first Intel discrete GPU, else the first Intel integrated GPU |
| `gpu` | The same as `auto` |
| `igpu`, `dgpu` | The first integrated, or the first discrete, Intel GPU |
| `igpu.N`, `dgpu.N` | The Nth (counting from 0) of that class |
| `gpu.N` | The device OpenVINO itself calls `GPU.N` |
| `pci:0000:03:00.0` | The GPU at that PCI address, as `lspci -D` prints it |
| `uuid:<32 hex digits>` | The GPU with that device UUID |

There is no value for the CPU, and the NPU is not supported yet (`npu` is reserved and is refused). GPUs from other vendors are never used, even if OpenVINO lists them.

On a computer with more than one GPU, pin the one you want with `pci:` or `uuid:`. The numbered forms depend on the order in which OpenVINO lists devices, which changes when a GPU is added or removed, or when an integrated GPU is disabled in firmware. `pci:` and `uuid:` do not.

`birda openvino devices` shows the runtime birda found, what your configured device resolves to, and every device OpenVINO lists. Each GPU comes with its class, PCI address, UUID and memory, each device birda will not use with the reason, and each usable GPU with the exact values that select it. `birda openvino devices --openvino-device dgpu` shows what a particular value resolves to. Add `--output-mode json` for a machine-readable form.

## What birda guarantees

- **Float32.** The compiled model runs at 32-bit float precision, which is not the GPU default (16-bit). At 16 bits the detections of BirdNET v2.4 and v3.0 change. birda reads the precision back from the compiled model and stops if the device did not honour it.
- **Checked against ONNX Runtime.** The first time each batch shape is compiled in a run, birda feeds the same deterministic test signal (a different one in every batch row) through ONNX Runtime on the CPU and through the GPU and compares the raw scores and embeddings. If they differ by more than 0.01 anywhere, the run stops with an `OpenVINO ... gives different results from ONNX Runtime` error. This also covers GPUs, drivers and runtime versions nobody has tested.
- **Two compiled shapes.** One for a single segment and one for the run's batch size (default 8, `-b` to change it). A file whose length is known up front and that has at most half a batch of segments is run as single segments; any other partial batch is padded to the batch size.

## The compiled-model cache

Compiling a model for a GPU takes a few seconds, and more for large models. birda keeps the compiled result in its cache directory (`~/.cache/birda/openvino_cache/<runtime version>/` on Linux, `~/Library/Caches` on macOS, `%LOCALAPPDATA%\birda\cache\openvino_cache\` on Windows), so later runs start in a fraction of a second. Each compiled shape of a model takes roughly one to two and a half times the model's size on disk. When the cache grows past 8 GiB, birda deletes the files used longest ago (by access or modification time) at the next start. To clear it, delete the directory.

## Errors

| Message starts with | What it means |
|---------------------|---------------|
| `OpenVINO runtime not found` | No `libopenvino_c` in the searched places. Install OpenVINO, or set `inference.openvino_path`. |
| `inference.openvino_path is set to ...` | The configured path does not exist, or has no runtime in it. |
| `could not load the OpenVINO runtime` | The library was found but the loader refused it, usually because the runtime's own library directory or TBB is not on the loader path. |
| `OpenVINO could not list its devices` | The runtime loaded but listing its devices failed; check the GPU driver and the compute runtime. |
| `OpenVINO runtime ... is too old` | Install 2025.1 or newer. |
| `no permission to use the Intel GPU render node` | Add your user to the `render` group. |
| `OpenVINO device '...' is not available` | The requested device is not in OpenVINO's list. The message lists what OpenVINO found and, on Linux, whether a render node or the compute runtime is missing. |
| `OpenVINO backend: ...` | The request names something unsupported: the NPU, a non-Intel GPU, or this platform. |
| `OpenVINO could not load the model` | OpenVINO could not read, reshape or compile the model; the message carries OpenVINO's reason. |
| `OpenVINO on ... gives different results from ONNX Runtime` | The check described above failed. Use `--cpu`, another device, or update the GPU driver and the OpenVINO runtime. |

## Performance

Throughput depends on the model and on the GPU, and an integrated GPU does not always beat a fast CPU. Measured on 10 minutes of real recordings at batch size 8, against ONNX Runtime on the same computer's CPU:

| Model | Iris Plus 640 (Gen9, 4-core i5, idle host) | Iris Xe (Gen12, 12-thread i7, shared host) |
|-------|--------------------------------------------|---------------------------------------------|
| BirdNET v2.4 | 1.4 to 1.6 times the CPU | about 1.3 times the CPU |
| BirdNET v3.0 | about the same | slower than the CPU |
| Perch v2 | about 1.2 times the CPU | about the same as the CPU |

The Iris Xe host was shared with other workloads, so its CPU figures varied by a factor of up to three between runs. Treat the table as indicative and measure on your own hardware. The batch size changes little on these GPUs: the default of 8 is within a few percent of 1, 4, 16 and 32. A discrete GPU may want a larger batch; that has not been measured.

OpenVINO is therefore mostly worth using when the GPU is idle and the CPU is not: BirdNET v2.4 on a busy computer, or a CPU that is needed for other work. birda never selects it automatically.

Memory: the process holds two compiled shapes of the model, and ONNX Runtime loads its own copy while the startup check runs. Peak resident memory measured at batch size 8: BirdNET v2.4 (67 MB model) 0.5 GB, the BirdNET v3.0 Nordic model (135 MB) 0.9 GB, Perch v2 Nordic (65 MB) 0.8 GB, and the full BirdNET v3.0 model (557 MB) 1.7 GB, which is the figure to compare with your free memory on a small computer.

## Measured platforms

Verified on an Intel Core i7-1260P (Iris Xe, Gen12) with OpenVINO 2025.4.1 and on an Intel Core i5-7260U (Iris Plus 640, Gen9) with OpenVINO 2026.2. Discrete Arc GPUs, multi-GPU computers, the NPU and Windows are not verified.
