# Configuration Guide

Birda uses a TOML configuration file with environment variable and CLI flag overrides. Precedence from highest to lowest:

1. CLI flags
2. Environment variables
3. Configuration file (`config.toml`)
4. Built-in defaults

## File Locations

| Platform | Config file (`config.toml`) | Data directory (models) | Cache directory |
|----------|-----------------------------|-------------------------|-----------------|
| Linux | `~/.config/birda/config.toml` | `~/.local/share/birda/` | `~/.cache/birda/` |
| macOS | `~/Library/Application Support/birda/config.toml` | `~/Library/Application Support/birda/` | `~/Library/Caches/birda/` |
| Windows | `%APPDATA%\birda\config\config.toml` | `%APPDATA%\birda\data\` | `%LOCALAPPDATA%\birda\cache\` |

Set `BIRDA_CONFIG_DIR` to redirect both the config directory (`config.toml` and `registry.json`) and the data directory (`models/`) to a single custom root on any platform.

## Configuration Commands

```bash
# Create a default configuration file
birda config init

# Show current configuration
birda config show

# Print configuration file path
birda config path

# Set a configuration key (use "" to clear or restore default)
birda config set defaults.model birdnet-v24
birda config set defaults.latitude -33.9
birda config set defaults.batch_size ""
```

### Keys Supported by `birda config set`

| Key | Value / Empty Behavior |
|-----|------------------------|
| `defaults.model` | Configured model name (`""` clears) |
| `defaults.min_confidence` | Float in `0.0-1.0` (`""` restores `0.1`) |
| `defaults.overlap` | Finite non-negative float in seconds (`""` restores `0.0`) |
| `defaults.batch_size` | Integer in `1-512` (`""` restores smart default) |
| `defaults.latitude` | Float in `-90.0` to `90.0` (`""` clears) |
| `defaults.longitude` | Float in `-180.0` to `180.0` (`""` clears) |
| `defaults.day_of_year` | Integer in `1-366` for BSG SDM (`""` restores timestamp auto-detection) |
| `defaults.range_threshold` | Float in `0.0-1.0` (`""` restores `0.01`) |
| `defaults.geomodel` | Path to geomodel ONNX file (`""` clears) |
| `defaults.geomodel_labels` | Path to geomodel labels file (`""` clears) |
| `defaults.range_unmatched` | `keep` or `drop` (`""` restores `keep`) |

Negative values such as `-33.9` are accepted directly without extra escaping.

## Example `config.toml`

```toml
[models.birdnet-v24]
path = "/path/to/birdnet.onnx"
labels = "/path/to/BirdNET_GLOBAL_6K_V2.4_Labels.txt"
type = "birdnet-v24"

[defaults]
model = "birdnet-v24"
min_confidence = 0.1
overlap = 0.0
formats = ["csv"]
# batch_size = 8  # omit to use the smart default per model and provider
# latitude = 60.17
# longitude = 24.94
# range_threshold = 0.01
# range_unmatched = "keep"  # keep or drop
# day_of_year = 150  # BSG SDM only

[defaults.csv_columns]
include = []  # accepted: lat, lon, week, model, overlap, sensitivity, min_conf, species_list

[inference]
device = "auto"  # auto, cpu, gpu, cuda, tensorrt, directml, coreml, rocm, openvino, onednn, qnn, acl, armnn, xnnpack
# openvino_device = "auto"  # auto, gpu, igpu, dgpu, gpu.N, igpu.N, dgpu.N, pci:..., uuid:...
# openvino_path = "/opt/intel/openvino_2026"

[output]
default_format = "human"  # human, json, or ndjson
```

## Configuration Validation

An analysis run validates the entire `config.toml` up front, regardless of whether a CLI flag overrides a particular setting for that run. The rules enforced across `config.toml`, CLI flags, and environment variables are:

- `defaults.min_confidence`: `0.0` to `1.0`
- `defaults.range_threshold`: `0.0` to `1.0`
- `defaults.overlap`: finite and non-negative
- `defaults.batch_size`: `1` to `512` (when set)
- `defaults.day_of_year`: `1` to `366` (when set)
- `defaults.latitude`: `-90.0` to `90.0`
- `defaults.longitude`: `-180.0` to `180.0`
- `defaults.formats`: must contain at least one output format (`csv`, `raven`, `audacity`, `kaleidoscope`, `json`, or `parquet`)
- `defaults.csv_columns.include`: must contain only recognized names (`lat`, `lon`, `week`, `model`, `overlap`, `sensitivity`, `min_conf`, `species_list`)
- `defaults.model`: if set, must name a model present under `[models.*]`
- `models.<name>.softmax_rest_logit`: if set, must be a finite number

### Repairing an Invalid Configuration

Only analysis is gated on full validation. Inspection and management commands (`config show`, `config set`, `models list`, `models check`, `models install`, `models remove`) remain usable so you can inspect and fix the file:

```bash
birda config show                          # displays the invalid values
birda config set defaults.latitude 60.17   # fixes one key
birda config set defaults.model ""         # clears a dangling default model name
```

- **Multiple faults**: `birda config set` allows repairing a broken file one key at a time. It writes the change as long as the edit introduces no new validation error, and prints a warning listing any remaining faults to fix.
- **Keys without `config set`**: `defaults.formats`, `defaults.csv_columns.include`, and `models.<name>.softmax_rest_logit` have no `config set` arm, so fix those by editing `config.toml` directly (`birda config path` shows its location).
- **Other writers**: `config init` and the `models` commands refuse to save an invalid config before touching the file, leaving the existing file intact. If `defaults.model` points at a removed model, clear it with `birda config set defaults.model ""` before installing another model.

## Atomic Writes, Locks, and `registry.json`

All config-modifying commands serialize access with an exclusive lock file (`config.toml.birda.lock`) and write atomically through a temporary file in the same directory before renaming over `config.toml`. This prevents a truncated file from silently loading as empty defaults.

Consequences to be aware of:

- **Symlinks**: If `config.toml` is a symlink (including a dangling symlink whose target does not exist yet), birda follows it and writes the target file, so dotfiles setups work.
- **Hardlinks**: Atomic rename replaces the path with a new inode, so hardlinks diverge and stop tracking.
- **Containers**: Bind-mount the config **directory**, not `config.toml` as a single file (renaming over a bind-mounted file fails with `EBUSY`).
- **Permissions**: A newly created `config.toml` is created private (`0600` on Unix). An existing file keeps its permissions.

### Cached `registry.json`

Birda caches its bundled model catalog at `registry.json` in the config directory using the same atomic write, with a few differences:

- It is created with your normal umask (not `0600`), since it contains only the public model catalog and is also read by Birda GUI.
- Existing symlinks are followed, but a dangling symlink at `registry.json` is replaced by a regular file.
- It is overwritten automatically when a newer birda release ships a higher `registry_version` or when the file cannot be parsed.
- If `registry.json` cannot be read (for example due to permissions or I/O errors), birda warns, leaves the file untouched on disk, and uses the bundled catalog in memory.
- If `registry.json` cannot be written (for example on a read-only config directory or single-file bind mount), birda warns and continues with the bundled catalog in memory.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `BIRDA_MODEL` | Default model name from config (`-m`/`--model`) |
| `BIRDA_MODEL_PATH` | Path to ONNX model file (`--model-path`) |
| `BIRDA_LABELS_PATH` | Path to labels file (`--labels-path`) |
| `BIRDA_MODEL_TYPE` | Model type for ad-hoc model (`--model-type`) |
| `BIRDA_GEOMODEL_PATH` | Path to BirdNET Geomodel v3.0.2 ONNX file (`--geomodel-path`) |
| `BIRDA_GEOMODEL_LABELS_PATH` | Path to BirdNET Geomodel v3.0.2 labels file (`--geomodel-labels-path`) |
| `BIRDA_FORMAT` | Output file formats, comma-separated (`-f`/`--format`) |
| `BIRDA_OUTPUT_DIR` | Output directory (`-o`/`--output-dir`) |
| `BIRDA_MIN_CONFIDENCE` | Minimum confidence threshold, `0.0-1.0` (`-c`/`--min-confidence`) |
| `BIRDA_OVERLAP` | Segment overlap in seconds, finite non-negative (`--overlap`) |
| `BIRDA_BATCH_SIZE` | Inference batch size, `1-512` (`-b`/`--batch-size`) |
| `BIRDA_LATITUDE` | Latitude for range filtering / BSG SDM, `-90.0` to `90.0` (`--lat`) |
| `BIRDA_LONGITUDE` | Longitude for range filtering / BSG SDM, `-180.0` to `180.0` (`--lon`) |
| `BIRDA_DAY_OF_YEAR` | Day of year for BSG SDM, `1-366` (`--day-of-year`) |
| `BIRDA_RANGE_THRESHOLD` | Range filter threshold, `0.0-1.0` (`--range-threshold`) |
| `BIRDA_RANGE_UNMATCHED` | Policy for species missing from geomodel: `keep` or `drop` (`--range-unmatched`) |
| `BIRDA_SPECIES_LIST` | Path to static species list file (`--slist`) |
| `BIRDA_OUTPUT_MODE` | CLI output mode: `human`, `json`, or `ndjson` (`--output-mode`) |
| `BIRDA_CONFIG_DIR` | Override directory for both config (`config.toml`, `registry.json`) and data (`models/`) |
| `BIRDA_INFERENCE_TIMEOUT` | Per-batch inference watchdog timeout in seconds (default: `10`) |
| `HF_ENDPOINT` | Hugging Face mirror base URL for model downloads (e.g. `https://hf-mirror.com`) |
| `ORT_DYLIB_PATH` | Explicit path to the ONNX Runtime shared library |
| `INTEL_OPENVINO_DIR` | OpenVINO runtime install root (see [OpenVINO Guide](openvino.md)) |
