# Models Guide

Birda supports multiple bioacoustic classification models and manages downloads, checksums, and configuration automatically through its built-in registry.

For bat species classifiers (BattyBirdNET), see the [Bat Detection Guide](bat-detection.md).

## Managing Models

```bash
# List models available for download
birda models list-available

# Install a model (downloads model, labels, and shared geomodel)
birda models install birdnet-v24
birda models install birdnet-plus-v30
birda models install perch-v2
birda models install bsg-fi-v44

# List configured models
birda models list

# Show model details (add --languages to list label languages)
birda models info <id-or-name>

# List regional tiles available for a model
birda models regions birdnet-plus-v30

# Show full region/variant manifest (best with --output-mode json)
birda models manifest birdnet-plus-v30 --output-mode json

# Verify installed model files exist and are valid
birda models check

# Remove a model from config (add --purge to also delete files from disk)
birda models remove <name> [--purge]

# Add a local ONNX model manually
birda models add <name> --path <model.onnx> --labels <labels.txt> --type <type> [--default]
# Supported types: birdnet-v24, birdnet-v30, perch-v2, bsg-finland
```

### License and Document Acceptance

When installing a model interactively, birda displays the model's license and asks you to type `accept`. If the publisher also requires agreement to an additional document (such as an acceptable use policy), birda lists that document and its link before the license.

When running without a terminal or with `--output-mode json`/`ndjson`, models that require document agreement can only be installed with `--yes`, which records your agreement. Birda stores the document kind, revision, and SHA-256 under `accepted_documents` in the model's `config.toml` entry.

## Available Models

### BirdNET v2.4 (Recommended)

```bash
birda models install birdnet-v24
```

- **License**: CC-BY-NC-SA-4.0 (non-commercial use only)
- **Vendor**: Cornell Lab of Ornithology & Chemnitz University of Technology
- **Sample rate**: 48 kHz
- **Segment duration**: 3 seconds
- **Species**: 6,522 classes globally
- **Range filtering**: Supported (BirdNET Geomodel v3.0.2, 6,217 of 6,522 labels covered)
- **Model type**: `birdnet-v24`
- **Source**: [BirdNET-onnx on Hugging Face](https://huggingface.co/justinchuby/BirdNET-onnx) (ONNX conversion by Justin Chu)

### BirdNET+ V3.0

```bash
birda models install birdnet-plus-v30
```

- **License**: Apache-2.0 (commercial use permitted)
- **Acceptable use**: The publisher's [acceptable use policy](https://huggingface.co/tphakala/BirdNET-Plus-V3.0/resolve/f90c3ba818b179adf5f4c9eabf2e6f8b801e4c06/ACCEPTABLE_USE-v3.0.1.md), which `models install` shows and asks you to agree to (pass `--yes` to agree without a prompt)
- **Vendor**: Cornell Lab of Ornithology & Chemnitz University of Technology
- **Sample rate**: 32 kHz
- **Segment duration**: 5 seconds
- **Species**: 11,725 classes globally (11,709 species and 16 sound classes such as Gun, Engine, and Siren); fewer per region
- **Range filtering**: Supported (BirdNET Geomodel v3.0.2, 10,533 of 11,725 classes covered). Domestic animals (dog, cat, cattle, horse, sheep, goat, chicken) have geomodel entries with low occurrence scores, so range filtering can drop them (tracked in [#455](https://github.com/tphakala/birda/issues/455)). Sound classes and *Homo sapiens* have no geomodel entry and pass through by default
- **Regional models**: 44 region-sliced models (see [Regional Models](#regional-models))
- **Model type**: `birdnet-v30`
- **Source**: [BirdNET-Plus-V3.0 on Hugging Face](https://huggingface.co/tphakala/BirdNET-Plus-V3.0)

Powered by [BirdNET](https://birdnet.cornell.edu/).

#### Upgrading from the BirdNET v3.0 Preview

BirdNET+ V3.0 replaces the BirdNET v3.0 developer preview, and `birdnet-v30` is no longer offered by `birda models install`. Previously installed preview models keep working under their own keys (`birdnet-v30`, `birdnet-v30-<region>`). BirdNET+ installs under `birdnet-plus-v30` and `birdnet-plus-v30-<region>`, so select it with `-m` or set it as default with `--default`. Birda does not migrate or delete preview files. To remove them, run `birda models remove birdnet-v30 --purge` (and the same for each `birdnet-v30-<region>`).

### Google Perch v2

```bash
birda models install perch-v2
```

- **License**: Apache-2.0
- **Vendor**: Google Research
- **Sample rate**: 32 kHz
- **Segment duration**: 5 seconds
- **Species**: 14,795 classes globally (birds, insects, amphibians, mammals; scientific names)
- **Range filtering**: Supported (BirdNET Geomodel v3.0.2, 11,145 of 14,795 classes covered)
- **Regional models**: 44 region-sliced models with per-tile score calibration stored as `softmax_rest_logit` in `config.toml` (see [Regional Models](#regional-models))
- **Model type**: `perch-v2`
- **Source**: [Perch-v2-Models on Hugging Face](https://huggingface.co/tphakala/Perch-v2-Models) (upstream: [Google Research Perch v2](https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2))

### BSG Finnish Birds v4.4

```bash
birda models install bsg-fi-v44
```

- **License**: BSG-NC-1.0 (non-commercial use only, no app stores)
- **Vendor**: University of Jyväskylä
- **Sample rate**: 48 kHz
- **Segment duration**: 3 seconds
- **Species**: 265 Finnish bird species (breeders, migrants, vagrants)
- **Architecture**: Fine-tuned BirdNET v2.4 backbone with custom Finnish classification head
- **Post-processing**:
  1. **Calibration (always applied)**: Per-species logistic regression (Platt scaling)
  2. **Species Distribution Model (optional)**: Adjusts predictions by seasonal and geographic plausibility using migration curves and distribution maps
- **Model type**: `bsg-finland`
- **Source**: [BSG on Hugging Face](https://huggingface.co/tphakala/BSG)
- **Citation**: Nokelainen et al. (2024) [doi:10.5334/cstp.710](https://doi.org/10.5334/cstp.710)

Usage with SDM (recommended for field recordings in Finland):

```bash
# With location and explicit date (day of year, 1-366)
birda recording.wav -m bsg-fi-v44 --lat 60.17 --lon 24.94 --day-of-year 150

# With location only (day of year auto-detected from file modification timestamp)
birda recording.wav -m bsg-fi-v44 --lat 60.17 --lon 24.94

# Calibration only (no geographic/seasonal adjustment)
birda recording.wav -m bsg-fi-v44
```

Notes:
- Dynamic geomodel range filtering (`--week`, `--month`/`--day`) is not used with BSG models; BSG uses its own SDM instead.
- If `--day-of-year` is omitted when `--lat`/`--lon` are given, birda derives the day of year from the audio file's modification timestamp.

### BirdNET Geomodel v3.0.2

```bash
birda models install geomodel
```

The shared range filter used by BirdNET v2.4, BirdNET+ V3.0, and Perch v2. It predicts an occurrence probability for 12,012 species at a given latitude, longitude, and week, which birda maps onto your classifier's labels by scientific name.

- **License**: CC-BY-SA-4.0
- **Vendor**: Cornell Lab of Ornithology & Chemnitz University of Technology
- **Species**: 12,012 scored classes (birds, mammals, insects, amphibians, reptiles)
- **Size**: 14.5 MB (model and labels combined)
- **Source**: [BirdNET-Geomodel on Hugging Face](https://huggingface.co/tphakala/BirdNET-Geomodel)

It is installed automatically alongside any classifier, or offered for download on first use when passing `--lat`/`--lon`. Remove it with `birda models remove geomodel --purge` (without `--purge`, only the config keys are cleared and birda continues using the files in its models directory).

See the [Species List Usage Guide](species-list-usage.md) for coverage details and `--range-unmatched`.

Powered by [BirdNET](https://birdnet.cornell.edu/).

## Regional Models

BirdNET+ V3.0 and Perch v2 each publish 44 region-sliced models alongside the global model. A regional model scores only the species of that region and uses less memory: on a Raspberry Pi 5, the BirdNET+ Nordic model peaked at 198 MB versus 457 MB for the global model. A BirdNET+ regional model is numerically identical to the global model on the species it keeps.

Perch v2 scores with a softmax across its classes, so a smaller slice would otherwise yield inflated probabilities. Each Perch tile carries a calibration constant that `models install` stores as `softmax_rest_logit` in `config.toml`, and both ONNX Runtime and OpenVINO apply it to keep scores aligned with the global model. If a regional Perch install lacks a stored value, birda runs uncalibrated (producing higher scores), warns once per run, and prints the reinstall command. Passing `--model-path` replaces the installed file and drops the stored value.

```bash
# Browse available regions grouped by continent
birda models regions birdnet-plus-v30
birda models regions perch-v2

# Install a regional model
birda models install birdnet-plus-v30 --region nordic

# Use it (registered as <model-id>-<region> so global and regional coexist)
birda -m birdnet-plus-v30-nordic recording.wav
```

### Picking a Region at Install Time

At a terminal, `birda models install birdnet-plus-v30` asks where you record:
- If `defaults.latitude` and `defaults.longitude` are configured, it proposes the matching region and asks you to confirm.
- Otherwise, or if you decline, it prompts for a country (name or ISO code) or coordinates (`lat,lon`), lists matching regions with species counts and download sizes, and offers the global model last.

With `--yes`, or without a terminal (including `--output-mode json`), birda skips the prompt and installs the global model unless `--region` is given.

To skip the prompt explicitly:

```bash
birda models install birdnet-plus-v30 --region nordic
birda models install birdnet-plus-v30 --region auto --lat 60.17 --lon 24.94
birda models install birdnet-plus-v30 --region global
```

- `--region auto` without `--lat`/`--lon` uses `defaults.latitude` and `defaults.longitude` from `config.toml`, and fails if either is unset.
- Place `--lat` and `--lon` after `models install` (before `models`, they belong to the analyze command and birda refuses to use them for region selection).
- When multiple regions cover a point, the most specific one wins (ties go to the nearest region center).
- Preview the selection without installing: `birda models regions birdnet-plus-v30 --lat 60.17 --lon 24.94`. Location matching runs locally against region boundaries in birda's registry with no network lookup.

## Choosing a Variant

Some models publish multiple variants for different hardware:
- **Perch v2**: `fp32` and `no-dft-fp32` for the global model (default: `no-dft-fp32`, which every backend including OpenVINO supports). All Perch regional models are `no-dft-fp32`. The older `int8-arm` variant has been withdrawn and `--variant int8-arm` is refused; on aarch64 the global install uses `no-dft-fp32` (~394 MB / 413 MB decimal).
- **BirdNET+ V3.0**: publishes `fp32` only.

Birda picks a variant automatically based on your configured inference device and detected GPU libraries, prints its choice and reason, and accepts `--variant` to override:

```bash
birda models install perch-v2 --variant no-dft-fp32
```

Auto-selection is conservative and falls back to the family default unless a definite signal is present. On memory-constrained devices, prefer a regional model over the global model.

## Downloading Through a Mirror

Set `HF_ENDPOINT` to use a Hugging Face mirror on networks where `huggingface.co` is unreachable:

```bash
HF_ENDPOINT=https://hf-mirror.com birda models install birdnet-plus-v30
```

## Custom Model Conversion

For converting custom BirdNET classifiers or optimizing models for specific hardware (such as Raspberry Pi or embedded devices), see [birdnet-onnx-converter](https://github.com/tphakala/birdnet-onnx-converter):

- TFLite to ONNX conversion
- Precision formats: FP32 (GPU/desktop), FP16 (RPi 5, modern GPUs), INT8 (CPU optimization)
- Platform-specific optimizations for ARM devices
