# Bat Detection Guide

Birda supports bat species detection using [BattyBirdNET](https://github.com/rdz-oss/BattyBirdNET-Analyzer) regional classifiers.

## How It Works

Bat echolocation calls are ultrasonic (20-120 kHz). BattyBirdNET uses a two-stage pipeline and a "slow-down trick":

1. 256 kHz bat recordings are fed directly into a BirdNET v2.4 embeddings backbone without resampling.
2. BirdNET's spectrogram pipeline (trained on 48 kHz audio) treats those samples as 48 kHz, shifting ultrasonic frequencies down into the audible range where it extracts 1,024-dimensional audio embeddings.
3. A regional BattyBirdNET classifier maps those embeddings to bat species probabilities.

## Setup

Install any regional bat classifier by ID (`bat-<region>`). On the first bat install, birda automatically downloads both the regional classifier and the shared BirdNET v2.4 embeddings backbone, verifying both against published checksums:

```bash
# List all available bat regions (listed under "Bat classifiers")
birda models list-available

# Inspect a region
birda models info bat-eu

# Install a region (also installs the shared embeddings backbone once)
birda models install bat-eu
```

To remove a bat region:

```bash
birda models remove bat-eu --purge
```

## Usage

Once a region is installed, pass `--bat <region>`. Birda resolves the shared embeddings backbone automatically, so `-m` is not required:

```bash
# Analyze with the Bavaria classifier
birda --bat bavaria bat_recording.wav

# Other installed regions
birda --bat uk bat_recording.wav
birda --bat eu bat_recording.wav
```

## Available Regions

| Region | Flag | Install ID | Species | Coverage |
|--------|------|------------|---------|----------|
| Bavaria | `--bat bavaria` | `bat-bavaria` | 32 | Germany, Central Europe |
| Bavaria (high confidence) | `--bat bavaria-high` | `bat-bavaria-high` | 24 | Germany, stricter thresholds |
| EU | `--bat eu` | `bat-eu` | 30 | Broad European coverage |
| Scotland | `--bat scotland` | `bat-scotland` | 11 | Scotland |
| South Wales | `--bat south-wales` | `bat-south-wales` | 29 | South Wales |
| Sweden | `--bat sweden` | `bat-sweden` | 23 | Sweden, Nordic |
| UK | `--bat uk` | `bat-uk` | 20 | United Kingdom |
| USA | `--bat usa` | `bat-usa` | 38 | United States (full) |
| USA East | `--bat usa-east` | `bat-usa-east` | 23 | Eastern United States |
| USA East (high confidence) | `--bat usa-east-high` | `bat-usa-east-high` | 17 | Eastern US, stricter thresholds |
| USA West | `--bat usa-west` | `bat-usa-west` | 28 | Western United States |

## Audio Requirements

- **Sample rate**: 256 kHz (standard for bat recorders such as AudioMoth)
- **Format**: WAV or FLAC recommended for ultrasonic recordings (MP3 and AAC are also accepted by the decoder)
- Birda warns if the source audio is not 256 kHz, but still attempts analysis without resampling

## Notes

- Bat mode overrides segment duration to 0.5625 s (144,000 samples at 256 kHz) with 25% overlap.
- The shared BirdNET v2.4 embeddings backbone installs once with the first bat region and is reused by all other bat regions.
- Dynamic geomodel range filtering is not applied in bat mode.
- Bat models and the BirdNET v2.4 backbone are licensed CC-BY-NC-SA-4.0 (non-commercial); crediting BirdNET is required wherever they are used.
- All standard output formats are supported (CSV, Parquet, Raven, Audacity, Kaleidoscope, JSON).
