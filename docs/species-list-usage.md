# Species List Usage Guide

This guide covers birda's species filtering capabilities, including both dynamic range filtering and static species lists.

## Overview

Birda supports two methods for filtering bird species detections:

1. **Dynamic Range Filtering** - Uses BirdNET's meta model to filter species based on location and date
2. **Static Species Lists** - Uses pre-generated text files containing species lists

Both methods are fully compatible with BirdNET-Analyzer's species list file format.

## Filtering Priority

When multiple filtering options are provided, birda uses this priority order:

1. **Latitude/Longitude** (dynamic filtering) - Highest priority
2. **Species List File** (static filtering) - Medium priority
3. **No Filtering** - Default (all species enabled)

If you provide both `--lat`/`--lon` AND `--slist`, the species list file will be ignored, and dynamic filtering will be used.

## Dynamic Range Filtering

Dynamic filtering uses the meta model to predict which species are likely to occur at a specific location and time.

### Requirements

- Model with `meta_model` configured in `config.toml`
- Location coordinates (latitude and longitude)
- Date information (week OR month+day)

### Usage Examples

**Filter by week number:**
```bash
# Analyze recording from Helsinki, Finland in mid-June (week 24)
birda recording.wav --lat 60.1699 --lon 24.9384 --week 24

# Use custom threshold (default: 0.01)
birda recording.wav --lat 60.1699 --lon 24.9384 --week 24 --range-threshold 0.05
```

**Filter by month and day:**
```bash
# Same as week 24, but using month+day
birda recording.wav --lat 60.1699 --lon 24.9384 --month 6 --day 15
```

**Re-rank results by location probability:**
```bash
# Multiply confidence scores by location probability
birda recording.wav --lat 60.1699 --lon 24.9384 --week 24 --rerank
```

### Range Filter Parameters

| Parameter | Description | Valid Range | Default |
|-----------|-------------|-------------|---------|
| `--lat` | Latitude | -90.0 to 90.0 | (required) |
| `--lon` | Longitude | -180.0 to 180.0 | (required) |
| `--week` | Week number | 1-48 | (one required) |
| `--month` | Month | 1-12 | (one required) |
| `--day` | Day of month | 1-31 | (one required) |
| `--range-threshold` | Minimum location score | 0.0-1.0 | 0.01 |
| `--rerank` | Re-rank by confidence × location | boolean | false |

**Note:** You must provide either `--week` OR `--month`+`--day`, but not both.

### How It Works

1. The meta model predicts a probability score for each species at your location and date
2. Species with scores below `--range-threshold` are filtered out
3. If `--rerank` is enabled, detection confidence scores are multiplied by location scores
4. Only detections above `--min-confidence` are reported

## Static Species Lists

Static species lists are text files containing pre-selected species. This is useful when:

- You want consistent filtering across multiple recordings
- You already know which species to expect
- You want to share species lists between birda and BirdNET-Analyzer

### File Format

Species list files use BirdNET-Analyzer's format:

```
Genus species_Common Name
```

**Example (`my_species.txt`):**
```
Parus major_Great Tit
Cyanistes caeruleus_Blue Tit
Erithacus rubecula_European Robin
Sturnus vulgaris_European Starling
Turdus merula_Eurasian Blackbird
```

**Format Rules:**
- One species per line
- Format: Scientific name (Genus + species) + underscore + Common name
- Blank lines are ignored
- Case-sensitive (must match labels file exactly)

### Usage Examples

**Use species list during analysis:**
```bash
# Use species list file
birda recording.wav --slist my_species.txt

# Process multiple files with same list
birda *.wav --slist my_species.txt
```

**Set default species list in config:**
```toml
# ~/.config/birda/config.toml
[defaults]
species_list_file = "/path/to/my_species.txt"
```

**Override config with environment variable:**
```bash
export BIRDA_SPECIES_LIST=/path/to/winter_species.txt
birda recording.wav
```

## Generating Species Lists

Use the `species` subcommand to generate species list files from the meta model.

### Basic Usage

```bash
# Generate species list for Helsinki in mid-June
birda species --lat 60.1699 --lon 24.9384 --week 24
```

This creates `species_list.txt` in the current directory.

### Advanced Options

**Custom output file:**
```bash
birda species --lat 60.1699 --lon 24.9384 --week 24 --output helsinki_summer.txt
```

**Adjust threshold:**
```bash
# Higher threshold = fewer species (more selective)
# Lower threshold = more species (more inclusive)
birda species --lat 60.1699 --lon 24.9384 --week 24 --threshold 0.1

# Note: Default is 0.03 (higher than filtering default of 0.01)
#       to reduce noise in generated lists
```

**Sort alphabetically instead of by probability:**
```bash
birda species --lat 60.1699 --lon 24.9384 --week 24 --sort alpha
```

**Use specific model:**
```bash
birda species --lat 60.1699 --lon 24.9384 --week 24 -m birdnet-v24
```

### Species Command Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--output` / `-o` | Output file path | `species_list.txt` |
| `--lat` | Latitude | (required) |
| `--lon` | Longitude | (required) |
| `--week` | Week number (1-48) | (one required) |
| `--month` | Month (1-12) | (one required) |
| `--day` | Day of month (1-31) | (one required) |
| `--threshold` | Minimum score | 0.03 |
| `--sort` | Sort order (freq/alpha) | freq |
| `--model` / `-m` | Model to use | (config default) |

### Example Workflow

```bash
# 1. Generate species list for your location and season
birda species --lat 42.3601 --lon -71.0589 --month 5 --day 15 \
  --output boston_may.txt --threshold 0.05

# 2. Review the generated list
cat boston_may.txt

# 3. Use it to analyze recordings
birda my_recordings/*.wav --slist boston_may.txt
```

## BirdNET-Analyzer Compatibility

Birda's species list files are **fully compatible** with BirdNET-Analyzer:

- Same file format (one species per line)
- Same species naming convention (`Genus species_Common Name`)
- Can be used interchangeably between tools

**Example: Generate in birda, use in BirdNET-Analyzer:**
```bash
# Generate with birda
birda species --lat 40.7128 --lon -74.0060 --week 20 --output nyc_may.txt

# Use in BirdNET-Analyzer (Python)
python analyze.py --i recording.wav --slist nyc_may.txt
```

**Example: Generate in BirdNET-Analyzer, use in birda:**
```bash
# Generate with BirdNET-Analyzer
python species.py --o my_list.txt --lat 51.5074 --lon -0.1278 --week 24

# Use in birda
birda recording.wav --slist my_list.txt
```

## Use Cases

### Dynamic Filtering

Best for:
- Real-time or one-off analysis where you know the location
- Varying locations (each recording from a different place)
- Seasonal migration patterns (date-specific filtering)
- When you want the most up-to-date range predictions

Example:
```bash
# Analyzing a recording from a specific field trip
birda field_trip_2025_06_15.wav --lat 45.5017 --lon -73.5673 --month 6 --day 15
```

### Static Species Lists

Best for:
- Batch processing recordings from the same location and time period
- When you've manually curated a species list
- Sharing analysis parameters with collaborators
- Reproducible analysis (list doesn't change)

Example:
```bash
# All recordings from May breeding bird survey
birda bbs_route_*.wav --slist route_42_may_species.txt
```

### Generating and Reusing Lists

Best for:
- Creating location-specific lists for future use
- Building seasonal species lists
- Distribution to team members or automated workflows

Example:
```bash
# Generate spring and fall migration lists
birda species --lat 41.8781 --lon -87.6298 --month 5 --day 1 \
  --output chicago_spring.txt

birda species --lat 41.8781 --lon -87.6298 --month 9 --day 15 \
  --output chicago_fall.txt

# Use throughout the season
birda spring_recordings/*.wav --slist chicago_spring.txt
birda fall_recordings/*.wav --slist chicago_fall.txt
```

## Configuration

### Meta Model Setup

To enable range filtering, add a meta model to your config:

```toml
# ~/.config/birda/config.toml

[models.birdnet-v24]
path = "/path/to/BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx"
labels = "/path/to/BirdNET_GLOBAL_6K_V2.4_Labels.txt"
type = "birdnet-v24"
meta_model = "/path/to/BirdNET_GLOBAL_6K_V2.4_MData_Model_FP32.onnx"

[defaults]
model = "birdnet-v24"
```

### Default Species List

Set a default species list for all analyses:

```toml
[defaults]
species_list_file = "/home/user/my_local_species.txt"
```

This will be used unless:
- You provide `--lat`/`--lon` (dynamic filtering overrides)
- You provide `--slist` on the command line (overrides config)

## Troubleshooting

### "meta model required" error

**Error:**
```
error: range filtering requires meta model (model birdnet-v24 has no meta model configured)
```

**Solution:**
Add `meta_model` path to your model config, or download a model that includes it:
```bash
birda models install birdnet-v24 --language en
```

### "species list file not found" error

**Error:**
```
error: failed to read species list file 'my_species.txt'
```

**Solution:**
- Verify the file exists: `ls -l my_species.txt`
- Use absolute path: `--slist /full/path/to/my_species.txt`
- Check file permissions

### No detections with species list

**Issue:** Analysis completes but reports no detections.

**Possible causes:**
1. Species list doesn't contain the actual species in the recording
2. Species names don't match labels file exactly (check case and format)
3. Min confidence threshold too high

**Solution:**
- Remove `--slist` temporarily to see all detections
- Compare detected species against your list
- Verify species names match labels file format exactly

### Threshold too restrictive

**Issue:** Species list generated but contains very few species.

**Solution:**
Lower the threshold:
```bash
# Try progressively lower thresholds
birda species --lat 60.17 --lon 24.94 --week 24 --threshold 0.01
```

Typical thresholds:
- `0.01` - Very inclusive (recommended for analysis)
- `0.03` - Moderate (default for list generation)
- `0.1` - Conservative (only common species)

## Performance Notes

- **Dynamic filtering:** Minimal overhead (~1-5ms per prediction batch)
- **Static lists:** Near-zero overhead (simple set lookup)
- **List generation:** One-time cost (~100-500ms depending on model)

Both methods are highly efficient and suitable for large-scale batch processing.

## Further Reading

- [BirdNET-Analyzer Species Lists](https://github.com/kahst/BirdNET-Analyzer#species-lists)
- [BirdNET Meta Model Documentation](https://github.com/kahst/BirdNET-Analyzer#species-prediction)
- [Range Filter Configuration](../README.md#configuration)
