# Species List Usage Guide

This guide covers birda's species filtering capabilities, including both dynamic range filtering and static species lists.

## Overview

Birda supports two methods for filtering bird species detections:

1. **Dynamic Range Filtering** - Uses the BirdNET Geomodel v3.0.2 to filter species based on location and date
2. **Static Species Lists** - Uses pre-generated text files containing species lists

Both methods are fully compatible with BirdNET-Analyzer's species list file format.

## Filtering Priority

When multiple filtering options are provided, birda uses this priority order:

1. **Latitude/Longitude** (dynamic filtering) - Highest priority
2. **Species List File** (static filtering) - Medium priority
3. **No Filtering** - Default (all species enabled)

If you provide both `--lat`/`--lon` AND `--slist`, the species list file will be ignored, and dynamic filtering will be used.

## Dynamic Range Filtering

Dynamic filtering uses the BirdNET Geomodel v3.0.2 to predict which species are likely to occur at a specific location and time. The geomodel covers 12,012 species across birds, mammals, insects, amphibians and reptiles, and is shared by every classifier: BirdNET v2.4, BirdNET v3.0 and Perch v2 all use the same one.

### Requirements

- The BirdNET Geomodel, which birda offers to download on first use (14.7 MB, CC BY-SA 4.0). Install it explicitly with `birda models install geomodel`.
- Location coordinates (latitude and longitude)
- Date information (week OR month+day)

### Species coverage

No classifier's species list is a subset of the geomodel's, so some species have no range data at all:

| Model | Species | With geomodel coverage |
|-------|---------|------------------------|
| BirdNET v2.4 | 6,522 | 6,217 |
| Perch v2 | 14,795 | 11,145 |

The BirdNET v2.4 shortfall is mostly recent eBird taxonomic revisions, plus the ten non-species labels (`Dog`, `Engine`, `Fireworks`, `Gun`, `Human non-vocal`, `Human vocal`, `Human whistle`, `Noise`, `Power tools`, `Siren`). Perch's is mostly environmental sound classes, insects and amphibians.

By default these species **pass through unfiltered**, so range filtering never silently loses a detection it has no evidence about. Use `--range-unmatched drop` to filter them out instead. birda prints the mapped and unmatched counts at startup, so coverage is visible rather than assumed.

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
| `--range-unmatched` | Species with no geomodel entry | keep, drop | keep |
| `--geomodel-path` | Override the geomodel ONNX path | path | from config |
| `--geomodel-labels-path` | Override the geomodel labels path | path | from config |
| `--yes` | Accept the geomodel download without prompting | boolean | false |

**Note:** You must provide either `--week` OR `--month`+`--day`, but not both.

### How It Works

1. The BirdNET Geomodel predicts an occurrence probability for each of its 12,012 species at your location and date
2. Those scores are matched onto your model's labels by scientific name, so localized label files work unchanged
3. Species with scores below `--range-threshold` are filtered out
4. Species with no geomodel entry pass through, unless `--range-unmatched drop` is set
5. If `--rerank` is enabled, detection confidence scores are multiplied by occurrence probability, and species with no geomodel entry are excluded: reranking has no probability to weight them by, and treating a missing one as certainty would push exactly the least-known species to the top
6. Only detections above `--min-confidence` are reported

Note that `--rerank` rewrites the reported confidence values, so they are not comparable with a non-reranked run. `--min-confidence` is applied before range filtering, so reranking can push a detection below your threshold and it will still be reported.

### Attribution

The geomodel is a BirdNET product, licensed CC BY-SA 4.0. Powered by [BirdNET](https://birdnet.cornell.edu/).

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

### How a generated list is written

`birda species` writes its `--output` file to a temporary beside it and renames it into place, so the list appears complete or not at all. That matters because a truncated species list is silently valid: every line in it is well formed, so a list cut short by an interrupted write reads as a shorter range and quietly filters a later analysis run down to whatever survived.

Consequences of writing it that way:

- The file keeps the permissions your umask gives it, as before.
- An output path that is a **hardlink** stops tracking, since a rename gives the path a new inode. An output path that is a **dangling symlink** is replaced by a regular file; an existing symlink is followed.
- Missing directories in the output path are created, where the write used to fail. `birda species -o reports/2026/list.txt` now creates `reports/2026/`, so a mistyped path produces directories rather than an error.
- Writing to a device or pipe still works: birda writes through those directly rather than trying to replace them. So `-o /dev/null`, or `-o /dev/stdout` with stdout on a terminal or a pipe, behave as they always did. Note that `-o /dev/stdout` **redirected to a regular file** resolves to that file, so birda replaces it, exactly as `>` would; if you want the list appended to something, write it to a real path and append that.

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

Use the `species` subcommand to generate species list files from the BirdNET Geomodel. The output is written in your model's own label language, so it can be fed straight back in with `--slist`.

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

### Geomodel Setup

`birda models install geomodel` records the geomodel paths for you, so this is usually not something you edit by hand. The geomodel is shared, so it is configured once under `[defaults]` rather than per model:

```toml
# ~/.config/birda/config.toml

[models.birdnet-v24]
path = "/path/to/BirdNET_GLOBAL_6K_V2.4_Model_FP32.onnx"
labels = "/path/to/BirdNET_GLOBAL_6K_V2.4_Labels.txt"
type = "birdnet-v24"

[defaults]
model = "birdnet-v24"
geomodel = "~/.local/share/birda/models/birdnet-geomodel-v3.0.2.onnx"
geomodel_labels = "~/.local/share/birda/models/birdnet-geomodel-v3.0.2-labels.txt"
range_unmatched = "keep"
```

The old per-model `meta_model` key is ignored. birda warns once when it sees one and drops it the next time the config is written.

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

### Range filtering is disabled

**Warning:**
```
Range filtering disabled: BirdNET Geomodel v3.0.2 is not installed;
run 'birda models install geomodel' to enable range filtering
```

This is what you see after upgrading from a version that used the old BirdNET v2.4 meta model, in a non-interactive run. Analysis continues without range filtering rather than failing, so scripts and cron jobs keep working.

**Solution:**
```bash
birda models install geomodel
```

In an interactive terminal birda offers the download instead of warning. Pass `--yes` to accept it without a prompt in a script.

### "no network connectivity" error

**Error:**
```
error: no network connectivity to huggingface.co;
run 'birda models install geomodel' when online
```

birda checks the download host before starting a transfer, so being offline reports this rather than timing out. Install the geomodel once while online and it is reused afterwards.

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
