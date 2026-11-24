# Output Formats Guide

Birda writes detection results in six file formats and also supports structured JSON/NDJSON CLI output for automation and GUI integration.

Select formats with `-f`/`--format` (comma-separated for multiple formats) or `defaults.formats` in `config.toml`:

```bash
birda -f csv,parquet recording.wav
```

## Summary of File Formats

| Format | Flag | File Suffix | Best For |
|--------|------|-------------|----------|
| CSV (default) | `-f csv` | `.BirdNET.results.csv` | Spreadsheets, scripts, BirdNET-Analyzer compatibility |
| Parquet | `-f parquet` | `.BirdNET.results.parquet` | Large datasets, Pandas, Polars, DuckDB, Spark |
| Raven | `-f raven` | `.BirdNET.selection.table.txt` | [Raven Pro](https://ravensoundsoftware.com/) |
| Audacity | `-f audacity` | `.BirdNET.results.txt` | [Audacity](https://www.audacityteam.org/) label tracks |
| Kaleidoscope | `-f kaleidoscope` | `.BirdNET.results.kaleidoscope.csv` | [Wildlife Acoustics Kaleidoscope](https://www.wildlifeacoustics.com/products/kaleidoscope) |
| JSON | `-f json` | `.BirdNET.json` | Structured metadata + summary statistics |

## Output File Naming and Skipping

Result files are named `<input stem><suffix>` and written next to the input file, or into the `-o`/`--output-dir` directory.

- **Same-name inputs**: When multiple inputs in one run would collide (such as `x.wav` and `x.flac` in one folder, or `in/a/x.wav` and `in/b/x.wav` with `-o out`), birda qualifies the filename with the full input name (`x.wav.BirdNET.results.csv`) and, with `-o`, mirrors the input subdirectory (`out/in/a/`, `out/in/b/`). Inputs with unique names keep the plain name.
- **Ownership checks**: An existing output that records a different input from outside the current run is never silently treated as a match: that input fails with `output_owned_by_other_input` unless `--force` is passed.

See [Output File Names](json-output.md#output-file-names) and [Skipping Existing Outputs](json-output.md#skipping-existing-outputs) in the JSON Output Guide for full rules.

## CSV (Default)

Standard CSV with columns: `Start (s)`, `End (s)`, `Scientific name`, `Common name`, `Confidence`, `File`.

```csv
Start (s),End (s),Scientific name,Common name,Confidence,File
0.0,3.0,Glaucidium passerinum,Eurasian Pygmy Owl,0.9237,recording.wav
3.0,6.0,Glaucidium passerinum,Eurasian Pygmy Owl,0.9849,recording.wav
```

- Includes a UTF-8 BOM by default so Excel on Windows detects UTF-8 properly. Pass `--no-csv-bom` to omit it.
- Optional metadata columns can be added via `defaults.csv_columns.include` in `config.toml`. Accepted column names: `lat`, `lon`, `week`, `model`, `overlap`, `sensitivity`, `min_conf`, `species_list`.

## Parquet

Apache Parquet columnar format (Snappy-compressed, Parquet 2.0) for large datasets and analytics workflows, typically 50-80% smaller than CSV with native typed columns.

```bash
# Single format
birda -f parquet recording.wav

# CSV and Parquet together
birda -f csv,parquet recording.wav
```

### Column Schema

- **Core columns**: `start_s` (Float32), `end_s` (Float32), `scientific_name` (Utf8), `common_name` (Utf8), `confidence` (Float32), `file` (Utf8)
- **Optional metadata columns** (configured via `defaults.csv_columns.include` in `config.toml`): `lat` (Float64), `lon` (Float64), `week` (UInt8), `model` (Utf8), `overlap` (Float32), `sensitivity` (Float32), `min_conf` (Float32), `species_list` (Utf8)

### Reading Parquet Files

```python
import pandas as pd
df = pd.read_parquet("recording.BirdNET.results.parquet")
print(df.head())
```

```python
import polars as pl
df = pl.read_parquet("recording.BirdNET.results.parquet")
print(df.describe())
```

```sql
-- DuckDB
SELECT scientific_name, common_name, COUNT(*) AS detections
FROM 'recording.BirdNET.results.parquet'
GROUP BY scientific_name, common_name
ORDER BY detections DESC;
```

## Raven Selection Table

Tab-separated selection table compatible with [Raven Pro](https://ravensoundsoftware.com/).

## Audacity Labels

Tab-separated start/end/label format for direct import into [Audacity](https://www.audacityteam.org/).

## Kaleidoscope CSV

CSV format compatible with [Wildlife Acoustics Kaleidoscope](https://www.wildlifeacoustics.com/products/kaleidoscope).

## JSON Detection Files

Structured JSON containing analysis metadata, summary statistics, and all detections:

```bash
birda -f json recording.wav
```

## Programmatic CLI Output (JSON / NDJSON / Stdout)

Separate from result file formats (`-f`), `--output-mode` controls what birda prints to stdout:

```bash
# Buffered JSON (single object at completion)
birda --output-mode json config show
birda --output-mode json models list

# Streaming NDJSON (real-time pipeline progress events)
birda --output-mode ndjson recording.wav 2>/dev/null

# Stream detections directly to stdout as NDJSON (single input file only;
# cannot be combined with --output-dir or --format)
birda --stdout recording.wav
```

See the [JSON Output Guide](json-output.md) for the complete envelope specification, event schemas, integration examples, and error handling.
