# Command Line Interface

PyReduce provides a command-line interface via the `reduce` command.

## Installation

After installing PyReduce, the CLI is available:

```bash
uv run reduce --help
```

Or if installed globally:

```bash
reduce --help
```

## Commands

### run - Full Pipeline

Run the reduction pipeline for an instrument and target:

```bash
uv run reduce run INSTRUMENT TARGET [OPTIONS]
```

**Arguments:**
- `INSTRUMENT` - Instrument name (e.g., UVES, HARPS, XSHOOTER)
- `TARGET` - Target star name or regex pattern

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--night` | `-n` | Observation night (YYYY-MM-DD format) |
| `--channel` | `-c` | Instrument channel/detector (e.g., RED, BLUE, middle) |
| `--steps` | `-s` | Comma-separated steps to run (default: all) |
| `--base-dir` | `-b` | Base data directory (default: $REDUCE_DATA or ~/REDUCE_DATA) |
| `--input-dir` | `-i` | Input directory relative to base (default: raw) |
| `--output-dir` | `-o` | Output directory relative to base (default: reduced) |
| `--plot` | `-p` | Plot level: 0=none, 1=save, 2=interactive |
| `--order-range` | | Order range to process (e.g., "1,21") |
| `--settings` | | JSON file with settings overrides |

**Examples:**

```bash
# Basic reduction
uv run reduce run UVES HD132205

# Specify night and channel
uv run reduce run UVES HD132205 --night 2010-04-01 --channel middle

# Run specific steps
uv run reduce run UVES HD132205 --steps bias,flat,trace,science

# Custom directories
uv run reduce run HARPS "HD 12345" --base-dir /data --output-dir processed

# With plotting
uv run reduce run XSHOOTER target --plot 1
```

### Individual Step Commands

Each reduction step can be run individually:

```bash
uv run reduce bias INSTRUMENT TARGET [OPTIONS]
uv run reduce flat INSTRUMENT TARGET [OPTIONS]
uv run reduce trace INSTRUMENT TARGET [OPTIONS]
uv run reduce curvature INSTRUMENT TARGET [OPTIONS]
uv run reduce scatter INSTRUMENT TARGET [OPTIONS]
uv run reduce norm_flat INSTRUMENT TARGET [OPTIONS]
uv run reduce wavecal_master INSTRUMENT TARGET [OPTIONS]
uv run reduce wavecal_init INSTRUMENT TARGET [OPTIONS]
uv run reduce wavecal INSTRUMENT TARGET [OPTIONS]
uv run reduce freq_comb_master INSTRUMENT TARGET [OPTIONS]
uv run reduce freq_comb INSTRUMENT TARGET [OPTIONS]
uv run reduce science INSTRUMENT TARGET [OPTIONS]
uv run reduce continuum INSTRUMENT TARGET [OPTIONS]
uv run reduce finalize INSTRUMENT TARGET [OPTIONS]
```

These accept the same options as `run` except `--steps`, plus:

| Option | Short | Description |
|--------|-------|-------------|
| `--file` | `-f` | Specific input file (bypasses file discovery) |
| `--settings` | | JSON file with settings overrides |

**Example with --file:**

```bash
# Run trace on a specific flat file
uv run reduce trace UVES HD132205 --file /path/to/flat.fits

# Override settings for a step
uv run reduce trace UVES HD132205 --settings my_settings.json
```

The settings file can contain partial overrides:

```json
{
  "trace": {
    "degree": 6,
    "noise": 50
  }
}
```

### download - Sample Data

Download sample datasets for testing:

```bash
uv run reduce download INSTRUMENT
```

**Examples:**

```bash
uv run reduce download UVES
uv run reduce download HARPS
uv run reduce download XSHOOTER
```

The data is downloaded to `~/REDUCE_DATA/INSTRUMENT/` by default.

### combine - Merge Spectra

Combine multiple reduced spectra into one:

```bash
uv run reduce combine FILES... --output OUTPUT
```

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output filename (default: combined.fits) |
| `--plot` | `-p` | Plot specific order for inspection |

**Examples:**

```bash
# Combine all final files
uv run reduce combine *.final.fits --output combined.fits

# Combine specific files
uv run reduce combine night1.fits night2.fits night3.fits -o combined.fits
```

### list-steps

List all available reduction steps:

```bash
uv run reduce list-steps
```

Output:
```
Available reduction steps:
  - bias
  - flat
  - trace
  - curvature
  - scatter
  - norm_flat
  - wavecal_master
  - wavecal_init
  - wavecal
  - freq_comb_master
  - freq_comb
  - science
  - continuum
  - finalize
```

## Directory Structure

PyReduce expects the following directory structure:

```
$REDUCE_DATA/
  INSTRUMENT/
    raw/
      *.fits         # Input FITS files
    reduced/
      *_bias.fits    # Master bias
      *_flat.fits    # Master flat
      *_orders.npz   # Order traces
      *.science.fits # Extracted spectra
      *.final.fits   # Final output
```

The base directory can be set via:
1. `--base-dir` option
2. `REDUCE_DATA` environment variable
3. Default: `~/REDUCE_DATA`

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (missing files, invalid options, etc.) |
