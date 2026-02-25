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
uv run reduce run INSTRUMENT [OPTIONS]
```

**Arguments:**
- `INSTRUMENT` - Instrument name (e.g., UVES, HARPS, XSHOOTER)

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--target` | `-t` | Target star name or regex pattern |
| `--night` | `-n` | Observation night (YYYY-MM-DD format) |
| `--channel` | `-c` | Instrument channel/detector (e.g., RED, BLUE, middle) |
| `--steps` | `-s` | Comma-separated steps to run (default: all) |
| `--base-dir` | `-b` | Base data directory (default: $REDUCE_DATA or ~/REDUCE_DATA) |
| `--input-dir` | `-i` | Input directory relative to base (default: raw) |
| `--output-dir` | `-o` | Output directory relative to base (default: reduced) |
| `--plot` | `-p` | Plot level: 0=none, 1=basic, 2=detailed |
| `--plot-dir` | | Save plots to this directory as PNG files |
| `--plot-show` | | Display mode: block, defer, or off |
| `--trace-range` | | Trace range to process (e.g., "1,21") |
| `--settings` | | JSON file with settings overrides |
| `--use` | | Fiber group(s) to reduce (e.g., "upper" or "upper,lower") |

**Examples:**

```bash
# Basic reduction
uv run reduce run UVES -t HD132205

# Specify night and channel
uv run reduce run UVES -t HD132205 --night 2010-04-01 --channel middle

# Run specific steps
uv run reduce run UVES -t HD132205 --steps bias,flat,trace,science

# Custom directories
uv run reduce run HARPS -t "HD 12345" --base-dir /data --output-dir processed

# With plotting - save to files
uv run reduce run XSHOOTER -t target --plot 1 --plot-dir /tmp/plots --plot-show off

# With plotting - show all at end (useful with webagg backend)
uv run reduce run UVES -t target --plot 1 --plot-show defer
```

### Individual Step Commands

Each reduction step can be run individually:

```bash
uv run reduce bias INSTRUMENT [OPTIONS]
uv run reduce flat INSTRUMENT [OPTIONS]
uv run reduce trace INSTRUMENT [OPTIONS]
uv run reduce curvature INSTRUMENT [OPTIONS]
uv run reduce scatter INSTRUMENT [OPTIONS]
uv run reduce norm_flat INSTRUMENT [OPTIONS]
uv run reduce wavecal_master INSTRUMENT [OPTIONS]
uv run reduce wavecal_init INSTRUMENT [OPTIONS]
uv run reduce wavecal INSTRUMENT [OPTIONS]
uv run reduce freq_comb_master INSTRUMENT [OPTIONS]
uv run reduce freq_comb INSTRUMENT [OPTIONS]
uv run reduce science INSTRUMENT [OPTIONS]
uv run reduce continuum INSTRUMENT [OPTIONS]
uv run reduce finalize INSTRUMENT [OPTIONS]
```

These accept the same options as `run` (including `-t/--target`) except `--steps`, plus:

| Option | Short | Description |
|--------|-------|-------------|
| `--file` | `-f` | Specific input file (bypasses file discovery) |
| `--settings` | | JSON file with settings overrides |

**Example with --file:**

```bash
# Run trace on a specific flat file
uv run reduce trace UVES -t HD132205 --file /path/to/flat.fits

# Override settings for a step
uv run reduce trace UVES -t HD132205 --settings my_settings.json
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
      *.fits                    # Input FITS files
    reduced/{night}/
      {inst}_{chan}.bias.fits   # Master bias
      {inst}_{chan}.flat.fits   # Master flat
      {inst}_{chan}.traces.fits # Traces and wavelength polynomials
      *.{chan}.science.fits     # Extracted spectra
      *.{chan}.final.fits       # Final output
```

For multi-fiber instruments, group-specific files use the pattern
`{inst}_{chan}_{group}.{step}.{ext}` (e.g., `harpspol_blue_upper.linelist.npz`).

The base directory can be set via:
1. `--base-dir` option
2. `REDUCE_DATA` environment variable
3. Default: `~/REDUCE_DATA`

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (missing files, invalid options, etc.) |
