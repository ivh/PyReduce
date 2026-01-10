# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PyReduce is a Python port of the REDUCE echelle spectrograph data reduction pipeline. It processes raw astronomical observations from instruments like HARPS, UVES, XSHOOTER, CRIRES+, JWST/NIRISS and others into calibrated 1D spectra.

## Quick Start

```bash
# Install
uv sync

# Download example data
uv run reduce download UVES

# Run example
PYREDUCE_PLOT=0 uv run python examples/uves_example.py

# Or use CLI
uv run reduce run UVES HD132205 --steps bias,flat,trace,science
```

## Package Structure

```
pyreduce/
├── __main__.py          # Click CLI entry point
├── pipeline.py          # Pipeline API (recommended entry point)
├── reduce.py            # Step class implementations
├── configuration.py     # Config loading (settings JSON)
├── extract.py           # Optimal extraction algorithm
├── trace_orders.py      # Order detection and tracing
├── wavelength_calibration.py  # Wavelength solution fitting
├── combine_frames.py    # Frame combination/calibration
├── echelle.py           # Echelle spectrum I/O
├── util.py              # Utilities, plotting helpers
├── cwrappers.py         # CFFI C extension wrappers
│
├── instruments/         # Instrument definitions
│   ├── common.py        # Base Instrument class
│   ├── models.py        # Pydantic config models
│   ├── filters.py       # File classification filters
│   ├── instrument_info.py  # Instrument loader
│   ├── defaults/        # Base settings and line atlases
│   │   ├── settings.json   # Default reduction parameters
│   │   ├── schema.json     # Settings validation schema
│   │   ├── config.yaml     # Base instrument config
│   │   └── atlas/          # Wavelength calibration line lists
│   └── {INSTRUMENT}/    # Per-instrument directory (e.g., UVES/, HARPS/)
│       ├── __init__.py     # Instrument class
│       ├── config.yaml     # Hardware/header config
│       ├── settings.json   # Reduction parameters
│       ├── wavecal_*.npz   # Pre-computed wavelength solutions
│       └── mask_*.fits.gz  # Bad pixel masks
│
└── clib/                # C source for extraction
    ├── slit_func_bd.c
    └── slit_func_2d_xi_zeta_bd.c
```

## Image Coordinate Convention

PyReduce uses the convention that **dispersion runs horizontally (along x-axis)** and **cross-dispersion runs vertically (along y-axis)**. The `clipnflip()` function in `instruments/common.py` rotates and flips raw images to ensure this orientation.

This means:
- **Columns (x)** = wavelength/dispersion direction
- **Rows (y)** = spatial/cross-dispersion direction
- **Traces** are polynomial functions of x, giving y-position
- **`extraction_height`** refers to pixels above/below each trace (in y)

## Pipeline Steps

The reduction pipeline consists of these steps (in typical order):

| Step | Class | Description |
|------|-------|-------------|
| `mask` | `Mask` | Load bad pixel mask for detector |
| `bias` | `Bias` | Combine bias frames into master bias |
| `flat` | `Flat` | Combine flat frames, subtract bias |
| `trace` | `OrderTracing` | Trace echelle order positions on flat |
| `curvature` | `SlitCurvatureDetermination` | Measure slit curvature from arc lamp |
| `scatter` | `BackgroundScatter` | Model inter-order scattered light |
| `norm_flat` | `NormalizeFlatField` | Normalize flat, extract blaze function |
| `wavecal_master` | `WavelengthCalibrationMaster` | Extract wavelength calibration spectrum |
| `wavecal_init` | `WavelengthCalibrationInitialize` | Initial line identification |
| `wavecal` | `WavelengthCalibrationFinalize` | Refine wavelength solution |
| `freq_comb_master` | `LaserFrequencyCombMaster` | Extract frequency comb spectrum |
| `freq_comb` | `LaserFrequencyCombFinalize` | Apply frequency comb calibration |
| `science` | `ScienceExtraction` | Optimally extract science spectra |
| `continuum` | `ContinuumNormalization` | Fit and normalize continuum |
| `finalize` | `Finalize` | Write final FITS output |

Each step class has:
- `run(files, **dependencies)` - Execute the step
- `save(...)` - Save results to disk
- `load(...)` - Load previous results
- `dependsOn` - List of required prior steps
- `savefile` - Output file path

## Configuration System

### Instrument Configs (YAML)

Location: `pyreduce/instruments/{INSTRUMENT}/config.yaml`

Defines what the instrument IS - hardware properties and header mappings:

```yaml
# Basic identification
instrument: HARPS
telescope: ESO-3.6m
channels: [red, blue]

# Detector properties
naxis: [4096, 4096]
orientation: 4
extension: 0
gain: ESO DET OUT1 CONAD    # Header keyword or literal value
readnoise: ESO DET OUT1 RON
dark: 0

# Header keyword mappings (instrument → internal name)
date: DATE-OBS
target: ESO OBS TARG NAME
instrument_mode: ESO INS MODE
exposure_time: EXPTIME
ra: RA
dec: DEC
jd: MJD-OBS

# File classification keywords
kw_bias: ESO DPR TYPE
kw_flat: ESO DPR TYPE
kw_wave: ESO DPR TYPE
kw_spec: ESO DPR TYPE
id_bias: BIAS
id_flat: FLAT.*
id_wave: WAVE,THAR
id_spec: OBJECT
```

Validated by Pydantic model `InstrumentConfig` in `models.py`.

### Reduction Settings (JSON)

Location: `pyreduce/instruments/{INSTRUMENT}/settings.json`

Defines HOW to reduce - algorithm parameters per step:

```json
{
  "bias": {
    "degree": 0
  },
  "trace": {
    "degree": 4,
    "noise": 100,
    "min_cluster": 500,
    "filter_size": 120
  },
  "norm_flat": {
    "extraction_height": 0.5,
    "smooth_slitfunction": 1,
    "oversampling": 10
  },
  "wavecal": {
    "degree": [6, 6],
    "threshold": 100,
    "iterations": 3
  },
  "science": {
    "extraction_method": "optimal",
    "extraction_height": 0.5,
    "oversampling": 10
  }
}
```

Settings cascade: `instruments/defaults/settings.json` < `instruments/{INSTRUMENT}/settings.json` < runtime overrides.

## Python API

### Recommended: Pipeline.from_instrument()

```python
from pyreduce.pipeline import Pipeline

result = Pipeline.from_instrument(
    instrument="UVES",
    target="HD132205",
    night="2010-04-01",
    channel="middle",
    steps=("bias", "flat", "trace", "science"),
    base_dir="/data",
    plot=1,
).run()
```

This handles:
- Loading instrument config
- Finding and sorting input files
- Setting up output directory
- Running requested steps

### Manual Pipeline Construction

```python
from pyreduce.pipeline import Pipeline

pipe = Pipeline(
    instrument="UVES",
    output_dir="/output",
    channel="middle",
    plot=0,
)
pipe.bias(bias_files)
pipe.flat(flat_files)
pipe.trace_orders()
pipe.extract(science_files)
result = pipe.run()
```

### Legacy API (deprecated)

```python
import pyreduce
pyreduce.reduce.main(
    instrument="UVES",
    target="HD132205",
    ...
)  # Shows deprecation warning
```

## CLI Commands

```bash
# Full pipeline
uv run reduce run UVES HD132205 --steps bias,flat,trace

# Individual steps (top-level commands)
uv run reduce bias UVES HD132205
uv run reduce trace UVES HD132205
uv run reduce wavecal UVES HD132205

# Combine reduced spectra
uv run reduce combine --output combined.fits *.final.fits

# Download sample data
uv run reduce download UVES

# List steps
uv run reduce list-steps
```

## Environment Variables

- `REDUCE_DATA` - Base data directory (default: `~/REDUCE_DATA`)
- `PYREDUCE_PLOT` - Override plot level (0=off, 1=basic, 2=detailed)
- `PYREDUCE_PLOT_DIR` - Save plots to directory as PNG files
- `PYREDUCE_PLOT_SHOW` - Display mode: `block` (default), `defer`, or `off`

Plot modes: `block` shows each plot interactively; `defer` accumulates all plots and shows at end (useful with webagg backend); `off` disables display. Save and display are independent.

## Development

### Commands

```bash
uv sync                              # Install dependencies
uv run reduce-build                  # Compile C extensions
uv run reduce-clean                  # Remove compiled extensions
uv run pre-commit install            # Setup hooks (once)
uv run pytest -m unit                # Fast unit tests
uv run pytest --instrument=UVES      # Test single instrument
uv run ruff check --fix .            # Lint and fix
```

After a fresh clone or `rm -rf .venv`, run `uv sync && uv run reduce-build` to set up.

### Adding Instruments

1. Create `pyreduce/instruments/{NAME}/` directory
2. Add `config.yaml` with detector/header config
3. Add `settings.json` for reduction parameters (can use `"__inherits__": "defaults"`)
4. Add `__init__.py` with instrument class if custom logic needed (optional)
5. Add wavecal/mask files if available
6. Add example script to `examples/name_example.py`

### Test Organization

- `@pytest.mark.unit` - Fast tests with synthetic data (~40 tests)
- `@pytest.mark.instrument` - Integration tests with real data (~70 tests)
- `@pytest.mark.slow` - Long-running tests (wavecal, continuum)

## Key Files

| File | Purpose |
|------|---------|
| `pyreduce/__main__.py` | Click CLI entry point |
| `pyreduce/pipeline.py` | Fluent Pipeline API, `from_instrument()` |
| `pyreduce/reduce.py` | Step class implementations |
| `pyreduce/extract.py` | Optimal extraction algorithm |
| `pyreduce/wavelength_calibration.py` | Wavelength solution fitting |
| `pyreduce/trace_orders.py` | Order detection and tracing |
| `pyreduce/instruments/common.py` | Base Instrument class |
| `pyreduce/instruments/models.py` | Pydantic config models |
| `pyreduce/clib/*.c` | C code for slit function decomposition |
| `hatch_build.py` | CFFI extension build hook |

## Release Process

To release a new version (e.g., `0.7a6`):

1. **Update documentation** for any renamed steps, new CLI options, etc:
   - `README.md` - Quick start examples
   - `docs/cli.md` - CLI reference
   - `docs/index.md`, `docs/howto.md`, `docs/examples.md` - Usage examples
   - `docs/configuration_file.md` - Config key names

2. **Update CHANGELOG.md** with release date and changes

3. **Update version** in `pyproject.toml`

3a. **sync** - run `uv sync` to get the new version into uv.lock

4. **Update example headers** - all `examples/*.py` have PEP 723 metadata:
   ```python
   # /// script
   # dependencies = ["pyreduce-astro>=0.7a6"]
   # ///
   ```

5. **Commit, tag, and push**:
   ```bash
   git add -A && git commit -m "Release v0.7a6"
   git tag v0.7a6
   git push && git push --tags
   ```

6. **Monitor GitHub Actions** - the tag push triggers:
   - Tests on Python 3.13
   - Wheel builds (Linux, Windows, macOS)
   - PyPI upload
   - GitHub Release creation

   ```bash
   gh run watch  # watch the triggered workflow
   ```
