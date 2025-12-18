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
uv run reduce run UVES --steps bias,flat,orders,science --output ./reduced
```

## Pipeline Steps

| Step | Description | Main Input |
|------|-------------|------------|
| `mask` | Creates bad pixel mask from detector defects | Instrument config |
| `bias` | Combines bias frames into master bias | Bias FITS files |
| `flat` | Combines flat field frames into master flat | Flat FITS files |
| `orders` | Traces echelle order positions across detector | Master flat |
| `curvature` | Measures slit tilt/shear for curved extraction | Calibration lamp files |
| `scatter` | Models inter-order scattered light background | Flat or science image |
| `norm_flat` | Normalizes flat field, extracts blaze function | Master flat + orders |
| `wavecal_master` | Extracts wavelength calibration lamp spectrum | ThAr/etalon FITS files |
| `wavecal_init` | Identifies spectral lines, creates initial solution | Master wavecal spectrum |
| `wavecal` | Refines wavelength solution polynomial fit | Initial line identifications |
| `freq_comb_master` | Extracts laser frequency comb spectrum (optional) | LFC FITS files |
| `freq_comb` | Applies frequency comb calibration (optional) | Master LFC spectrum |
| `rectify` | Rectifies 2D spectrum image (optional) | Orders + curvature |
| `science` | Optimally extracts 1D spectra from science frames | Science FITS files |
| `continuum` | Fits and normalizes the continuum | Extracted spectra |
| `finalize` | Applies corrections, writes final FITS output | All calibrations + spectra |

## Architecture

### API Layers

```
CLI (reduce command)     →  pyreduce/cli.py
     ↓
Pipeline (fluent API)    →  pyreduce/pipeline.py
     ↓
Step classes             →  pyreduce/reduce.py
     ↓
Core algorithms          →  pyreduce/extract.py, trace_orders.py, wavelength_calibration.py
     ↓
C extensions (CFFI)      →  pyreduce/clib/*.c
```

### Key Entry Points

- **CLI**: `uv run reduce <command>` - See `uv run reduce --help`
- **Python API**: `pyreduce.reduce.main()` or `Pipeline.from_files()`
- **Examples**: `examples/*.py` - Working scripts for each instrument

### Configuration

- **Instrument definitions**: `pyreduce/instruments/*.yaml` - Hardware parameters (detector, modes, headers)
- **Reduction settings**: `pyreduce/settings/settings_*.json` - Algorithm parameters per step
- **Pydantic models**: `pyreduce/instruments/models.py` - Validates instrument config

### Data Paths

- **Default data directory**: `~/REDUCE_DATA` (override with `$REDUCE_DATA` env var)
- **Input**: Raw FITS files in `base_dir/input_dir/`
- **Output**: Intermediate `.npz` files + final `.fits` spectra in `output_dir/`

### Runtime Options

- **Plot level**: `plot=0` (off), `plot=1` (basic), `plot=2` (detailed)
- **Environment override**: `PYREDUCE_PLOT=0` overrides script settings

## Development

### Commands

```bash
uv sync                              # Install dependencies
uv run pre-commit install            # Setup hooks (once)
uv run pytest -m unit                # Fast unit tests
uv run pytest --instrument=UVES      # Test single instrument
uv run ruff check --fix .            # Lint and fix
```

### Adding Instruments

1. Create `pyreduce/instruments/name.yaml` with detector/header config
2. Create `pyreduce/instruments/name.py` if custom logic needed (optional)
3. Create `pyreduce/settings/settings_NAME.json` for reduction parameters
4. Add example script to `examples/name_example.py`

See `examples/custom_instrument_example.py` for template.

### Test Organization

- `@pytest.mark.unit` - Fast tests with synthetic data (~40 tests)
- `@pytest.mark.instrument` - Integration tests with real data (~70 tests)
- `@pytest.mark.slow` - Long-running tests (wavecal, continuum)

## Build & Release

### Local Build

```bash
uv build                    # Build wheel
uv run pytest -m unit       # Verify
```

### Release (automated)

```bash
# Update version in pyproject.toml + CHANGELOG.md
git commit -am "Release vX.Y.Z"
git tag vX.Y.Z
git push origin master --tags
# GitHub Actions builds wheels and uploads to PyPI
```

## Key Files

| File | Purpose |
|------|---------|
| `pyreduce/pipeline.py` | Fluent Pipeline API, step orchestration |
| `pyreduce/reduce.py` | Step class implementations |
| `pyreduce/cli.py` | Click-based command line interface |
| `pyreduce/extract.py` | Optimal extraction algorithm |
| `pyreduce/wavelength_calibration.py` | Wavelength solution fitting |
| `pyreduce/trace_orders.py` | Order detection and tracing |
| `pyreduce/instruments/common.py` | Base Instrument class |
| `pyreduce/instruments/models.py` | Pydantic config models |
| `pyreduce/clib/*.c` | C code for slit function decomposition |
| `hatch_build.py` | CFFI extension build hook |
