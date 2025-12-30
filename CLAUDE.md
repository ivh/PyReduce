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
uv run reduce run UVES HD132205 --steps bias,flat,orders,science
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
│   ├── instrument_info.py  # Instrument loader
│   ├── *.yaml           # Instrument configs (one per instrument)
│   └── *.py             # Custom instrument logic (optional)
│
├── settings/            # Reduction parameters
│   ├── settings_default.json
│   └── settings_*.json  # Per-instrument settings
│
└── clib/                # C source for extraction
    ├── slit_func_bd.c
    └── slit_func_2d_xi_zeta_bd.c
```

## Pipeline Steps

The reduction pipeline consists of these steps (in typical order):

| Step | Class | Description |
|------|-------|-------------|
| `mask` | `Mask` | Load bad pixel mask for detector |
| `bias` | `Bias` | Combine bias frames into master bias |
| `flat` | `Flat` | Combine flat frames, subtract bias |
| `orders` | `OrderTracing` | Trace echelle order positions on flat |
| `curvature` | `SlitCurvatureDetermination` | Measure slit tilt/shear from arc lamp |
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

Location: `pyreduce/instruments/*.yaml`

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

Location: `pyreduce/settings/settings_*.json`

Defines HOW to reduce - algorithm parameters per step:

```json
{
  "bias": {
    "degree": 0
  },
  "orders": {
    "degree": 4,
    "noise": 100,
    "min_cluster": 500,
    "filter_size": 120
  },
  "norm_flat": {
    "extraction_width": 0.5,
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
    "extraction_width": 0.5,
    "oversampling": 10
  }
}
```

Settings cascade: `settings_default.json` < `settings_INSTRUMENT.json` < runtime overrides.

## Python API

### Recommended: Pipeline.from_instrument()

```python
from pyreduce.pipeline import Pipeline

result = Pipeline.from_instrument(
    instrument="UVES",
    target="HD132205",
    night="2010-04-01",
    channel="middle",
    steps=("bias", "flat", "orders", "science"),
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
uv run reduce run UVES HD132205 --steps bias,flat,orders

# Individual steps (top-level commands)
uv run reduce bias UVES HD132205
uv run reduce orders UVES HD132205
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
- `PYREDUCE_PLOT` - Override plot level (0, 1, 2)
- `PYREDUCE_PLOT_DIR` - Save plots to directory instead of displaying

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
