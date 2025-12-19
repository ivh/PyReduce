# PyReduce Documentation

PyReduce is a data reduction pipeline for echelle spectrographs. It processes
raw FITS observations into calibrated 1D spectra.

Supported instruments include HARPS, HARPS-N, UVES, XSHOOTER, CRIRES+, JWST/NIRISS, JWST/MIRI, and more.

## Quick Start

```bash
# Install
uv add pyreduce-astro

# Download sample data
uv run reduce download UVES

# Run reduction
uv run reduce run UVES HD132205 --steps bias,flat,orders,science
```

Or use Python:

```python
from pyreduce.pipeline import Pipeline

Pipeline.from_instrument(
    instrument="UVES",
    target="HD132205",
    arm="middle",
).run()
```

## Contents

```{toctree}
:maxdepth: 2

installation
howto
cli
examples
configuration_file
instruments
wavecal_linelist
modules
```

## Development

```{toctree}
:maxdepth: 1

redesign
fiber_bundle_tracing
```
