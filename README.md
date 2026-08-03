[![CI](https://github.com/ivh/PyReduce/actions/workflows/python-publish.yml/badge.svg)](https://github.com/ivh/PyReduce/actions/workflows/python-publish.yml)
[![Documentation Status](https://readthedocs.org/projects/pyreduce-astro/badge/?version=latest)](https://pyreduce-astro.readthedocs.io/en/latest/?badge=latest)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

# What's new?

Version 0.9 adds two pure-Python extraction backends, so PyReduce can run without a compiled C extension, and replaces `PYREDUCE_USE_CHARSLIT` with `PYREDUCE_EXTRACTION` (`c`, `charslit`, `numba`, `numpy`). See [CHANGELOG.md](CHANGELOG.md) for all changes.

# PyReduce

A data reduction pipeline for echelle spectrographs (HARPS, UVES, XSHOOTER, CRIRES+, JWST/NIRISS, ANDES, MOSAIC, NEID, and more).

Based on the [REDUCE](http://www.astro.uu.se/~piskunov/RESEARCH/REDUCE/) package. See the papers:
- Piskunov & Valenti (2001) [doi:10.1051/0004-6361:20020175](https://doi.org/10.1051/0004-6361:20020175)
- Piskunov, Wehrhahn & Marquart (2021) [doi:10.1051/0004-6361/202038293](https://doi.org/10.1051/0004-6361/202038293)

## Installation

```bash
# Using uv (recommended)
uv add pyreduce-astro

# Or pip
pip install pyreduce-astro
```

For development:
```bash
git clone https://github.com/ivh/PyReduce
cd PyReduce
uv sync
uv run reduce-build
```

### No wheel for your platform, or the C extension won't build?

Extraction also runs in pure Python. Set `PYREDUCE_EXTRACTION=numpy` (or pass
`--extraction numpy`) and nothing needs compiling — numpy and scipy are already
core dependencies:

```bash
PYREDUCE_EXTRACTION=numpy uv run reduce run UVES -t HD132205
```

It is ~2x slower than the C extension and produces bit-identical output files.
`PYREDUCE_EXTRACTION=numba` is ~1.5-1.8x with `uv sync --extra numba` installed.
See [Extraction Backends](https://pyreduce-astro.readthedocs.io/en/latest/howto.html#extraction-backends).

## Quick Start

```bash
# Download sample data
uv run reduce download UVES

# Run reduction
uv run reduce run UVES -t HD132205 --steps bias,flat,trace,science

# Or run individual steps
uv run reduce bias UVES -t HD132205
uv run reduce flat UVES -t HD132205
```

Or use the Python API:
```python
from pyreduce.pipeline import Pipeline

Pipeline.from_instrument(
    instrument="UVES",
    target="HD132205",
    night="2010-04-01",
    channel="middle",
    steps=("bias", "flat", "trace", "science"),
).run()
```

## Plotting

Control plotting with environment variables:

```bash
# Save plots to files (headless/CI)
PYREDUCE_PLOT=1 PYREDUCE_PLOT_DIR=/tmp/plots PYREDUCE_PLOT_SHOW=off uv run reduce run ...

# Show all plots at end (browser via webagg)
MPLBACKEND=webagg PYREDUCE_PLOT=1 PYREDUCE_PLOT_SHOW=defer uv run reduce run ...
```

See [How To](https://pyreduce-astro.readthedocs.io/en/latest/howto.html#plot-modes) for details.

## Documentation

Full documentation at [ReadTheDocs](https://pyreduce-astro.readthedocs.io/).

## Output

PyReduce creates `.fits` files (standard FITS with binary table extension). Headers include original keywords plus PyReduce-specific ones prefixed with `e_`.
