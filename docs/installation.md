# Installation

## Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is the recommended way to install PyReduce:

```bash
uv add pyreduce-astro
```

Or to install globally:

```bash
uv tool install pyreduce-astro
```

## Using pip

```bash
pip install pyreduce-astro
```

## For Development

Clone the repository and use uv:

```bash
git clone https://github.com/ivh/PyReduce
cd PyReduce/
uv sync
```

This will automatically:

- Create a virtual environment
- Install all dependencies
- Build the CFFI C extensions
- Install PyReduce in editable mode

To run commands:

```bash
uv run reduce --help              # CLI
uv run pytest -m unit             # Tests
uv run python examples/uves_example.py
```

### Building C Extensions

The C extensions are built automatically during `uv sync`. To manually rebuild them:

```bash
uv run reduce-build               # Build C extensions
uv run reduce-clean               # Remove compiled extensions
```

This is needed after re-vendoring the C slit decomposition with `uv run python tools/sync_slitdec.py ../SlitDec.git`; `pyreduce/clib/slitdec.{c,h}` are not edited here.

## Platform Notes

PyReduce uses CFFI to link to C code. On non-Linux platforms you may need to install libffi.
See https://cffi.readthedocs.io/en/latest/installation.html#platform-specific-instructions for details.

## Running Without the C Extension

The extraction algorithm also exists as a pure-Python (numpy/scipy) port, so a failed
build or a platform with no prebuilt wheel is not a dead end. Select it with
`PYREDUCE_EXTRACTION` (or the `--extraction` CLI option):

```bash
PYREDUCE_EXTRACTION=numpy uv run reduce run UVES -t HD132205
```

It needs nothing beyond the core dependencies (numpy and scipy) and runs at
~1.8-2.1x the C extension. It implements the same algorithm as the C and is tested
against it directly: they agree to a few 1e-13 in float64, and the written FITS
products are bit-identical.

There is no automatic fallback — with the extension missing, the default
`PYREDUCE_EXTRACTION=c` raises `ModuleNotFoundError` at the first extraction, so set
the variable explicitly. The backend in use is logged once per run at INFO level.
