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

This is useful after modifying the C source files in `pyreduce/clib/`.

## Platform Notes

PyReduce uses CFFI to link to C code. On non-Linux platforms you may need to install libffi.
See https://cffi.readthedocs.io/en/latest/installation.html#platform-specific-instructions for details.
