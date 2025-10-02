Installation
============

For Users
---------

The latest version can be installed with pip::

    pip install git+https://github.com/ivh/PyReduce

The version available from PyPI is slightly outdated, but functional::

    pip install pyreduce-astro

For Development
---------------

If you foresee making changes to PyReduce itself, clone the repository and use `uv <https://docs.astral.sh/uv/>`_ for fast, modern package management::

    git clone <your fork url>
    cd PyReduce/
    uv sync

This will automatically:

- Create a virtual environment
- Install all dependencies
- Build the CFFI C extensions
- Install PyReduce in editable mode

To run commands, use ``uv run``::

    uv run pytest                           # Run tests
    uv run python examples/uves_example.py  # Run example

**Note:** PyReduce uses CFFI to link to C code. On non-Linux platforms you might need to install libffi.
See https://cffi.readthedocs.io/en/latest/installation.html#platform-specific-instructions for details.
