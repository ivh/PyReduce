Installation
============

Using uv (Recommended)
----------------------

`uv <https://docs.astral.sh/uv/>`_ is the recommended way to install PyReduce::

    uv add pyreduce-astro

Or to install globally::

    uv tool install pyreduce-astro

Using pip
---------

::

    pip install pyreduce-astro

For Development
---------------

Clone the repository and use uv::

    git clone https://github.com/ivh/PyReduce
    cd PyReduce/
    uv sync

This will automatically:

- Create a virtual environment
- Install all dependencies
- Build the CFFI C extensions
- Install PyReduce in editable mode

To run commands::

    uv run reduce --help              # CLI
    uv run pytest -m unit             # Tests
    uv run python examples/uves_example.py

Platform Notes
--------------

PyReduce uses CFFI to link to C code. On non-Linux platforms you may need to install libffi.
See https://cffi.readthedocs.io/en/latest/installation.html#platform-specific-instructions for details.
