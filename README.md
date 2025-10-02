[![CI](https://github.com/ivh/PyReduce/actions/workflows/python-publish.yml/badge.svg)](https://github.com/ivh/PyReduce/actions/workflows/python-publish.yml)
[![Documentation Status](https://readthedocs.org/projects/pyreduce-astro/badge/?version=latest)](https://pyreduce-astro.readthedocs.io/en/latest/?badge=latest)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

# PyReduce

PyReduce is a port of the [REDUCE](http://www.astro.uu.se/~piskunov/RESEARCH/REDUCE/) package to Python.
It is a complete data reduction pipeline for the echelle spectrographs, e.g. HARPS or UVES.

The methods are descibed in the papers
* Original REDUCE: Piskunov & Valenti (2001) [doi:10.1051/0004-6361:20020175](https://doi.org/10.1051/0004-6361:20020175)
* Updates to curved slit extraction and PyReduce: Piskunov, Wehrhahn & Marquart (2021) [10.1051/0004-6361/202038293](https://doi.org/10.1051/0004-6361/202038293)

Some documentation on how to use PyReduce is available at [ReadTheDocs](https://pyreduce-astro.readthedocs.io/en/latest/index.html).

Installation
------------

### For Users

The latest version can be installed with pip:

```bash
pip install git+https://github.com/ivh/PyReduce
```

The version available from PyPI is slightly outdated, but functional: ``pip install pyreduce-astro``.

### For Development

If you foresee making changes to PyReduce itself, clone the repository and use [uv](https://docs.astral.sh/uv/) for fast, modern package management:

```bash
git clone <your fork url>
cd PyReduce/
uv sync
```

This will automatically:
- Create a virtual environment
- Install all dependencies
- Build the CFFI C extensions
- Install PyReduce in editable mode

To run commands, use `uv run`:
```bash
uv run pytest                    # Run tests
uv run python examples/uves_example.py  # Run example
```

**Note:** PyReduce uses CFFI to link to C code. On non-Linux platforms you might need to install libffi.
See https://cffi.readthedocs.io/en/latest/installation.html#platform-specific-instructions for details.

Output Format
-------------
PyReduce will create ``.ech`` files when run. Despite the name those are just regular ``.fits`` files and can be opened with any programm that can read ``.fits``. The data is contained in a table extension. The header contains all the keywords of the input science file, plus some extra PyReduce specific keyword, all of which start with ``e_``.

How To
------
PyReduce is designed to be easy to use, but still be flexible.
``examples/uves_example.py`` is a good starting point, to understand how it works.
First we define the instrument, target, night, and instrument mode (if applicable) of our reduction. Then we tell PyReduce where to find the data, and lastly we define all the specific settings of the reduction (e.g. polynomial degrees of various fits) in a json configuration file.
We also define which steps of the reduction to perform. Steps that are not specified, but are still required, will be loaded from previous runs if possible, or executed otherwise.
All of this is then passed to pyreduce.reduce.main to start the reduction.

In this example, PyReduce will plot all intermediary results, and also plot the progres during some of the steps. Close them to continue calculations, if it seems nothing is happening. Once you are statisified with the results you can disable them in settings_UVES.json (with "plot":false in each step) to speed up the computation.
