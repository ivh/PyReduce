![Python application](https://github.com/ivh/PyReduce/workflows/Python%20application/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pyreduce-astro/badge/?version=latest)](https://pyreduce-astro.readthedocs.io/en/latest/?badge=latest)
[![Updates](https://pyup.io/repos/github/ivh/PyReduce/shield.svg)](https://pyup.io/repos/github/ivh/PyReduce/)

# PyReduce

PyReduce is a port of the [REDUCE](http://www.astro.uu.se/~piskunov/RESEARCH/REDUCE/) package to Python.
It is a complete data reduction pipeline for the echelle spectrographs, e.g. HARPS or UVES.

The methods are descibed in the papers
* Original REDUCE: Piskunov & Valenti (2001) [doi:10.1051/0004-6361:20020175](https://doi.org/10.1051/0004-6361:20020175)
* Updates to curved slit extraction and PyReduce: Piskunov, Wehrhahn & Marquart (2021) [10.1051/0004-6361/202038293](https://doi.org/10.1051/0004-6361/202038293)

Some documentation on how to use PyReduce is available at [ReadTheDocs](https://pyreduce-astro.readthedocs.io/en/latest/index.html).

Installation
------------

The latest version can be installed using ``pip install git+https://github.com/ivh/PyReduce``.

The version that is available from PyPI is slightly outdated, but functional and installable via ``pip install pyreduce-astro``.

If you foresee making changes to PyReduce itself, feel free to (fork and) clone
this repository, and install the requirements and your local copy into a fresh
environment, e.g.

```
conda create -n pyreduce
conda activate pyreduce
git clone <your fork url>
cd PyReduce/
pip install -r requirements.txt
pip install -e .
```

PyReduce uses CFFI to link to the C code, on non-linux platforms you might have to install libffi.
See also https://cffi.readthedocs.io/en/latest/installation.html#platform-specific-instructions for details.

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

