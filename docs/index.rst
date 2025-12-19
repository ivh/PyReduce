PyReduce Documentation
======================

PyReduce is a data reduction pipeline for echelle spectrographs. It processes
raw FITS observations into calibrated 1D spectra.

Supported instruments include HARPS, UVES, XSHOOTER, CRIRES+, JWST/NIRISS, and more.

Quick Start
-----------

.. code-block:: bash

    # Install
    uv add pyreduce-astro

    # Download sample data
    uv run reduce download UVES

    # Run reduction
    uv run reduce run UVES HD132205 --steps bias,flat,orders,science

Or use Python:

.. code-block:: python

    from pyreduce.pipeline import Pipeline

    Pipeline.from_instrument(
        instrument="UVES",
        target="HD132205",
        arm="middle",
    ).run()

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   howto
   examples
   configuration_file
   instruments
   wavecal_linelist
   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
