How To Use PyReduce
===================

PyReduce offers two ways to run reductions: the command-line interface (CLI) and the Python API.

Command Line Interface
----------------------

The simplest way to run a reduction::

    # Download sample data
    uv run reduce download UVES

    # Run full pipeline
    uv run reduce run UVES HD132205 --steps bias,flat,orders,science

    # Run individual steps
    uv run reduce bias UVES HD132205
    uv run reduce orders UVES HD132205

    # List available steps
    uv run reduce list-steps

CLI options::

    uv run reduce run UVES HD132205 \
        --night 2010-04-01 \
        --arm middle \
        --steps bias,flat,orders,science \
        --base-dir /data \
        --plot 1

Python API
----------

The recommended Python entry point is ``Pipeline.from_instrument()``:

.. code-block:: python

    from pyreduce.pipeline import Pipeline

    result = Pipeline.from_instrument(
        instrument="UVES",
        target="HD132205",
        night="2010-04-01",
        arm="middle",
        steps=("bias", "flat", "orders", "science"),
        plot=1,
    ).run()

This handles:

- Loading the instrument configuration
- Finding and sorting input files
- Setting up output directories
- Running the requested steps

For more control, construct a Pipeline manually:

.. code-block:: python

    from pyreduce.pipeline import Pipeline

    pipe = Pipeline(
        instrument="UVES",
        output_dir="/data/reduced",
        arm="middle",
    )
    pipe.bias(bias_files)
    pipe.flat(flat_files)
    pipe.trace_orders()
    pipe.extract(science_files)
    result = pipe.run()

Environment Variables
---------------------

- ``REDUCE_DATA`` - Base data directory (default: ``~/REDUCE_DATA``)
- ``PYREDUCE_PLOT`` - Override plot level (0=off, 1=basic, 2=detailed)
- ``PYREDUCE_PLOT_DIR`` - Save plots to directory instead of displaying

Pipeline Steps
--------------

Steps are run in dependency order. Available steps:

- ``bias`` - Combine bias frames
- ``flat`` - Combine flat frames
- ``orders`` - Trace echelle orders on flat
- ``curvature`` - Measure slit tilt/shear
- ``scatter`` - Model inter-order background
- ``norm_flat`` - Normalize flat, extract blaze
- ``wavecal_master`` - Extract wavelength calibration spectrum
- ``wavecal_init`` - Initial line identification
- ``wavecal`` - Refine wavelength solution
- ``freq_comb_master`` - Extract frequency comb spectrum
- ``freq_comb`` - Apply frequency comb calibration
- ``science`` - Optimally extract science spectra
- ``continuum`` - Normalize continuum
- ``finalize`` - Write final output

Steps not explicitly requested but required as dependencies will be loaded
from previous runs if available, or executed automatically.
