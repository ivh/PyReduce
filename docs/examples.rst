Examples
========

PyReduce includes example scripts for each supported instrument in the ``examples/`` directory.

Running an Example
------------------

The UVES example is a good starting point::

    # Download sample data
    uv run reduce download UVES

    # Run the example
    uv run python examples/uves_example.py

Or use the CLI directly::

    uv run reduce run UVES "HD[- ]?132205" --steps bias,flat,orders,science

Example Structure
-----------------

Each example script follows the same pattern:

.. code-block:: python

    from pyreduce.pipeline import Pipeline
    from pyreduce import datasets

    # Define parameters
    instrument = "UVES"
    target = "HD132205"
    night = "2010-04-01"
    arm = "middle"
    steps = ("bias", "flat", "orders", "science")

    # Download/locate data
    base_dir = datasets.UVES()

    # Run pipeline
    Pipeline.from_instrument(
        instrument,
        target,
        night=night,
        arm=arm,
        steps=steps,
        base_dir=base_dir,
        plot=1,
    ).run()

Modifying Steps
---------------

Edit the ``steps`` tuple to control which reduction steps run:

.. code-block:: python

    steps = (
        "bias",
        "flat",
        "orders",
        # "curvature",    # Skip curvature
        # "scatter",      # Skip scatter
        "norm_flat",
        "wavecal",
        "science",
        # "continuum",    # Skip continuum
        "finalize",
    )

Steps not in the list but required as dependencies will be loaded from
previous runs if the output files exist.

Available Examples
------------------

- ``uves_example.py`` - ESO UVES
- ``harps_example.py`` - ESO HARPS
- ``xshooter_example.py`` - ESO XSHOOTER
- ``crires_plus_example.py`` - ESO CRIRES+
- ``jwst_niriss_example.py`` - JWST NIRISS
- ``jwst_miri_example.py`` - JWST MIRI
- ``nirspec_example.py`` - Keck NIRSPEC
- ``mcdonald_example.py`` - McDonald Observatory
- ``lick_apf_example.py`` - Lick APF
- ``custom_instrument_example.py`` - Template for new instruments
