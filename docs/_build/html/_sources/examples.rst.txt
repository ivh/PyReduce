Examples
========

The PyReduce distribution includes one example dataset for the UVES
spectrograph. If everything is set up correctly, it can be run by simply
calling uves_example.py in the examples directory.

>>> python examples/uves_example.py

This will download the necessary data and run all extraction steps. Inside the
script, the individual steps that are executed can be changed by modyfying the
"steps" list, e.g. by commenting some entries out.

>>> steps = (
    "bias",
    "flat",
    # "orders",
    # "norm_flat",
    # "wavecal",
    # "shear",
    # "science",
    # "continuum",
    # "finalize",
    )

The output files will be placed in /examples/dataset/UVES/...
