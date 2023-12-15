How To use PyReduce
===================

Using PyReduce is easy. Simply specify the instrument and where to find the
data, and the rest should take care of itself. Of course you can also set all
parameters yourself if you want more control over what is happening.

A good starting point is the examples section. But here is what you need.

>>> import pyreduce
>>> from pyreduce import datasets

Define parameters

>>> # The instrument name as specified in the supported instruments
>>> instrument = "UVES"
>>> # The name of the observation target as specified in the file structure
>>> target = "HD132205"
>>> # The observation night as a string, as specified in the file structure
>>> night = "2010-04-02"
>>> # The instrument mode/setting that is used, depends on the instrument
>>> mode = "middle"
>>> # The data reduction steps to run
>>> steps = (
    "bias",
    "flat",
    "orders",
    "norm_flat",
    "wavecal",
    "freq_comb",
    "shear",
    "science",
    "continuum",
    "finalize",
    )

Some basic settings
Expected Folder Structure: ``base_dir/datasets/HD132205/*.fits.gz``
Feel free to change this to your own preference, values in curly brackets
will be replaced with the actual values {}

>>> input_dir = "{target}/"
>>> output_dir = "reduced/{instrument}/{target}/{night}/{mode}"

Load dataset (and save the location)

For the example dataset use

>>> base_dir = datasets.UVES_HD132205()

For your own observations set base_dir to the path that points to your files.
Note that the full path is given by base_dir + input_dir / output_dir.
If these are completely independant you can set base_dir = "" instead.

>>> base_dir = "your-file-path-here"

Start the extraction

>>> pyreduce.reduce.main(
    instrument,
    target,
    night,
    mode,
    steps,
    base_dir=base_dir,
    input_dir=input_dir,
    output_dir=output_dir,
    configuration="settings_UVES.json",
    )
