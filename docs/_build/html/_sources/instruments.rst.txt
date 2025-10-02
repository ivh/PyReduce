Supporting Custom instruments
=============================


PyReduce supports a number of instruments by default, but it can be extended to support your instrument too.
There are two ways to implement your instrument. The first works on a script by script basis. The second is a more complete ompelementation.
The second choice is recommended for proper use, although the first can be useful for experimentation.


Method 1: Create your instrument as part of the script
------------------------------------------------------

There is a very simple example in the examples folder for the support of a custom instrument "custom_instrument_example.py".
The idea is that we create a custom class on the fly and define all the necessary parameters within the script that is currently being run.
This is in general less flexible than the second method, but can be useful since all variables are available within the script itself.
Once a good solution has been found it is recommended to convert this to what is described below in method 2, since we can then use the same
instrument configuration in other scripts.


Method 2: Create the instrument class and configuration
-------------------------------------------------------

In this method, we create a custom class and configuration as seperate files, similar to how instruments are implemented in PyReduce itself.
The general instrument is defined by the pyreduce.instruments.common.Instrument class, so the easiest way is to create your own class that inherits from it.

Here are the general steps that need to be followed to support your instrument:

    - You need to implement your own version of the instrument class, that inherits from pyreduce.instruments.common.Instrument
    - In that class there are a few important functions that may need to be adapted:

        - load_info, this loads a json file with information mostly about the FITS header (e.g. which keyword gives us the time of the observation etc.) Look at other pyreduce.instruments.instrument_schema to get some information about what is what
        - sort_files, finds and sorts the files for the different steps of pyreduce. There is a system here that might work for your instrument as well, depending on the FITS headers. In that case you just need to set the kw_bias etc fields in load_info correctly
        - add_header_info, this modifies the fits header information. Usually to combine fields in the header to get the correct information (e.g. to get the time in the middle of the observation, instead of at the beginning).
        - get_wavecal_filename, should return the filename to the wavelength calibration first guess. See the Wavelength Calibration section on how to create this file.
        - get_wavelength_range, this returns an initial guess for the wavelength of each order, if the initial first guess file from get_wavecal_filename is not provided.
        - (optional) get_mask_filename, should return the filename of the bad pixel map. A fits file with the badpixel map in the main extension. With the same size as the input fits files

    - You probably also want to override the settings used by PyReduce (config in the example scripts). You can find examples for settings in pyreduce.settings. (settings_pyreduce.json has all available settings, they all need to be specified)
    - When calling PyReduce instead of passing the name of the instrument you pass an instance of your instrument class.
