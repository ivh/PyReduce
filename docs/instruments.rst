Instruments
===========

PyReduce supports many instruments out of the box. Custom instruments can be added
by creating YAML configuration files.

Supported Instruments
---------------------

- **ESO**: HARPS, HARPS-N, UVES, XSHOOTER, CRIRES+
- **Space**: JWST NIRISS, JWST MIRI
- **Ground**: Keck NIRSPEC, Lick APF, McDonald

Adding a Custom Instrument
--------------------------

Method 1: YAML Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a YAML file defining your instrument:

.. code-block:: yaml

    # myinstrument.yaml
    instrument: MyInstrument
    telescope: MyTelescope
    arms: [default]

    # Detector
    naxis: [2048, 2048]
    orientation: 0
    extension: 0
    gain: 1.0
    readnoise: 5.0

    # Header keywords
    date: DATE-OBS
    target: OBJECT
    exposure_time: EXPTIME

    # File classification
    kw_bias: IMAGETYP
    id_bias: bias
    kw_flat: IMAGETYP
    id_flat: flat
    kw_spec: IMAGETYP
    id_spec: object

Load it directly:

.. code-block:: python

    from pyreduce.instruments import load_instrument

    inst = load_instrument("/path/to/myinstrument.yaml")

Method 2: Python Class
^^^^^^^^^^^^^^^^^^^^^^

For instruments needing custom logic, create a Python class:

.. code-block:: python

    from pyreduce.instruments.common import Instrument

    class MyInstrument(Instrument):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Custom initialization

        def add_header_info(self, header):
            # Compute derived header values
            header["MJD_MID"] = header["MJD_OBS"] + header["EXPTIME"] / 86400 / 2
            return header

        def get_wavecal_filename(self, header, arm, **kwargs):
            # Return path to wavelength calibration file
            return f"wavecal_{arm}.npz"

Place the YAML in ``pyreduce/instruments/`` and the Python file alongside it.

Instrument Configuration Fields
-------------------------------

Required fields:

- ``instrument`` - Instrument name
- ``telescope`` - Telescope name
- ``arms`` - List of instrument arms/modes
- ``naxis`` - Detector dimensions [x, y]

Detector properties:

- ``extension`` - FITS extension (0 or name)
- ``orientation`` - Rotation/flip code (0-7)
- ``gain`` - Detector gain (value or header keyword)
- ``readnoise`` - Read noise (value or header keyword)
- ``dark`` - Dark current (value or header keyword)
- ``prescan_x``, ``overscan_x`` - Prescan/overscan regions

Header mappings (instrument keyword -> internal name):

- ``date`` - Observation date
- ``target`` - Target name
- ``exposure_time`` - Exposure time
- ``ra``, ``dec`` - Coordinates
- ``jd`` - Julian date
- ``instrument_mode`` - Instrument mode

File classification:

- ``kw_bias``, ``id_bias`` - Bias file keyword and pattern
- ``kw_flat``, ``id_flat`` - Flat file keyword and pattern
- ``kw_wave``, ``id_wave`` - Wavelength calibration keyword and pattern
- ``kw_spec``, ``id_spec`` - Science file keyword and pattern

See ``pyreduce/instruments/models.py`` for the complete schema.

Reduction Settings
------------------

Each instrument can have its own settings file at ``pyreduce/settings/settings_INSTRUMENT.json``.
This overrides the defaults for that instrument. See :doc:`configuration_file` for details.
