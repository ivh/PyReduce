Configuration
=============

PyReduce uses two types of configuration files:

- **Instrument configs** (YAML) - Define the instrument hardware and header mappings
- **Reduction settings** (JSON) - Define algorithm parameters for each step

Reduction Settings
------------------

Location: ``pyreduce/settings/settings_*.json``

These control HOW the reduction is performed - polynomial degrees, thresholds,
extraction parameters, etc.

.. code-block:: json

    {
      "bias": {
        "degree": 0
      },
      "orders": {
        "degree": 4,
        "noise": 100,
        "min_cluster": 500,
        "filter_size": 120
      },
      "science": {
        "extraction_method": "optimal",
        "extraction_width": 0.5,
        "oversampling": 10
      }
    }

Settings are loaded in order:

1. ``settings_pyreduce.json`` - Base defaults
2. ``settings_INSTRUMENT.json`` - Instrument-specific overrides
3. Runtime overrides via ``configuration`` parameter

To override settings at runtime:

.. code-block:: python

    from pyreduce.configuration import get_configuration_for_instrument

    config = get_configuration_for_instrument("UVES")
    config["orders"]["degree"] = 5
    config["science"]["oversampling"] = 8

    Pipeline.from_instrument(
        instrument="UVES",
        ...,
        configuration=config,
    ).run()

Instrument Configs
------------------

Location: ``pyreduce/instruments/*.yaml``

These define WHAT the instrument is - detector properties, header keyword
mappings, file classification patterns.

.. code-block:: yaml

    # Identity
    instrument: HARPS
    telescope: ESO-3.6m
    arms: [red, blue]

    # Detector
    naxis: [4096, 4096]
    orientation: 4
    extension: 0
    gain: ESO DET OUT1 CONAD
    readnoise: ESO DET OUT1 RON

    # Header mappings
    date: DATE-OBS
    target: ESO OBS TARG NAME
    exposure_time: EXPTIME

    # File classification
    kw_bias: ESO DPR TYPE
    id_bias: BIAS
    kw_flat: ESO DPR TYPE
    id_flat: FLAT.*

Instrument configs are validated by Pydantic models at load time.
See ``pyreduce/instruments/models.py`` for the full schema.

Default Settings
----------------

The default values are defined in:

.. literalinclude:: /../pyreduce/settings/settings_pyreduce.json
    :language: json
    :caption: settings_pyreduce.json
