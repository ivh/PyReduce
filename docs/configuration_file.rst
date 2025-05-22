PyReduce Configuration
======================

All free parameters of the PyReduce pipeline are defined in a configuration
file, that is passed to the main function. If certain values are not explicitly
defined, default values will be used, which may or may not work well.
Some configurations for common instruments are provided in the examples
directory.

All input is validated using the jsonschema ``settings/settings_schema.json``.

Configuring Multi-Fiber Layouts
-------------------------------

For instruments that simultaneously observe multiple fibers, PyReduce allows defining their
relative physical layout on the detector. This is configured via the ``fiber_layout``
object, which is part of the main ``instrument`` block in an instrument's specific
settings JSON file (e.g., ``settings_MYINSTRUMENT.json``).

The ``fiber_layout`` configuration helps PyReduce to:

*   Identify and trace individual fibers based on primary traces found by the order tracing step.
*   Apply appropriate spatial and spectral offsets to these primary traces.
*   Produce separate data products for each configured fiber.

JSON Structure
~~~~~~~~~~~~~~

The core of the configuration is the ``physical_order_groups`` array. Each element in
this array represents a group of fibers whose traces are expected to be physically
close on the detector (e.g., within the same echelle order footprint).

Inside each group, a ``fibers`` array lists individual fibers, each with:

*   ``id`` (string, required): A unique identifier for the fiber (e.g., "SCI", "SKY1").
    This ID is used in output filenames and FITS headers.
*   ``spatial_offset`` (number, required): The offset in the spatial (cross-dispersion)
    direction in pixels, relative to a reference fiber within the same group.
    Positive values typically mean upwards (increasing row index).
*   ``spectral_offset`` (number, optional): The offset in the spectral (dispersion)
    direction in pixels, relative to the reference fiber. Positive values
    typically mean towards the right (increasing column index). Defaults to ``0.0``.

One fiber in each ``physical_order_group`` **must** have ``spatial_offset: 0.0``
(and implicitly ``spectral_offset: 0.0`` if not specified). This acts as the
**reference fiber** for that group. The primary trace found by ``trace_orders.py``
for that group is assumed to be this reference fiber's trace.

Mapping to Primary Traces
~~~~~~~~~~~~~~~~~~~~~~~~~
The ``physical_order_groups`` in your JSON configuration are mapped **sequentially**
to the primary traces found by PyReduce's order tracing step (`trace_orders.py`).
For example, the first object in `physical_order_groups` will correspond to the first
primary trace identified, the second object to the second trace, and so on.

Example ``fiber_layout`` Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is an example of how ``fiber_layout`` would be structured within an
instrument's settings file:

.. code-block:: json

  {
    "__instrument__": "MyMultiFiberSpec",
    "reduce": {
      // ... reduce settings ...
    },
    "instrument": {
      "gain": 1.2,
      "readnoise": 3.5,
      // ... other instrument parameters ...
      "fiber_layout": {
        "physical_order_groups": [
          {
            "fibers": [
              {
                "id": "SCI_CENTER",
                "spatial_offset": 0.0
              },
              {
                "id": "SKY_ABOVE",
                "spatial_offset": 50.7,
                "spectral_offset": -0.5
              },
              {
                "id": "SKY_BELOW",
                "spatial_offset": -48.3,
                "spectral_offset": 0.2
              }
            ]
          },
          {
            "fibers": [
              {
                "id": "CALIB_FIBER",
                "spatial_offset": 0.0
              }
            ]
          }
        ]
      }
    },
    "bias": {
      // ... bias settings ...
    },
    "orders":{
        // ... order tracing settings, ensure it finds one trace per physical_order_group ...
    }
    // ... other step settings ...
  }

For more detailed information on the workflow, data propagation, FITS header keywords,
and assumptions related to this feature, please refer to the
:doc:`fiber_layout_specification` document.

Default Configuration Values
----------------------------

The default values for all PyReduce settings are defined in
``pyreduce/settings/settings_pyreduce.json``:

.. literalinclude:: /../pyreduce/settings/settings_pyreduce.json
    :language: json
    :caption: settings_pyreduce.json
