PyReduce Configuration
======================

All free parameters of the PyReduce pipeline are defined in a configuration
file, that is passed to the main function. If certain values are not explicitly
defined, default values will be used, which may or may not work well.
Some configurations for common instruments are provided in the examples
directory.

All input is validated using the jsonschema ``settings/settings_schema.json``.

The default values are defined as:

.. literalinclude:: /../pyreduce/settings/settings_pyreduce.json
    :language: json
    :caption: settings_pyreduce.json
