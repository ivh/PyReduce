import os.path
import logging
import json
import jsonschema


def load_config(configuration, instrument, j=0):
    if configuration is None:
        config = "settings_%s.json" % instrument.upper()
    elif isinstance(configuration, dict):
        if instrument in configuration.keys():
            config = configuration[instrument]
        elif "__instrument__" in configuration.keys() and configuration["__instrument__"] == instrument.upper():
            config = configuration
        else:
            raise KeyError("This configuration is for a different instrument")
    elif isinstance(configuration, list):
        config = configuration[j]
    elif isinstance(configuration, str):
        config = configuration

    if isinstance(config, str):
        if os.path.isfile(config):
            logging.info("Loading configuration from %s", config)
            with open(config) as f:
                config = json.load(f)
        else:
            logging.warning(
                "No configuration found at %s, using default values", config
            )
            config = {}

    # Load general settings
    settings = read_config()
    nparam1 = count_parameters(settings)
    # Update values with given settings
    for key in settings.keys():
        if key in config.keys():
            if isinstance(settings[key], dict):
                settings[key].update(config[key])
            else:
                settings[key] = config[key]
    nparam2 = count_parameters(settings, exclude="instrument")
    if nparam2 > nparam1:
        logging.warning("New parameter(s) in instrument config, Check spelling!")

    # If it doesn't raise an Exception everything is as expected
    validate_config(settings)
    logging.debug("Configuration succesfully validated")

    return settings


def read_config(fname="settings_pyreduce.json"):
    this_dir = os.path.dirname(__file__)
    fname = os.path.join(this_dir, "settings", fname)

    if os.path.exists(fname):
        with open(fname) as file:
            settings = json.load(file)
            return settings
    else:
        raise FileNotFoundError(f"Settings file {fname} not found")


def validate_config(config):
    fname = "settings_schema.json"
    this_dir = os.path.dirname(__file__)
    fname = os.path.join(this_dir, "settings", fname)

    with open(fname) as f:
        schema = json.load(f)
    try:
        jsonschema.validate(schema=schema, instance=config)
    except jsonschema.ValidationError as ve:
        logging.error("Configuration failed validation check.\n%s", ve.message)
        raise ve


def count_parameters(config, exclude=()):
    nparam = 0

    for key, value in config.items():
        if isinstance(value, dict) and key not in exclude:
            nparam += count_parameters(value)
        elif key not in exclude:
            nparam += 1

    return nparam
