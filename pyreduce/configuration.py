"""Loads configuration files

This module loads json configuration files from disk,
and combines them with the default settings,
to create one dict that contains all parameters.
It also checks that all parameters exists, and that
no new parameters have been added by accident.
"""

import json
import logging
from os.path import dirname, exists, join

import jsonschema

logger = logging.getLogger(__name__)

if int(jsonschema.__version__[0]) < 3:  # pragma: no cover
    logger.warning(
        "Jsonschema %s found, but at least 3.0.0 is required to check configuration. Skipping the check.",
        jsonschema.__version__,
    )
    hasJsonSchema = False
else:
    hasJsonSchema = True


def get_configuration_for_instrument(instrument, **kwargs):
    local = dirname(__file__)
    instrument = str(instrument)
    if instrument in ["pyreduce", "defaults", None]:
        fname = join(local, "instruments", "defaults", "settings.json")
    else:
        fname = join(local, "instruments", instrument.upper(), "settings.json")

    config = load_config(fname, instrument)

    for kwarg_key, kwarg_value in kwargs.items():
        for key, _value in config.items():
            if isinstance(config[key], dict) and kwarg_key in config[key].keys():
                config[key][kwarg_key] = kwarg_value

    return config


def load_settings_override(config, settings_file):
    """Apply settings overrides from a JSON file.

    Parameters
    ----------
    config : dict
        Base configuration to override
    settings_file : str
        Path to JSON file with override settings

    Returns
    -------
    config : dict
        Updated configuration
    """
    with open(settings_file) as f:
        overrides = json.load(f)
    return update(config, overrides, check=False)


def _resolve_inheritance(config, seen=None):
    """Recursively resolve __inherits__ chain.

    Parameters
    ----------
    config : dict
        Configuration dict, possibly with __inherits__ key
    seen : set, optional
        Set of already-visited parent names for cycle detection

    Returns
    -------
    dict
        Fully resolved configuration with all inherited values merged
    """
    if seen is None:
        seen = set()

    parent_name = config.pop("__inherits__", "defaults")

    if parent_name is None:
        return config

    if parent_name in seen:
        raise ValueError(f"Circular inheritance detected: {parent_name}")
    seen.add(parent_name)

    instruments_dir = join(dirname(__file__), "instruments")
    if parent_name in ["pyreduce", "defaults"]:
        parent_file = join(instruments_dir, "defaults", "settings.json")
    else:
        parent_file = join(instruments_dir, parent_name.upper(), "settings.json")

    if not exists(parent_file):
        raise FileNotFoundError(f"Inherited settings file not found: {parent_file}")

    with open(parent_file) as f:
        parent = json.load(f)

    parent = _resolve_inheritance(parent, seen)
    return update(parent, config, check=False)


def load_config(configuration, instrument, j=0):
    if configuration is None:
        logger.info(
            "No configuration specified, using default values for this instrument"
        )
        config = get_configuration_for_instrument(instrument, plot=False)
    elif isinstance(configuration, dict):
        if instrument in configuration.keys():
            config = configuration[str(instrument)]
        elif (
            "__instrument__" in configuration.keys()
            and configuration["__instrument__"] == str(instrument).upper()
        ):
            config = configuration
        else:
            raise KeyError("This configuration is for a different instrument")
    elif isinstance(configuration, list):
        config = configuration[j]
    elif isinstance(configuration, str):
        config = configuration

    if isinstance(config, str):
        logger.info("Loading configuration from %s", config)
        try:
            with open(config) as f:
                config = json.load(f)
        except FileNotFoundError:
            # Try to find settings file by instrument name
            # e.g. "settings_UVES.json" -> instruments/UVES/settings.json
            base_dir = dirname(__file__)
            if config.startswith("settings_") and config.endswith(".json"):
                inst_name = config[9:-5]  # Extract instrument name
                fname = join(
                    base_dir, "instruments", inst_name.upper(), "settings.json"
                )
            else:
                fname = join(base_dir, "instruments", "defaults", config)
            with open(fname) as f:
                config = json.load(f)

    # Resolve inheritance chain (defaults to inheriting from pyreduce)
    settings = _resolve_inheritance(config)

    # If it doesn't raise an Exception everything is as expected
    validate_config(settings)
    logger.debug("Configuration succesfully validated")

    return settings


def update(dict1, dict2, check=True, name="dict1"):
    """
    Update entries in dict1 with entries of dict2 recursively,
    i.e. if the dict contains a dict value, values inside the dict will
    also be updated

    Parameters
    ----------
    dict1 : dict
        dict that will be updated
    dict2 : dict
        dict that contains the values to update
    check : bool
        If True, will check that the keys from dict2 exist in dict1 already.
        Except for those contained in field "instrument"

    Returns
    -------
    dict1 : dict
        the updated dict

    Raises
    ------
    KeyError
        If dict2 contains a key that is not in dict1
    """
    # Instrument is a 'special' section as it may include any number of values
    # In that case we don't want to raise an error for new keys
    exclude = ["instrument"]
    for key, value in dict2.items():
        if check and key not in dict1.keys():
            logger.warning(f"{key} is not contained in {name}")
        if isinstance(value, dict):
            if dict1.get(key) is None:
                dict1[key] = value
            else:
                dict1[key] = update(
                    dict1[key], value, check=key not in exclude, name=key
                )
        else:
            dict1[key] = value
    return dict1


def read_config(fname=None):
    """Read the configuration file from disk

    If no filename is given it will load the default configuration.
    The configuration file must be a json file.

    Parameters
    ----------
    fname : str, optional
        Filename of the configuration. By default the default settings.

    Returns
    -------
    config : dict
        The read configuration file
    """
    this_dir = dirname(__file__)
    if fname is None:
        fname = join(this_dir, "instruments", "defaults", "settings.json")
    elif not exists(fname):
        fname = join(this_dir, "instruments", "defaults", fname)

    with open(fname) as file:
        settings = json.load(file)
        return settings


def validate_config(config):
    """Test that the input configuration complies with the expected schema

    Since it requires features from jsonschema 3+, it will only run if that is installed.
    Otherwise show a warning but continue. This is incase some other module needs an earlier,
    jsonschema (looking at you jwst).

    If the function runs through without raising an exception, the check was succesful or skipped.

    Parameters
    ----------
    config : dict
        Configurations to check

    Raises
    ------
    ValueError
        If there is a problem with the configuration.
        Usually that means a setting has an unallowed value.
    """
    if not hasJsonSchema:  # pragma: no cover
        # Can't check with old version
        return
    this_dir = dirname(__file__)
    fname = join(this_dir, "instruments", "defaults", "schema.json")

    with open(fname) as f:
        schema = json.load(f)
    try:
        jsonschema.validate(schema=schema, instance=config)
    except jsonschema.ValidationError as ve:
        logger.error("Configuration failed validation check.\n%s", ve.message)
        raise ValueError(ve.message) from ve
