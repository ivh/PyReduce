import json
import os

import pytest

from pyreduce import configuration as conf

pytestmark = pytest.mark.unit


def test_configuration():
    config = conf.load_config(None, "UVES", 0)
    assert isinstance(config, dict)

    config = conf.load_config(config, "UVES", 0)
    assert isinstance(config, dict)

    config = conf.load_config("settings_UVES.json", "UVES", 0)
    assert isinstance(config, dict)

    config = conf.load_config({"UVES": "settings_UVES.json"}, "UVES", 0)
    assert isinstance(config, dict)

    config = conf.load_config(["settings_UVES.json"], "UVES", 0)
    assert isinstance(config, dict)

    with pytest.raises(KeyError):
        config = conf.load_config({"UVES": "settings_UVES.json"}, "HARPS", 0)

    with pytest.raises(IndexError):
        config = conf.load_config(["settings_UVES.json"], "UVES", 1)


def test_update():
    dict1 = {"bla": 0, "blub": {"foo": 0, "bar": 0}}
    dict2 = {"bla": 1, "blub": {"bar": 1}}
    res = conf.update(dict1, dict2)

    assert isinstance(res, dict)
    assert "bla" in res.keys()
    assert "blub" in res.keys()
    assert isinstance(res["blub"], dict)
    assert "foo" in res["blub"].keys()
    assert "bar" in res["blub"].keys()

    assert res["bla"] == 1
    assert res["blub"]["foo"] == 0
    assert res["blub"]["bar"] == 1

    # This only shows a warning now
    # with pytest.raises(KeyError):
    #     conf.update(dict1, {"foo": 1}, check=True)

    res = conf.update(dict1, {"foo": "bar"}, check=False)
    assert res["foo"] == "bar"


def test_read_config():
    # Reads the default values
    res = conf.read_config()

    assert isinstance(res, dict)

    with pytest.raises(FileNotFoundError):
        conf.read_config(fname="blablub.json")


def test_validation():
    config = conf.get_configuration_for_instrument("UVES")
    config["trace"]["degree"] = -1

    with pytest.raises(ValueError):
        conf.validate_config(config)


def test_per_channel_settings(tmp_path):
    """Test that channel-specific settings files are loaded and inherit correctly."""
    inst_dir = os.path.join(os.path.dirname(conf.__file__), "instruments", "UVES")
    channel_file = os.path.join(inst_dir, "settings_TESTCHAN.json")

    try:
        # Create channel-specific settings that inherits from UVES
        channel_settings = {
            "__instrument__": "UVES",
            "__inherits__": "UVES",
            "science": {"extraction_height": 99},
        }
        with open(channel_file, "w") as f:
            json.dump(channel_settings, f)

        # Test: channel file is loaded when channel specified
        config = conf.get_configuration_for_instrument("UVES", channel="TESTCHAN")
        assert config["science"]["extraction_height"] == 99

        # Test: other settings inherited from UVES
        base_config = conf.get_configuration_for_instrument("UVES")
        assert config["trace"]["degree"] == base_config["trace"]["degree"]

        # Test: falls back to base settings when channel file doesn't exist
        config_fallback = conf.get_configuration_for_instrument(
            "UVES", channel="NONEXISTENT"
        )
        assert config_fallback["science"]["extraction_height"] != 99

        # Test: load_config passes channel through
        config_via_load = conf.load_config(None, "UVES", channel="TESTCHAN")
        assert config_via_load["science"]["extraction_height"] == 99

    finally:
        if os.path.exists(channel_file):
            os.remove(channel_file)
