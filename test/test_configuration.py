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


def test_explicit_path_inheritance(tmp_path):
    """Test that explicit path inheritance works (e.g., 'MOSAIC/settings_VIS1.json')."""
    inst_dir = os.path.join(os.path.dirname(conf.__file__), "instruments", "UVES")
    parent_file = os.path.join(inst_dir, "settings_PARENT.json")
    child_file = os.path.join(inst_dir, "settings_CHILD.json")

    try:
        # Create parent settings
        parent_settings = {
            "__instrument__": "UVES",
            "__inherits__": "defaults/settings.json",
            "trace": {"degree": 7, "noise": 999},
        }
        with open(parent_file, "w") as f:
            json.dump(parent_settings, f)

        # Create child that inherits from parent using explicit path
        child_settings = {
            "__instrument__": "UVES",
            "__inherits__": "UVES/settings_PARENT.json",
            "trace": {"noise": 123},
        }
        with open(child_file, "w") as f:
            json.dump(child_settings, f)

        # Test: child inherits from parent
        config = conf.get_configuration_for_instrument("UVES", channel="CHILD")
        assert config["trace"]["degree"] == 7  # from parent
        assert config["trace"]["noise"] == 123  # overridden in child

        # Test: values from defaults are still inherited
        assert "min_cluster" in config["trace"]  # from defaults

    finally:
        for f in [parent_file, child_file]:
            if os.path.exists(f):
                os.remove(f)


def test_inheritance_file_not_found():
    """Test that missing inheritance file raises FileNotFoundError."""
    from pyreduce.configuration import _resolve_inheritance

    config = {"__inherits__": "NONEXISTENT/settings.json", "trace": {}}
    with pytest.raises(FileNotFoundError):
        _resolve_inheritance(config)


def test_legacy_inheritance_formats():
    """Test that legacy inheritance formats still work."""
    from pyreduce.configuration import _resolve_inheritance

    # "defaults" should work
    config1 = {"__inherits__": "defaults", "trace": {"degree": 5}}
    result1 = _resolve_inheritance(config1.copy())
    assert "trace" in result1
    assert result1["trace"]["degree"] == 5

    # "pyreduce" should work
    config2 = {"__inherits__": "pyreduce", "trace": {"degree": 6}}
    result2 = _resolve_inheritance(config2.copy())
    assert result2["trace"]["degree"] == 6

    # Bare instrument name should work (e.g., "UVES" -> "UVES/settings.json")
    config3 = {"__inherits__": "UVES", "science": {"extraction_height": 42}}
    result3 = _resolve_inheritance(config3.copy())
    assert result3["science"]["extraction_height"] == 42
