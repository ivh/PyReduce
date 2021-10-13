# -*- coding: utf-8 -*-
import pytest

from pyreduce import configuration as conf


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
    config["orders"]["degree"] = -1

    with pytest.raises(ValueError):
        conf.validate_config(config)
