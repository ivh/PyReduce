import pytest

from pyreduce import reduce


def test_configuration():
    config = reduce.load_config(None, "UVES", 0)
    assert isinstance(config, dict)

    config = reduce.load_config("settings_UVES.json", "UVES", 0)
    assert isinstance(config, dict)

    config = reduce.load_config({"UVES": "settings_UVES.json"}, "UVES", 0)
    assert isinstance(config, dict)

    config = reduce.load_config(["settings_UVES.json"], "UVES", 0)
    assert isinstance(config, dict)

    with pytest.raises(KeyError):
        config = reduce.load_config({"UVES": "settings_UVES.json"}, "HARPS", 0)

    with pytest.raises(IndexError):
        config = reduce.load_config(["settings_UVES.json"], "UVES", 1)


def test_main():
    reduce.main(steps=())
