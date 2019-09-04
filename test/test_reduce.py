import pytest
import os
import sys

from pyreduce import reduce


def test_main(files, instrument, target, night, mode, input_dir, output_dir):
    output = reduce.main(
        instrument,
        target,
        night,
        {instrument: mode},
        base_dir="",
        input_dir=input_dir,
        output_dir=output_dir,
        steps=(),
    )

    assert isinstance(output, list)
    assert len(output) == 1
    assert "config" in output[0].keys()
    assert "files" in output[0].keys()

    # Test default options
    # Just just not find anything
    output = reduce.main(instrument, target, night, steps=())


def test_all(files, instrument, target, night, mode, input_dir, output_dir):
    output = reduce.main(
        instrument,
        target,
        night,
        mode,
        base_dir="",
        input_dir=input_dir,
        output_dir=output_dir,
        order_range=(0, 1),
        steps="all",
    )


def test_load_all(instrument, target, night, mode, input_dir, output_dir):
    # Delete existing intermediate files
    files = [f for f in os.listdir(output_dir) if not f.startswith("test_")]
    for f in files:
        try:
            os.remove(f)
        except:
            pass

    output = reduce.main(
        instrument,
        target,
        night,
        mode,
        base_dir="",
        input_dir=input_dir,
        output_dir=output_dir,
        order_range=(0, 1),
        steps=["finalize"],
    )
