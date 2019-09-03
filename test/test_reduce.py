import pytest

from pyreduce import reduce

def test_main(files, instrument, target, night, mode, input_dir, output_dir):
    output = reduce.main(
        instrument,
        target,
        night,
        mode,
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
    output = reduce.main(
        instrument,
        target,
        night,
        mode,
        steps=(),
    )

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
