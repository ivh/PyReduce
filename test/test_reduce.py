import pytest

from pyreduce import reduce


@pytest.mark.instrument
@pytest.mark.downloads
@pytest.mark.slow
def test_main(instrument, target, night, arm, input_dir, output_dir):
    output = reduce.main(
        instrument,
        target,
        night,
        arm,
        base_dir="",
        input_dir=input_dir,
        output_dir=output_dir,
        steps=(),
    )

    # reduce.main() returns a list of Pipeline.run() results (one per arm/night combo)
    assert isinstance(output, list)
    assert len(output) >= 1
    # With steps=(), each result is an empty dict (no steps executed)
    assert isinstance(output[0], dict)

    # Test default options - should not find anything with default paths
    output = reduce.main(instrument, target, night, steps=())


@pytest.mark.skip
@pytest.mark.instrument
@pytest.mark.downloads
@pytest.mark.slow
def test_run_all(instrument, target, night, arm, input_dir, output_dir, order_range):
    reduce.main(
        instrument,
        target,
        night,
        arm,
        base_dir="",
        input_dir=input_dir,
        output_dir=output_dir,
        order_range=order_range,
        steps="all",
    )


@pytest.mark.skip
@pytest.mark.instrument
@pytest.mark.downloads
@pytest.mark.slow
def test_load_all(instrument, target, night, arm, input_dir, output_dir, order_range):
    reduce.main(
        instrument,
        target,
        night,
        arm,
        base_dir="",
        input_dir=input_dir,
        output_dir=output_dir,
        order_range=order_range,
        steps=["finalize"],
    )


@pytest.mark.instrument
@pytest.mark.downloads
@pytest.mark.slow
def test_step_abstract(step_args):
    step = reduce.Step(*step_args, **{"plot": False})

    assert isinstance(step.dependsOn, list)
    assert isinstance(step.loadDependsOn, list)
    assert isinstance(step.prefix, str)
    assert isinstance(step.output_dir, str)

    with pytest.raises(NotImplementedError):
        step.load()

    with pytest.raises(NotImplementedError):
        step.run([])

    with pytest.raises(NotImplementedError):
        step.save()
