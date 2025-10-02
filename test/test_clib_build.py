import pytest

from pyreduce.clib import build_extract

pytestmark = pytest.mark.unit


def test_build_extract():
    build_extract.build()
