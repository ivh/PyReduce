# -*- coding: utf-8 -*-
import os

import numpy as np
import pytest
import scipy.constants

from pyreduce import echelle


@pytest.fixture
def fname():
    name = "test_ech.ech"
    yield name

    if os.path.exists(name):
        os.remove(name)


@pytest.fixture
def clight():
    return scipy.constants.speed_of_light * 1e-3


def test_read_write_echelle(fname):
    data = np.linspace(1, 200, num=100, dtype=float)
    header = {"BLA": "Blub"}

    echelle.save(fname, header, test=data)
    ech = echelle.Echelle.read(fname)
    ech.save(fname)

    assert isinstance(ech, echelle.Echelle)
    assert "test" in ech
    assert np.allclose(ech["test"], data)

    assert ech.header["BLA"] == "Blub"


def test_expand_2d():
    # 2d make wave format
    wave = [0, 100, 2, 1, 0, 0, 0, 4, 2, 2, 1000, 100, 0, 100, 0, 0, 0, 0, 0]
    wave = np.array(wave)

    new = echelle.expand_polynomial(2, wave)

    assert isinstance(new, np.ndarray)
    assert new.ndim == 2
    assert new.shape[0] == 2
    assert new.shape[1] == 100

    cmp = np.linspace(1001, 1011, 100, endpoint=False)
    assert np.allclose(new[0], cmp)

    cmp = np.linspace(501, 506, 100, endpoint=False)
    assert np.allclose(new[1], cmp)
    wave = [
        0,
        100,
        2,
        1,
        0,
        0,
        0,
        6,
        3,
        3,
        1000,
        100,
        0,
        0,
        100,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    wave = np.array(wave)
    new = echelle.expand_polynomial(2, wave)

    assert isinstance(new, np.ndarray)
    assert new.ndim == 2
    assert new.shape[0] == 2
    assert new.shape[1] == 100

    cmp = np.linspace(1001, 1011, 100, endpoint=False)
    assert np.allclose(new[0], cmp)

    cmp = np.linspace(501, 506, 100, endpoint=False)
    assert np.allclose(new[1], cmp)


def test_expand_1d():
    wave = [[0.1, 1001], [0.05, 501]]
    wave = np.array(wave)

    new = echelle.expand_polynomial(100, wave)

    assert isinstance(new, np.ndarray)
    assert new.ndim == 2
    assert new.shape[0] == 2
    assert new.shape[1] == 100

    cmp = np.linspace(1001, 1011, 100, endpoint=False)
    assert np.allclose(new[0], cmp)

    cmp = np.linspace(501, 506, 100, endpoint=False)
    assert np.allclose(new[1], cmp)


def test_wavelength_correction(fname, clight):
    bc = 100
    rv = 200
    header = {"barycorr": bc, "radvel": rv}
    wave = np.linspace(5000, 6000, 100)
    wave = np.atleast_2d(wave)

    echelle.save(fname, header, wave=wave)

    # Read raw frame
    ech = echelle.Echelle.read(fname, raw=True)
    assert np.allclose(ech["wave"], wave)
    assert ech.header["barycorr"] == bc
    assert ech.header["radvel"] == rv

    # Apply only barycentric correction
    ech = echelle.Echelle.read(
        fname, barycentric_correction=True, radial_velociy_correction=False
    )
    assert np.allclose(ech["wave"], wave * (1 - bc / clight))
    assert ech.header["barycorr"] == 0
    assert ech.header["radvel"] == rv

    # Apply only radial velocity correction
    ech = echelle.Echelle.read(
        fname, barycentric_correction=False, radial_velociy_correction=True
    )
    assert np.allclose(ech["wave"], wave * (1 + rv / clight))
    assert ech.header["barycorr"] == bc
    assert ech.header["radvel"] == 0

    # Apply both corrections
    ech = echelle.Echelle.read(
        fname, barycentric_correction=True, radial_velociy_correction=True
    )
    assert np.allclose(ech["wave"], wave * (1 + (rv - bc) / clight))
    assert ech.header["barycorr"] == 0
    assert ech.header["radvel"] == 0


def test_column_range_mask(fname):
    spec = np.linspace(1, 1000, 100)
    spec = np.stack([spec, spec])
    sig = np.copy(spec)
    cont = np.copy(spec)
    wave = np.copy(spec)

    columns = np.array([[10, 99], [0, 50]])

    echelle.save(fname, {}, spec=spec, sig=sig, cont=cont, wave=wave, columns=columns)

    ech = echelle.read(fname)

    assert "mask" in ech
    assert isinstance(ech["spec"], np.ma.masked_array)

    for iord in range(2):
        assert all(ech["mask"][iord, : columns[iord, 0]])
        assert all(ech["mask"][iord, columns[iord, 1] :])
        assert not any(ech["mask"][iord, columns[iord, 0] : columns[iord, 0]])


def test_continuum_normalization(fname):
    spec = np.full((2, 100), 10, dtype=float)
    sig = np.full((2, 100), 1, dtype=float)
    cont = np.full((2, 100), 2, dtype=float)

    echelle.save(fname, {}, spec=spec, sig=sig, cont=cont)
    ech = echelle.read(fname)

    assert np.allclose(ech["spec"], spec / cont)
    assert np.allclose(ech["sig"], sig / cont)
    assert np.allclose(ech["cont"], cont)
