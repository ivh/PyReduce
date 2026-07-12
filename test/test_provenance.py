"""Tests for provenance stamping of output FITS files."""

import numpy as np
import pytest
from astropy.io import fits

from pyreduce import __version__, provenance


class TestGitRevision:
    @pytest.mark.unit
    def test_returns_str_or_none(self):
        provenance.get_git_revision.cache_clear()
        rev = provenance.get_git_revision()
        assert rev is None or (isinstance(rev, str) and len(rev) > 0)

    @pytest.mark.unit
    def test_git_failure_returns_none(self, monkeypatch):
        provenance.get_git_revision.cache_clear()

        def boom(*args, **kwargs):
            raise OSError("no git")

        monkeypatch.setattr(provenance.subprocess, "run", boom)
        assert provenance.get_git_revision() is None
        provenance.get_git_revision.cache_clear()


class TestAddProvenance:
    @pytest.mark.unit
    def test_stamps_version_and_githash(self, monkeypatch):
        monkeypatch.setattr(
            provenance, "get_git_revision", lambda: "v0.9b1-6-gabc123-dirty"
        )
        header = provenance.add_provenance()
        assert header["PR_version"] == __version__
        assert header["PR_githash"] == "v0.9b1-6-gabc123-dirty"

    @pytest.mark.unit
    def test_no_githash_outside_checkout(self, monkeypatch):
        monkeypatch.setattr(provenance, "get_git_revision", lambda: None)
        header = provenance.add_provenance(fits.Header())
        assert header["PR_version"] == __version__
        assert "PR_githash" not in header


class TestStampedFiles:
    """The central FITS writers stamp provenance into their headers."""

    @pytest.mark.unit
    def test_save_traces_stamped(self, tmp_path):
        from pyreduce.trace_model import Trace, save_traces

        tr = Trace(m=90, pos=np.array([0.0, 0.0, 100.0]), column_range=(10, 990))
        fname = tmp_path / "t.traces.fits"
        save_traces(fname, [tr], steps=["trace"])

        head = fits.getheader(fname)
        assert head["PR_version"] == __version__
        rev = provenance.get_git_revision()
        if rev is not None:
            assert head["PR_githash"] == rev

    @pytest.mark.unit
    def test_spectra_save_stamped(self, tmp_path):
        from pyreduce.spectra import Spectra, Spectrum

        sp = Spectrum(
            m=90,
            spec=np.ones(64, dtype=np.float32),
            sig=np.ones(64, dtype=np.float32),
        )
        fname = tmp_path / "s.fits"
        Spectra(header=fits.Header(), data=[sp]).save(str(fname), steps=["science"])

        head = fits.getheader(str(fname))
        assert head["PR_version"] == __version__
        rev = provenance.get_git_revision()
        if rev is not None:
            assert head["PR_githash"] == rev
