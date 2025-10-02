# Changelog

## [0.6.0b1] - 2025-10-02

### Changed
- Modern build system using uv package manager
- Minimum Python version now 3.11, default 3.13
- Migrated build system from setuptools to Hatchling (PEP 517)
- Replaced black, isort, flake8, and pyupgrade with Ruff
- Improved code formatting across entire codebase using Ruff
- Modernized GitHub Actions workflow (uv-based, Python 3.11-3.13 matrix)
- Updated documentation to reflect modern installation with uv
- Consolidated legacy build files (setup.py, requirements.txt) into pyproject.toml
- Removed automatic PyPI publishing from GitHub Actions (now manual)
- GitHub Actions tests now manual-only on master branch
- Modernized ReadTheDocs build configuration

### Fixed
- Close FITS file handles after reading to prevent resource leaks ([#28](https://github.com/ivh/PyReduce/pull/28))
- Store wavelength calibration in double precision instead of single ([#30](https://github.com/ivh/PyReduce/pull/30))
- Add fallback value for E_ORIENT header keyword ([#31](https://github.com/ivh/PyReduce/pull/31))
- Return final wavelength linelist with flags from calibration ([#33](https://github.com/ivh/PyReduce/pull/33))
- Fix MaskedArray filling in normflat routine
- Fix undefined TypeFilter import in NEID instrument
- Fix Sphinx documentation build warnings
- Fix pre-commit hook configuration issues

### Documentation
- Updated README badges for GitHub Actions and Python versions
- Updated installation instructions to use uv instead of pip
- Improved documentation build process
- CLAUDE.md / AGENTS.md for AI-assisted development guidance
