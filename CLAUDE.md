# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PyReduce is a Python port of the REDUCE echelle spectrograph data reduction pipeline. It processes raw astronomical observations from instruments like HARPS, UVES, XSHOOTER, CRIRES+, JWST/NIRISS and others into calibrated 1D spectra.

The pipeline performs sequential steps: bias correction, flat fielding, order tracing, wavelength calibration, spectrum extraction, and continuum normalization. Each step can be run independently or as part of a complete reduction.

## Development Setup

This project uses **uv** for fast, modern Python package management. All Python commands should use `uv run` instead of direct Python invocation.

```bash
# Install dependencies (use uv, not pip)
uv sync

# Install with development dependencies
uv sync --all-extras
```

## Common Commands

### Using uv
**IMPORTANT: Always use `uv run` to execute Python commands.** This ensures the correct environment and dependencies.

```bash
# Run tests
uv run pytest

# Run specific test file
uv run pytest test/test_extract.py

# Run tests with coverage
uv run pytest --cov=pyreduce --cov-report=html

# Run tests by marker (see Test Organization section below)
uv run pytest -m unit                    # Fast unit tests only (~40 tests, <10s)
uv run pytest -m instrument              # Integration tests with datasets (~70 tests)
uv run pytest -m "instrument and not slow"  # Skip slow integration tests
uv run pytest -m "not downloads"         # Offline mode (no dataset downloads)

# Run example script
uv run python examples/uves_example.py

# Run PyReduce as module
uv run python -m pyreduce
```

### Building and Publishing
```bash
# Build source distribution and wheel (uses Hatchling)
uv build

# The build will:
# 1. Compile two CFFI C extensions via hatch_build.py:
#    - _slitfunc_bd (vertical extraction)
#    - _slitfunc_2d (curved extraction)
# 2. Create .tar.gz and .whl in dist/

# Note: setuptools is required in build-system even though we use Hatchling
# because CFFI requires it on Python 3.12+

# Publish to PyPI (manual with twine, API keys in ~/.pypirc)
uv run twine upload dist/*
```

### Code Quality
```bash
# Format and lint with Ruff (replaces black, isort, flake8)
uv run ruff format .
uv run ruff check .
uv run ruff check --fix .

# Run pre-commit hooks (runs automatically on commit, or manually)
uv run pre-commit run --all-files

# Pre-commit hooks include:
# - File checks (trailing whitespace, end-of-file, yaml/json/toml validation)
# - Ruff linter and formatter
# All legacy tools (black, isort, flake8) have been replaced with Ruff
```

## Build System

### Modern Tooling Stack (2025)
- **Package manager**: uv (fast, modern alternative to pip/poetry)
- **Build backend**: Hatchling (PEP 517 compliant, replaces setuptools)
- **Linter/formatter**: Ruff (replaces black, isort, flake8, pyupgrade)
- **Python version**: 3.13+ (specified in pyproject.toml)

### Build Configuration
```toml
[build-system]
requires = ["hatchling>=1.25.0", "cffi>=1.17.1", "setuptools"]
build-backend = "hatchling.build"
```

**Why setuptools is still required**: CFFI internally uses setuptools for building C extensions on Python 3.12+. This is a CFFI requirement, not a PyReduce build system requirement.

### CFFI Extension Build Process
The `hatch_build.py` file implements a Hatchling build hook that:
1. Reads C source files from `pyreduce/clib/`
2. Uses CFFI's `FFI()` to generate wrapper code
3. Compiles two extensions:
   - `_slitfunc_bd.so` - Vertical slit function extraction
   - `_slitfunc_2d.so` - Curved slit function extraction (2D)
4. Places compiled `.so` files in `pyreduce/clib/`

The hook runs automatically during `uv build` or `uv sync` when installing in editable mode.

## Architecture

### Pipeline Flow

The main entry point is `pyreduce.reduce.main()`, which orchestrates these steps:

1. **bias** - Combines bias frames and fits polynomial vs exposure time
2. **flat** - Combines flat field frames
3. **orders** - Traces echelle orders on the detector using `trace_orders.py`
4. **norm_flat** - Normalizes flat field, creates blaze function
5. **curvature** - Determines slit curvature using `make_shear.py`
6. **wavecal** - Wavelength calibration from ThAr/etalon spectra (`wavelength_calibration.py`)
7. **science** - Extracts 1D spectrum from 2D image using `extract.py`
8. **continuum** - Continuum normalization via `continuum_normalization.py`
9. **finalize** - Writes final .ech FITS files

### Key Modules

**pyreduce/reduce.py** - Main reduction orchestrator with step classes (Bias, Flat, OrderTracing, etc.)

**pyreduce/extract.py** - Optimal spectrum extraction using the slit function decomposition method. Calls C code via `cwrappers.py`.

**pyreduce/wavelength_calibration.py** - Three-stage wavelength solution:
- Initialize: Creates initial wavelength guess from atlas lines
- Master: Refines solution across all orders
- Finalize: Applies final wavelength solution

**pyreduce/trace_orders.py** - Automatic order detection and tracing using polynomial fits

**pyreduce/instruments/** - Instrument-specific configuration. Each instrument has:
- `.py` file with class inheriting from `instruments.common.Instrument`
- `.json` file with instrument parameters (detector size, orientation, etc.)

**pyreduce/settings/** - JSON configuration files controlling reduction parameters for each step (polynomial degrees, iteration counts, plotting options)

**pyreduce/clib/** - C extensions for performance-critical extraction code:
- `slit_func_bd.c` / `slit_func_bd.h` - Vertical slit function decomposition
- `slit_func_2d_xi_zeta_bd.c` / `slit_func_2d_xi_zeta_bd.h` - Curved 2D extraction
- Compiled `.so` files for each Python version

**pyreduce/wavecal/** - Wavelength calibration reference data:
- `atlas/` - Spectral line atlases (ThAr, etalon)
- Pre-computed wavelength solutions as `.npz` files

### Configuration System

Configuration uses a two-tier approach:

1. **Instrument files** (`pyreduce/instruments/*.json`) - Hardware parameters
2. **Settings files** (`pyreduce/settings/settings_*.json`) - Reduction parameters

Both are validated against JSON schemas (`instrument_schema.json`, `settings_schema.json`).

Settings can be loaded via `pyreduce.configuration.get_configuration_for_instrument()`.

### Data Flow

Input: Raw FITS files organized as `base_dir/input_dir/{night}/`

Output:
- Intermediate products saved as `.npz` files in `output_dir/`
- Final 1D spectra as `.ech` files (FITS format with table extension)
- Header keywords from input preserved, PyReduce keywords prefixed with `e_`

### C Integration

Performance-critical extraction uses C code wrapped via CFFI:
- `pyreduce/cwrappers.py` provides Python interface to `slitfunc()` and `slitfunc_curved()`
- C source in `pyreduce/clib/*.c`
- Build happens automatically during `uv build` or `uv sync` via `hatch_build.py`
- CFFI requires setuptools on Python 3.12+ (not used for build system, only CFFI internals)

## Testing

### Test Organization

Tests are organized using pytest markers for efficient test selection:

**Markers:**
- `@pytest.mark.unit` - Fast unit tests with synthetic data (~40 tests, <10s)
- `@pytest.mark.instrument` - Integration tests using real instrument datasets (~70 tests)
- `@pytest.mark.slow` - Long-running tests (>5 seconds, typically wavecal/continuum)
- `@pytest.mark.downloads` - Tests that download sample datasets from the web

**Running Tests:**
```bash
# Run all tests
uv run pytest

# Fast unit tests only (development workflow)
uv run pytest -m unit

# Integration tests with instruments
uv run pytest -m instrument

# Skip slow tests
uv run pytest -m "not slow"

# Offline mode (no downloads, unit tests only)
uv run pytest -m "not downloads"

# Debug specific instrument failures
uv run pytest -m instrument -k NIRSPEC

# Combine markers
uv run pytest -m "instrument and not slow"
```

**Test Structure:**
- Tests use pytest with fixtures defined in `test/conftest.py`
- `instrument` fixture provides dataset instances for UVES, XSHOOTER, NIRSPEC, JWST_NIRISS
- Each reduction step has a corresponding class fixture (e.g., `bias_step`, `flat_step`)
- Integration tests download small sample datasets automatically via `pyreduce.datasets`

**Unit Tests** (no instrument fixtures):
- `test_extract.py` - Extraction algorithm with synthetic data
- `test_cwrappers.py` - C wrapper tests
- `test_clipnflip.py`, `test_combine.py`, `test_echelle.py` - Utility function tests
- `test_configuration.py` - Config parsing
- `test_instruments.py` - Instrument loading from JSON
- Selected tests in `test_bias.py` - Error handling tests

**Integration Tests** (use instrument datasets):
- `test_flat.py`, `test_normflat.py` - Flat field processing
- `test_orders.py`, `test_mask.py` - Order tracing and masking
- `test_scatter.py`, `test_shear.py` - Background and curvature
- `test_science.py` - Science spectrum extraction
- `test_wavecal.py` - Wavelength calibration (slow)
- `test_continuum.py` - Continuum normalization (slow)
- `test_reduce.py` - Full pipeline integration (slow)

## Instrument Support

To add a new instrument:
1. Create `pyreduce/instruments/instrument_name.py` inheriting from `Instrument`
2. Create `pyreduce/instruments/instrument_name.json` with detector parameters
3. Create `pyreduce/settings/settings_INSTRUMENT_NAME.json` for reduction defaults
4. Add example script to `examples/instrument_name_example.py`

See `examples/custom_instrument_example.py` for template.

## GitHub and CI/CD

### GitHub Actions Workflow
The repository uses GitHub Actions (`.github/workflows/python-publish.yml`) for CI/CD:

**On pull requests to master:**
- Runs pre-commit hooks (Ruff linting/formatting, file checks)
- Runs pytest test suite with coverage reporting
- Builds distribution packages to verify build works

**On regular pushes to master:**
- Tests do NOT run automatically (saves CI time since pre-commit hooks run locally)
- Manually trigger tests when needed: `gh workflow run "PyReduce CI/CD"`
- Or trigger from GitHub UI: Actions tab → "PyReduce CI/CD" → "Run workflow"

### Release Process

To publish a new release to PyPI:

1. **Edit CHANGELOG.md** - Document changes in the new version
2. **Update version** - Edit version number in `pyproject.toml`
3. **Commit and tag**:
   ```bash
   git commit -am "Release vX.Y.Z"
   git tag vX.Y.Z
   git push
   git push --tags
   ```
4. **Clean and build**:
   ```bash
   rm -rf dist/ build/ *.egg-info
   uv build
   ```
5. **Upload to PyPI**:
   ```bash
   uvx twine upload dist/*
   ```
   (Requires API keys in `~/.pypirc`)
6. **GitHub Release (optional)**
   ```bash
   gh release create vX.Y.Z --notes-file CHANGELOG.md
   ```

CI does NOT auto-publish (manual control for releases)

### Using the `gh` CLI Tool

```bash
# View recent workflow runs
gh run list --limit 5

# View logs from a failed run
gh run view <run-id> --log-failed

# Watch a workflow run in real-time
gh run watch <run-id>

# View workflow configuration
gh workflow list
gh workflow view "PyReduce CI/CD"

# Work with issues and PRs
gh issue list
gh pr create
gh pr view <pr-number>
```

### Common CI/CD Tasks

**Check if CI is passing:**
```bash
gh run list --limit 1
```

**Debug failed CI run:**
```bash
gh run list --limit 5
gh run view <run-id> --log-failed
```

**Re-run failed jobs:**
```bash
gh run rerun <run-id>
```

## Important Notes

- Always use `uv run` for Python commands to ensure correct environment
- Pre-commit hooks enforce code quality (runs Ruff automatically)
- CI runs pre-commit hooks - make sure they pass locally before pushing
- Interactive plotting can be disabled in settings JSON files with `"plot": false`
- All reduction steps are resumable - intermediate products are cached
- The C extensions must compile successfully for extraction to work
- Output `.ech` files are standard FITS files despite the extension name
- `docs/_build/` is gitignored - built docs shouldn't be committed

## Ruff Configuration

Ruff errors are configured in `pyproject.toml` with intelligent ignores:
- E402: Module imports not at top (legitimate in `__init__.py`, `docs/conf.py`)
- E722: Bare except clauses (legacy code, requires careful refactoring)
- UP031: Printf-style formatting (low priority in `tools/` and `examples/`)
- F401: Unused imports in `__init__.py` (intentional re-exports)

All Ruff checks currently pass. When adding new code, run `uv run ruff check --fix .` to auto-fix issues.
