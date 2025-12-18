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

# Install pre-commit hooks (IMPORTANT: run this once after cloning)
uv run pre-commit install
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

# Run tests for specific instruments using CLI arguments
uv run pytest --instrument=UVES          # Test only UVES (with default target)
uv run pytest --instrument=XSHOOTER --target="UX-Ori"  # Custom target
uv run pytest test/test_flat.py --instrument=NIRSPEC   # Single test file, one instrument

# Run tests in parallel (unit tests only - integration tests share datasets)
uv run --with pytest-xdist pytest -n auto -m unit

# Run example script
uv run python examples/uves_example.py

# Run PyReduce as module
uv run python -m pyreduce
```

### Building Locally

```bash
# Build platform-specific wheel for local testing
uv build

# Forces source build on unsupported platforms (requires C compiler)
uv pip install --no-binary pyreduce-astro pyreduce-astro
```

**Note:** See "Release Process" section below for publishing to PyPI.

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
1. Marks the wheel as platform-specific (sets `pure_python=False` and `infer_tag=True`)
2. Reads C source files from `pyreduce/clib/`
3. Uses CFFI's `FFI()` to generate wrapper code
4. Compiles two extensions:
   - `_slitfunc_bd.so/.pyd` - Vertical slit function extraction
   - `_slitfunc_2d.so/.pyd` - Curved slit function extraction (2D)
5. Places compiled extensions in `pyreduce/clib/` (`.so` on Linux/macOS, `.pyd` on Windows)

The hook runs automatically during:
- `uv build` - Creates platform-specific wheel (e.g., `cp313-cp313-macosx_14_0_arm64.whl`)
- `uv sync` - When installing in editable mode
- `cibuildwheel` - In CI, builds wheels for all platforms

**Platform-specific wheels:** The build creates different wheel files for each OS/architecture combination. This ensures users get pre-compiled C extensions that work on their system without requiring a C compiler.

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
9. **finalize** - Writes final FITS files

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
- Compiled extensions (`.so` on Linux/macOS, `.pyd` on Windows) for each Python version

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
- Final 1D spectra as FITS files (with binary table extension)
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

# Test specific instrument (faster than running all 4 instruments)
uv run pytest --instrument=UVES                          # UVES with default target
uv run pytest --instrument=XSHOOTER --target="UX-Ori"   # Custom target
uv run pytest test/test_orders.py --instrument=NIRSPEC   # Single file, one instrument

# Debug specific instrument failures (keyword matching)
uv run pytest -m instrument -k NIRSPEC

# Combine markers
uv run pytest -m "instrument and not slow"

# Parallel test execution (faster, but only for unit tests)
uv run --with pytest-xdist pytest -n auto -m unit  # ~3-4x faster
# Note: Integration tests cannot run in parallel - they share dataset directories
```

**Test Structure:**
- Tests use pytest with fixtures defined in `test/conftest.py`
- `instrument` fixture provides dataset instances for UVES, XSHOOTER, NIRSPEC, JWST_NIRISS
- Each reduction step has a corresponding class fixture (e.g., `bias_step`, `flat_step`)
- Integration tests download small sample datasets automatically via `pyreduce.datasets`

**CLI Arguments:**
Tests accept optional command-line arguments to filter by instrument/target:
- `--instrument=NAME` - Run tests for single instrument (UVES, XSHOOTER, NIRSPEC, JWST_NIRISS)
- `--target=NAME` - Override default target for the instrument (optional)
- Without arguments: Tests run for all instruments (full parametrized matrix)
- Use cases:
  - Development: `uv run pytest --instrument=UVES` (faster iteration)
  - Debugging: `uv run pytest test/test_wavecal.py --instrument=NIRSPEC` (isolate failures)
  - CI: No arguments needed (runs full matrix automatically)

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

**On pull requests:**
- Runs pre-commit hooks (Ruff linting/formatting, file checks)
- Runs pytest test suite with coverage reporting (non-slow tests)
- Uploads coverage to Codecov

**On push to master:**
- Nothing (saves CI time - tests run on PRs)

**On tag push (v*):**
- Runs tests first (blocks builds if tests fail)
- Builds platform-specific wheels for all platforms using **cibuildwheel**
- Builds source distribution
- **Automatically uploads to PyPI** using trusted publishing
- **Creates GitHub Release** with CHANGELOG.md as release notes

### cibuildwheel Configuration

Builds wheels for **Python 3.11, 3.12, 3.13** on:
- **Linux**: manylinux x86_64
- **macOS**: Intel (x86_64) and Apple Silicon (arm64)
- **Windows**: x86_64

Total: ~12-15 wheels per release. Configuration in `pyproject.toml`:

```toml
[tool.cibuildwheel]
build = "cp311-* cp312-* cp313-*"
skip = "*-musllinux_* *-win32 *-manylinux_i686"
test-command = "pytest {project}/test -m unit"  # Tests installed wheel
test-requires = ["pytest"]
```

### Release Process

**Automated release workflow (recommended):**

1. **Update version and changelog:**
   ```bash
   # Edit version in pyproject.toml
   # Update CHANGELOG.md with changes
   git commit -am "Release vX.Y.Z"
   ```

2. **Create and push tag:**
   ```bash
   git tag vX.Y.Z
   git push origin master
   git push origin vX.Y.Z
   ```

3. **GitHub Actions automatically:**
   - Runs tests (blocks if tests fail)
   - Builds wheels for all platforms (takes ~30-60 minutes)
   - Uploads to PyPI via trusted publishing
   - Creates GitHub Release with CHANGELOG.md as notes

4. **Verify release:**
   ```bash
   # Monitor workflow progress
   gh run watch

   # Check PyPI upload
   open https://pypi.org/project/pyreduce-astro/

   # Check GitHub Release
   gh release view vX.Y.Z
   ```

**Manual release (if CI fails):**
```bash
# Download wheels from GitHub Actions
gh run download <run-id> -D dist/

# Upload to PyPI
uvx twine upload dist/*

# Create GitHub Release manually
gh release create vX.Y.Z --notes-file CHANGELOG.md
```

**Setup requirements:**
- **PyPI trusted publishing**: Configure `pypi` environment in GitHub repository settings
- **GitHub permissions**: Workflow has `contents: write` for creating releases
- **No API tokens needed** - uses GitHub OIDC for PyPI authentication

### Common `gh` CLI Commands

```bash
# Monitor workflow runs
gh run watch              # Watch current run in real-time
gh run list --limit 5     # List recent runs
gh run view <run-id> --log-failed  # Debug failed run
gh run rerun <run-id>     # Re-run failed jobs

# View releases
gh release list
gh release view vX.Y.Z

# Issues and PRs
gh issue list
gh pr create
gh pr view <pr-number>
```

## Important Notes

- Always use `uv run` for Python commands to ensure correct environment
- Pre-commit hooks enforce code quality (runs Ruff automatically)
- CI runs pre-commit hooks - make sure they pass locally before pushing
- Interactive plotting can be disabled in settings JSON files with `"plot": false`
- All reduction steps are resumable - intermediate products are cached
- The C extensions must compile successfully for extraction to work
- Output spectrum files are standard FITS files with binary table extensions
- `docs/_build/` is gitignored - built docs shouldn't be committed

## Ruff Configuration

Ruff errors are configured in `pyproject.toml` with intelligent ignores:
- E402: Module imports not at top (legitimate in `__init__.py`, `docs/conf.py`)
- E722: Bare except clauses (legacy code, requires careful refactoring)
- UP031: Printf-style formatting (low priority in `tools/` and `examples/`)
- F401: Unused imports in `__init__.py` (intentional re-exports)

All Ruff checks currently pass. When adding new code, run `uv run ruff check --fix .` to auto-fix issues.
