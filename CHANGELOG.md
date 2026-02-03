# Changelog


## [0.8a2] - 2026-02-03

### Added
- `Trace` dataclass (trace_model.py) for unified trace data model
- `Spectrum`/`Spectra` classes (spectra.py) replacing legacy Echelle format
- `Trace.wlen(x)` method to evaluate wavelength polynomial
- Per-group wavelength calibration support
- LFC wavecal support for ANDES_RIZ
- docs/output_formats.md documenting new file formats

### Changed
- All pipeline steps use `list[Trace]` interface instead of array-based
- `extract()` takes `list[Trace]`, returns `list[Spectrum]`
- Renamed `nord`/`iord` to `ntrace`/`idx` for clarity
- Renamed `Trace.fiber` to `Trace.group`, added `fiber_idx` for per-fiber wavecal
- File format: traces.npz -> traces.fits (unified FITS format)
- Wavelength calibration stored in traces.fits instead of separate wavecal.npz
- Spectra use NaN masking instead of COLUMNS+MASK redundancy

### Removed
- `curvature_model.py` (curvature now stored in Trace)
- `echelle.py` functionality (deprecated, use spectra.py)
- Dead code from util.py

## [0.8a1] - 2026-02-03

### Added
- Charslit extraction backend with degree 1-5 curvature support
- ANDES_RIZ instrument configuration and settings
- `trace_by` config for separate tracing of illumination groups
- HDF-to-wavecal tool and reference files for ANDES_RIZ
- ANDES spectra plotting tool

### Changed
- Curvature: use literal pixel extraction_height, not fractional
- Simplify extraction_height to single full-height value
- Move fiber selection default from hardcoded to config
- Update METIS_IFU for dynamic wavelength-based channels
- Use git URL for charslit optional dependency

### Removed
- PEP 723 inline metadata from examples (caused confusion with `uv run`)

### Fixed
- Fix curve_height to use new full-height format in plot_comparison
- Fix extraction_height to give exactly N rows
- Fix CRIRES_PLUS docstring (was incorrectly HARPS)

## [0.7] - 2026-01-25

First stable release of the 0.7 series!

### Changed
- Rename `OrderTracing` class to `Trace`, save file from `.orders.npz` to `.traces.npz`
- Store per-trace extraction heights in traces.npz
- Compute extraction heights for fiber bundles automatically
- CLI: consistent `-t/--target` option across all commands
- Allow explicit paths in settings `__inherits__` directive

### Fixed
- Limit trace overlay to column_range in science step plot

### Documentation
- Add WhatsNew.md summarizing 0.7 series improvements
- Update docs for traces.npz rename and mask convention

## [0.7b4] - 2026-01-22

### Added
- Per-channel settings support via `settings_{channel}.json` files
- Fiber grouping configuration for multi-fiber instruments (bundle_centers, fiber_groups)
- `preset_slitfunc` parameter for single-pass extraction using pre-computed slit function
- `noise_relative` parameter for trace threshold scaling
- `extraction_reject` threshold parameter for outlier rejection
- ANDES_YJH instrument with multi-channel support
- MOSAIC VIS quadrant support (VIS1-VIS4 channels) and NIR settings
- HARPN fiber B wavecal file
- Extraction residual panel in ProgressPlot
- `tools/plot_swath_debug.py` for analyzing extraction debug data

### Changed
- **Mask convention**: Switch to numpy convention (True/1=bad) on Python side; C code uses inverted (1=good)
- Extraction convergence now based on spectrum change with global RMS for rejection
- Improved outlier rejection using 6*sigma instead of MAD in 2D extraction
- Norm_flat step saves slit function with metadata for reuse
- Refactor Pipeline trace API: add `trace_raw()` and `organize()` methods
- Remove `extraction_cutoff` parameter (dead code)

### Fixed
- `smooth_spectrum` normalization (port from cr2res)
- Masked array handling in swath debug plotting
- Minimum 4 iterations enforced in curved extraction

## [0.7b3] - 2026-01-11

### Added
- MOSAIC instrument support with fiber group detection and curvature step
- Extraction animation controls: pause/step buttons and speed control (`PYREDUCE_PLOT_ANIMATION_SPEED`)
- `--plot-dir` and `--plot-show` CLI options for flexible plot output

### Changed
- Rename `orders` to `traces` in rectify and slit_curve modules for consistency
- Downgrade extraction max-iterations message from ERROR to WARNING
- Only warn about missing files for steps that are actually requested

### Fixed
- Handle channel mismatch gracefully in CLI
- Curvature plotting index error when peaks need int casting

## [0.7b2] - 2026-01-09

### Changed
- **Reorganize instrument files**: All instrument-related files now in per-instrument directories
  - `instruments/{name}.py` → `instruments/{NAME}/__init__.py`
  - `instruments/{name}.yaml` → `instruments/{NAME}/config.yaml`
  - `settings/settings_{NAME}.json` → `instruments/{NAME}/settings.json`
  - `wavecal/{name}_*.npz` → `instruments/{NAME}/wavecal_*.npz`
  - `masks/mask_{name}_*.fits.gz` → `instruments/{NAME}/mask_*.fits.gz`
- Base settings and schema moved to `instruments/defaults/`
- Wavelength atlas files moved to `instruments/defaults/atlas/`

### Removed
- Orphan mask files for undefined instruments (elodie, sarg, hds, etc.)

## [0.7b1] - 2026-01-06

### Changed
- **Curvature algorithm rewrite**: Replace 2D model fitting with row-tracking method for better robustness
- Rename `make_shear.py` to `slit_curve.py`
- Rename curvature coefficients: `tilt`/`shear` → `p1`/`p2` throughout codebase
- Rename `arc_extraction` to `simple_extraction`
- Split curvature `extraction_height` into separate `extraction_height` and `curve_height` parameters
- Save file renamed from `.shear.npz` to `.curve.npz`

### Added
- `discover_channels()` for automatic channel detection from data files
- CLI: `--target` is now optional; loops over all targets if not specified
- CLI: Uses `$REDUCE_DATA` for base_dir, reads default input_dir from config
- Comprehensive CLI test coverage

### Fixed
- Curvature step index error when traces are removed during processing
- Validate base_dir and input_dir exist with clear error messages
- File sorting to correctly loop over all nights when not specified
- CLI dynamic dependency loading for step commands

## [0.7a7] - 2026-01-04

### Added
- `--settings` option for CLI to override reduction parameters from JSON file

### Changed
- Rename `extraction_width` to `extraction_height` in settings (clarifies coordinate system)
- Rename `orders` to `traces` in extract.py internal API

## [0.7a6] - 2026-01-03

### Added
- `--file` option for CLI step commands to bypass file discovery
- NEID instrument with multi-amplifier support

### Changed
- Rename `orders` step to `trace` throughout codebase (CLI, API, configs)

### Fixed
- test_normflat to use column_range returned by extract

## [0.7a5] - 2025-12-30

### Changed
- Rename `arm` to `channel` throughout codebase (API, CLI, configs)
- Remove decker from CRIRES+ channel format (`J1228_det1` instead of `J1228_Open_det1`)
- Example scripts now use `$REDUCE_DATA` env var instead of hardcoded paths

## [0.7a4] - 2025-12-23

### Added
- `reduce examples --run` flag to download and execute examples directly
- PEP 723 inline metadata in examples for `uv run` compatibility

## [0.7a3] - 2025-12-23

### Added
- `reduce examples` command to list/download examples from GitHub matching installed version

### Changed
- CLI startup 12x faster via lazy imports (1.2s -> 0.1s)

## [0.7a2] - 2025-12-23

### Added
- Manual API calls documentation (manual_calls.md)

### Fixed
- ReadTheDocs build failing due to missing myst_parser
- Trace module reference in documentation

### Changed
- Docs Makefile now uses uv
- pyproject.toml cleanup

## [0.7a1] - 2025-12-22

### Added
- New Pipeline API with `Pipeline.from_instrument()` for simplified usage
- Click-based CLI replacing argparse (`uv run reduce run UVES HD132205`)
- Pydantic models for instrument configuration validation
- YAML instrument configs replacing JSON
- IPython startup script for interactive development
- `plot_dir` option to save plots as PNG files
- Fiber bundle tracing support for multi-fiber instruments
- `filter_x` and `filter_type` options for order tracing

### Changed
- Rename `mode` to `channel` terminology throughout
- Output extension changed from `.ech` to `.fits`
- Documentation converted from RST to Markdown
- Trace detection parameters renamed (`opower` -> `degree`, `filter_size` -> `filter_y`)

### Fixed
- Plotting issues with non-finite values
- Use `interpolate_replace_nans` for masked pixels

## [0.6.0] - 2025-12-22

### Added
- JSON schema validation test for instrument configurations
- GitHub workflow creates GitHub Release on tag push

### Fixed
- Fix test_wavecal to match WavelengthCalibration.execute() signature
- Fix spec fixture to match ScienceExtraction.run() signature
- Fix instrument schema: replace invalid 'value' keyword with standard JSON Schema 'type'

## [0.6.0b5] - 2025-10-04

### Fixed
- Include *.pyd files for Windows builds
- Remove redundant build test from workflow

## [0.6.0b4] - 2025-10-03

### Changed
- workflow builds multi-arch wheels
- minor fixes to make build pass

## [0.6.0b3] - 2025-10-02

### Added
- Test organization with pytest markers (unit, instrument, slow, downloads)
- ANDES instrument configuration and settings

### Changed
- CI now runs fast tests (~7s) on every push to master
- Test suite optimized with slow marker for tests >3s

### Fixed
- ANDES instrument configuration errors (missing decker fields, regex pattern)
- NIRSPEC FITS header errors with illegal keywords containing dots
- Pre-commit hooks now documented and working correctly

## [0.6.0b2] - 2025-10-02

### Changed
- lazy imports
- cosmetic fixes like version string

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
