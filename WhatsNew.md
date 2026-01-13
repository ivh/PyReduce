## What's New in PyReduce 0.6 & 0.7

### Installation & Build
- **Modern tooling**: Now uses `uv` package manager. Install with `uv sync`, run with `uv run reduce ...`
- **Python 3.11+** required (3.13 default)
- Build system migrated from setuptools to Hatchling

### New CLI
The old argparse CLI is replaced with a cleaner Click-based interface:
```bash
# Old
python -m pyreduce.reduce UVES HD132205 ...

# New
uv run reduce run UVES HD132205 --steps bias,flat,trace,science
uv run reduce download UVES        # get sample data
uv run reduce examples --run       # run examples directly
```

### New Pipeline API
```python
from pyreduce.pipeline import Pipeline

result = Pipeline.from_instrument(
    instrument="UVES",
    target="HD132205",
    steps=("bias", "flat", "trace", "science"),
).run()
```
The old `pyreduce.reduce.main()` still works but shows a deprecation warning.

### Multi-Fiber Instrument Support
New `fibers` configuration in instrument YAML for IFU and multi-fiber spectrographs:
- **Named groups**: Define fiber ranges (e.g., science A/B, calibration) with merge strategies
- **Bundle patterns**: For repeating fiber bundles (e.g., 7 fibers per IFU target)
- **Per-order grouping**: For echelle instruments with fiber groups repeated across orders
- **Missing fiber handling**: `bundle_centers_file` assigns traces by proximity when fibers are broken
- Per-step trace selection (`science: [A, B]`, `wavecal: [cal]`, etc.)

See `docs/fiber_bundle_tracing.md` for details.

### Terminology Changes
- `mode` → `channel` (e.g., `mode="middle"` is now `channel="middle"`)
- `orders` step → `trace` step
- `extraction_width` → `extraction_height`
- Output files: `.ech` → `.fits`

### New Instruments
- **MOSAIC** (ELT) with multi-fiber bundle support
- **NEID** with multi-amplifier support
- **ANDES** configuration added

### Configuration
- Instrument configs now use YAML (validated by Pydantic)
- Files reorganized into `instruments/{NAME}/` directories
- Settings cascade: defaults → instrument → runtime overrides

### Plotting
- `PYREDUCE_PLOT_DIR` saves plots as PNG
- `PYREDUCE_PLOT_SHOW=defer` accumulates plots for batch viewing
- Extraction animation with pause/step controls

### Bug Fixes
- FITS file handles properly closed
- Wavelength calibration stored in double precision
- Various curvature and extraction fixes
