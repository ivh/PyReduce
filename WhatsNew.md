## What's New in PyReduce 0.7 ?

### Multi-Fiber Instrument Support

New `fibers` configuration in instrument YAML for IFU and multi-fiber spectrographs:

- **Named groups**: Define fiber ranges (e.g., science A/B, calibration) with merge strategies
- **Bundle patterns**: For repeating fiber bundles (e.g., 7 fibers per IFU target)
- **Per-order grouping**: For echelle instruments with fiber groups repeated across orders
- **Missing fiber handling**: `bundle_centers_file` assigns traces by proximity when fibers are broken
- Per-step trace selection (`science: [A, B]`, `wavecal: [cal]`, etc.)

See [docs/fiber_bundle_tracing.md](docs/fiber_bundle_tracing.md) for details.

### New Instruments

- **MOSAIC** (ELT) with multi-fiber bundle support, VIS quadrants (VIS1-VIS4)
- **ANDES_YJH** with multi-channel support
- **NEID** with multi-amplifier support

See [docs/instruments.md](docs/instruments.md) for the full list.

### Configuration Changes

- Instrument configs now use YAML (validated by Pydantic)
- Files reorganized into `instruments/{NAME}/` directories
- Settings cascade: defaults → instrument → runtime overrides
- **Per-channel settings**: `settings_{channel}.json` for channel-specific parameters

See [docs/configuration_file.md](docs/configuration_file.md) for details.

### Terminology Changes

- `mode` → `channel` (e.g., `mode="middle"` is now `channel="middle"`)
- `orders` step → `trace` step
- `OrderTracing` class → `Trace`, output `.orders.npz` → `.traces.npz`
- `extraction_width` → `extraction_height`
- Output files: `.ech` → `.fits`
- **Mask convention**: Now uses numpy standard (1=bad, 0=good)

### Plotting

- `PYREDUCE_PLOT_DIR` saves plots as PNG files
- `PYREDUCE_PLOT_SHOW=defer` accumulates plots for batch viewing
- Extraction animation with pause/step controls and residual panel
- Trace overlay on calibration plots

### New Pipeline API

```python
from pyreduce.pipeline import Pipeline

result = Pipeline.from_instrument(
    instrument="UVES",
    target="HD132205",
    steps=("bias", "flat", "trace", "science"),
).run()
```

The old `pyreduce.reduce.main()` still works but shows a deprecation warning. See [docs/examples.md](docs/examples.md) for more.

### New CLI

The old argparse CLI is replaced with a cleaner Click-based interface. See [docs/cli.md](docs/cli.md) for the full reference.

### Extraction Improvements

- **Preset slit function**: In some occasions it is useful to extract with a fixed slit function (e.g. from `norm_flat` step) for single-pass extraction (faster, more robust for faint sources)
- **Per-trace extraction heights**: Stored in traces file, computed automatically for fiber bundles
- Convergence based on spectrum change, outlier rejection improved.

### Installation & Build

- Now uses `uv` package manager (`uv sync`, `uv run reduce ...`), but `pip install pyreduce-astro` still works
- **Python 3.11+** required (3.13 recommended)
- Build system migrated from setuptools to Hatchling

See [docs/installation.md](docs/installation.md) for details.
