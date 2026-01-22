# How To Use PyReduce

PyReduce offers two ways to run reductions: the command-line interface (CLI) and the Python API.

## Image Coordinate Convention

PyReduce uses the convention that **dispersion runs horizontally (along x-axis)** and **cross-dispersion runs vertically (along y-axis)**. The `clipnflip()` function rotates and flips raw images from each instrument to ensure this standard orientation.

This means:

- **Columns (x)** = wavelength/dispersion direction
- **Rows (y)** = spatial/cross-dispersion direction
- **Traces** are polynomial functions of x, giving y-position
- **`extraction_height`** refers to pixels above/below each trace (in y)

## Command Line Interface

The CLI is the simplest way to run reductions. See [CLI Reference](cli.md) for full details.

Quick start:

```bash
# Download sample data
uv run reduce download UVES

# Run full pipeline
uv run reduce run UVES HD132205 --steps bias,flat,trace,science

# Run individual steps
uv run reduce bias UVES HD132205
uv run reduce trace UVES HD132205

# List available steps
uv run reduce list-steps
```

## Python API

The recommended Python entry point is `Pipeline.from_instrument()`:

```python
from pyreduce.pipeline import Pipeline

result = Pipeline.from_instrument(
    instrument="UVES",
    target="HD132205",
    night="2010-04-01",
    channel="middle",
    steps=("bias", "flat", "trace", "science"),
    plot=1,
).run()
```

This handles:

- Loading the instrument configuration
- Finding and sorting input files
- Setting up output directories
- Running the requested steps

For more control, construct a Pipeline manually:

```python
from pyreduce.pipeline import Pipeline

pipe = Pipeline(
    instrument="UVES",
    output_dir="/data/reduced",
    channel="middle",
)
pipe.bias(bias_files)
pipe.flat(flat_files)
pipe.trace()
pipe.extract(science_files)
result = pipe.run()
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `REDUCE_DATA` | Base data directory (default: `~/REDUCE_DATA`) |
| `PYREDUCE_PLOT` | Override plot level (0=off, 1=basic, 2=detailed) |
| `PYREDUCE_PLOT_DIR` | Save plots to directory as PNG files |
| `PYREDUCE_PLOT_SHOW` | Display mode: `block` (default), `defer`, or `off` |
| `PYREDUCE_PLOT_ANIMATION_SPEED` | Frame delay in seconds for extraction animation (default: 0.3) |

## Plot Modes

PyReduce supports three plot display modes via `PYREDUCE_PLOT_SHOW`:

| Mode | Description |
|------|-------------|
| `block` | Show each plot interactively, blocking until closed (default) |
| `defer` | Accumulate all plots, show together at end of pipeline |
| `off` | Don't display plots (useful with `PYREDUCE_PLOT_DIR` to save only) |

Save and display are independent—you can save to files AND display:

```bash
# Save only (headless/CI)
PYREDUCE_PLOT=1 PYREDUCE_PLOT_DIR=/tmp/plots PYREDUCE_PLOT_SHOW=off uv run ...

# Show all at end (useful with webagg backend for browser viewing)
MPLBACKEND=webagg PYREDUCE_PLOT=1 PYREDUCE_PLOT_SHOW=defer uv run ...

# Save AND show all at end
PYREDUCE_PLOT=1 PYREDUCE_PLOT_DIR=/tmp/plots PYREDUCE_PLOT_SHOW=defer uv run ...
```

Note: Plot level 2 (interactive progress plots during extraction) only works with `block` mode.

### Extraction Animation

When `plot=2`, extraction shows an animated progress plot with the current swath, slit function fit, and residuals. Interactive controls are available:

- **Pause/Resume** button - pause animation to examine current state
- **Step** button - advance one swath at a time (when paused)
- **Speed slider** - adjust animation frame rate

Set `PYREDUCE_PLOT_ANIMATION_SPEED` to control the default frame delay (in seconds). A value of 0 shows swaths as fast as possible.

## Pipeline Steps

Steps are run in dependency order. Available steps:

| Step | Description |
|------|-------------|
| `bias` | Combine bias frames |
| `flat` | Combine flat frames |
| `trace` | Trace echelle orders on flat |
| `curvature` | Measure slit curvature |
| `scatter` | Model inter-order background |
| `norm_flat` | Normalize flat, extract blaze |
| `wavecal_master` | Extract wavelength calibration spectrum |
| `wavecal_init` | Initial line identification |
| `wavecal` | Refine wavelength solution |
| `freq_comb_master` | Extract frequency comb spectrum |
| `freq_comb` | Apply frequency comb calibration |
| `science` | Optimally extract science spectra |
| `continuum` | Normalize continuum |
| `finalize` | Write final output |

Steps not explicitly requested but required as dependencies will be loaded
from previous runs if available, or executed automatically.
