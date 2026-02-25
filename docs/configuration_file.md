# Configuration

PyReduce uses two types of configuration files:

- **Instrument configs** (YAML) - Define the instrument hardware and header mappings
- **Reduction settings** (JSON) - Define algorithm parameters for each step

## Reduction Settings

Location: `pyreduce/instruments/{INSTRUMENT}/settings.json`

These control HOW the reduction is performed - polynomial degrees, thresholds,
extraction parameters, etc.

```json
{
  "bias": {
    "degree": 0
  },
  "trace": {
    "degree": 4,
    "noise": 100,
    "min_cluster": 500,
    "filter_y": 120
  },
  "science": {
    "extraction_method": "optimal",
    "extraction_height": null,
    "oversampling": 10
  }
}
```

### Settings Cascade

Settings are resolved by merging files in order, with later levels overriding earlier ones:

1. `instruments/defaults/settings.json` — base defaults
2. `instruments/{INSTRUMENT}/settings.json` — instrument-specific overrides
3. `instruments/{INSTRUMENT}/settings_{channel}.json` — per-channel overrides (if channel is specified and the file exists)
4. Runtime overrides via `configuration` parameter or keyword arguments

Each file only needs to specify the values it wants to change. Missing keys are inherited from the parent file.

### Inheritance (`__inherits__`)

Every settings file (except `defaults/settings.json`) declares its parent via `__inherits__`:

```json
{
    "__inherits__": "ANDES_RIZ/settings.json",
    "curvature": {
        "curve_height": 171,
        "extraction_height": 20
    }
}
```

The path is relative to `pyreduce/instruments/`. Inheritance is resolved recursively — a per-channel file inherits from the instrument file, which inherits from defaults.

You can also inherit from another channel's settings to avoid duplication. For example, MOSAIC VIS2-VIS4 inherit from VIS1:

```json
{
    "__inherits__": "MOSAIC/settings_VIS1.json"
}
```

### Per-Channel Settings

Instruments with multiple channels often need different parameters (extraction heights, curvature settings, etc.). Create `settings_{channel}.json` files alongside `settings.json`:

```
pyreduce/instruments/ANDES_RIZ/
    settings.json           # Shared settings for all ANDES_RIZ channels
    settings_r.json         # Overrides for R channel
    settings_iz.json        # Overrides for IZ channel
```

When you pass `channel=` to the pipeline, the per-channel file is loaded automatically:

```python
Pipeline.from_instrument("ANDES_RIZ", channel="r", ...)  # loads settings_r.json
Pipeline.from_instrument("ANDES_RIZ", ...)                # loads settings.json
```

If `settings_{channel}.json` doesn't exist, it falls back to `settings.json`.

To override settings at runtime:

```python
from pyreduce.configuration import get_configuration_for_instrument

config = get_configuration_for_instrument("UVES")
config["trace"]["degree"] = 5
config["science"]["oversampling"] = 8

Pipeline.from_instrument(
    instrument="UVES",
    ...,
    configuration=config,
).run()
```

## Instrument Configs

Location: `pyreduce/instruments/{INSTRUMENT}/config.yaml`

These define WHAT the instrument is - detector properties, header keyword
mappings, file classification patterns.

```yaml
# Identity
instrument: HARPS
telescope: ESO-3.6m
channels: [red, blue]

# Detector
naxis: [4096, 4096]
orientation: 4
extension: 0
gain: ESO DET OUT1 CONAD
readnoise: ESO DET OUT1 RON

# Header mappings
date: DATE-OBS
target: ESO OBS TARG NAME
exposure_time: EXPTIME

# File classification
kw_bias: ESO DPR TYPE
id_bias: BIAS
kw_flat: ESO DPR TYPE
id_flat: FLAT.*
```

Instrument configs are validated by Pydantic models at load time.
See `pyreduce/instruments/models.py` for the full schema.

## Common Settings

### Trace (Order Tracing)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `degree` | Polynomial degree for trace fitting | 4 |
| `noise` | Absolute noise threshold for detection | 0 |
| `noise_relative` | Relative noise threshold (fraction of image max) | 0 |
| `min_cluster` | Minimum pixels for valid order | 500 |
| `filter_y` | Median filter size in y | null |
| `filter_x` | Median filter size in x | 0 |
| `filter_type` | Filter type ("boxcar" or "median") | "boxcar" |
| `border_width` | Pixels to ignore at edges. Int or `[top, bottom, left, right]` | null |

Use either `noise` (absolute threshold) or `noise_relative` (e.g., 0.01 for 1% of image maximum) for trace detection.

### Science (Extraction)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `extraction_method` | "optimal" or "simple" | "optimal" |
| `extraction_height` | Extraction aperture (see below) | null |
| `oversampling` | Slit function oversampling | 10 |
| `smooth_slitfunction` | Smoothing factor | 0.1 |
| `smooth_spectrum` | Spectrum smoothing factor | 1e-7 |
| `swath_width` | Width of extraction swaths | 300 |
| `extraction_reject` | Sigma threshold for outlier rejection | 6 |
| `maxiter` | Maximum extraction iterations | 30 |

#### extraction_height

The extraction aperture can be specified as:

- **`null`** (default) - Use per-trace heights computed during tracing, stored in `traces.fits`. This provides optimal apertures based on actual trace spacing.
- **Pixels** (≥2) - Explicit pixel height, e.g., `20` for 20 pixels total (10 above, 10 below trace)
- **Fraction** (<2) - Fraction of order separation, e.g., `0.5` for half the distance to neighbors

The automatic heights (null) are recommended for most cases. They adapt to varying trace spacing across the detector and between orders.

#### Using a Pre-computed Slit Function

For faster extraction, the slit function computed during `norm_flat` can be reused in subsequent steps. The normalized flat step saves the slit function to `.flat_norm.npz` with metadata (extraction_height, osample). To use it:

```python
import numpy as np
from pyreduce.extract import extract

# Load slit function from norm_flat output
norm_data = np.load("output/uves.flat_norm.npz", allow_pickle=True)
slitfunc_list = list(norm_data["slitfunc"])
slitfunc_meta = norm_data["slitfunc_meta"].item()

# Extract with preset slit function
spectra = extract(
    image,
    traces,
    extraction_height=slitfunc_meta["extraction_height"],
    osample=slitfunc_meta["osample"],
    preset_slitfunc=slitfunc_list,
)
```

This performs single-pass extraction without iterating to find the slit function shape, which is useful for instruments with stable slit profiles.

### Wavelength Calibration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `degree` | Polynomial degree [x, order] | [6, 6] |
| `threshold` | Line detection threshold | 100 |
| `iterations` | Refinement iterations | 3 |
| `medium` | Refractive medium ("air" or "vacuum") | "air" |
| `dimensionality` | "1D" (per-trace) or "2D" (shared polynomial) | "2D" |
| `atlas` | Line atlas name ("thar", "une", "lfc", etc.) | "thar" |

#### Per-Group Wavelength Calibration

For multi-fiber instruments, wavelength calibration can process fiber groups
separately. This is configured via `fibers.use.wavecal` in `config.yaml`:

```yaml
fibers:
  groups:
    A: {range: [1, 36], merge: average}
    cal: {range: [37, 40], merge: average}
    B: {range: [40, 76], merge: average}

  use:
    wavecal: [A, B]      # Separate calibration for each group
    # OR
    wavecal: [cal]       # Use only calibration fiber
    # OR
    wavecal: per_fiber   # Separate calibration per fiber_idx
    # OR
    wavecal: all         # All traces together (default)
```

- **`[A, B]`** - Calibrate each named group separately. Useful when groups have
  different optical paths (e.g., science fibers A and B).

- **`per_fiber`** - Calibrate each fiber index separately. Each unique
  `fiber_idx` value gets its own wavelength polynomial. Use this when individual
  fibers within a group have slightly different wavelength solutions and you
  want maximum precision.

- **`all`** - Combine all traces into a single calibration (default for
  single-fiber instruments).

The wavelength polynomial is stored in each `Trace.wave` attribute and saved to
`traces.fits`. Subsequent steps (science extraction, continuum normalization)
read wavelengths directly from the traces.

See [Fiber Bundle Configuration](fiber_bundle_tracing.md) for full details on
multi-fiber setup.

### Continuum Normalization

| Parameter | Description | Default |
|-----------|-------------|---------|
| `degree` | Polynomial degree for fit | 5 |
| `sigma` | Sigma clipping threshold | 3 |
| `iterations` | Fit iterations | 5 |
