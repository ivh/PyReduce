# Configuration

PyReduce uses two types of configuration files:

- **Instrument configs** (YAML) - Define the instrument hardware and header mappings
- **Reduction settings** (JSON) - Define algorithm parameters for each step

## Reduction Settings

Location: `pyreduce/settings/settings_*.json`

These control HOW the reduction is performed - polynomial degrees, thresholds,
extraction parameters, etc.

```json
{
  "bias": {
    "degree": 0
  },
  "orders": {
    "degree": 4,
    "noise": 100,
    "min_cluster": 500,
    "filter_size": 120
  },
  "science": {
    "extraction_method": "optimal",
    "extraction_width": 0.5,
    "oversampling": 10
  }
}
```

Settings are loaded in order:

1. `settings_pyreduce.json` - Base defaults
2. `settings_INSTRUMENT.json` - Instrument-specific overrides
3. Runtime overrides via `configuration` parameter

To override settings at runtime:

```python
from pyreduce.configuration import get_configuration_for_instrument

config = get_configuration_for_instrument("UVES")
config["orders"]["degree"] = 5
config["science"]["oversampling"] = 8

Pipeline.from_instrument(
    instrument="UVES",
    ...,
    configuration=config,
).run()
```

## Instrument Configs

Location: `pyreduce/instruments/*.yaml`

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

### Orders (Order Tracing)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `degree` | Polynomial degree for trace fitting | 4 |
| `noise` | Noise threshold for detection | 100 |
| `min_cluster` | Minimum pixels for valid order | 500 |
| `filter_size` | Median filter size | 120 |
| `border_width` | Pixels to ignore at edges | 10 |

### Science (Extraction)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `extraction_method` | "optimal" or "arc" | "optimal" |
| `extraction_width` | Width in order widths | 0.5 |
| `oversampling` | Slit function oversampling | 10 |
| `smooth_slitfunction` | Smoothing factor | 1 |
| `swath_width` | Width of extraction swaths | 300 |

### Wavelength Calibration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `degree` | Polynomial degree [x, order] | [6, 6] |
| `threshold` | Line detection threshold | 100 |
| `iterations` | Refinement iterations | 3 |
| `medium` | Refractive medium ("air" or "vacuum") | "air" |

### Continuum Normalization

| Parameter | Description | Default |
|-----------|-------------|---------|
| `degree` | Polynomial degree for fit | 5 |
| `sigma` | Sigma clipping threshold | 3 |
| `iterations` | Fit iterations | 5 |
