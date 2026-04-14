---
name: new-instrument
description: Set up a new instrument in PyReduce with config.yaml, settings.json, __init__.py, and example script
---

# Adding a New Instrument to PyReduce

## Overview

Each instrument lives in `pyreduce/instruments/{NAME}/` (uppercase) and needs at minimum three files plus an example script.

## Step-by-step

### 1. Inspect the FITS data

Read headers from one file of each type (bias, flat, wavecal, science) to determine:

- **Detector dimensions**: NAXIS1 x NAXIS2
- **Gain keyword or value**: e.g. `DETGAIN`, or literal like `1.2`
- **Readnoise keyword or value**: may not exist in header, use known value
- **Date keyword**: usually `DATE-OBS`
- **Target keyword**: usually `OBJECT`
- **Instrument keyword**: usually `INSTRUME`, note the value for `id_instrument`
- **File classification keyword**: find which header keyword distinguishes bias/flat/wavecal/science (e.g. `OBSMODE`, `EXPTYPE`, `ESO DPR TYPE`)
- **Classification values**: the values of that keyword for each file type
- **Coordinates**: RA/DEC keywords, observatory lon/lat/alt

To find the **overscan/prescan** regions, compare a flat field with the known active CCD area:
- Look at column means: prescan columns have bias-level values even in flats
- Look at row means: overscan rows drop to bias level at the detector edge
- Confirm: `prescan_x + active_pixels + overscan_x = NAXIS1` (and similarly for y)

### 2. Determine orientation

The **orientation** code rotates the raw image so dispersion runs along x. Try to figure out which raw axis is longer (dispersion) and which is shorter (cross-dispersion). Common orientations:
- `0`: no change
- `1`: 90 deg CCW
- `3`: 270 deg CCW (common when NAXIS2 > NAXIS1 is the dispersion axis)
- `4`: transpose

If uncertain, pick the most likely value and verify with a trace run. Traces should appear as roughly horizontal lines.

### 3. Create `pyreduce/instruments/{NAME}/config.yaml`

Required fields:

```yaml
__instrument__: NAME
instrument: INSTRUME          # header keyword containing instrument name
id_instrument: NAME           # value to match in that keyword
telescope: TelescopeName

date: DATE-OBS
date_format: fits

extension: 0                  # FITS extension with image data
orientation: 3                # rotation code
transpose: false

prescan_x: 53                 # pixels to trim from raw NAXIS1 start
overscan_x: 47                # pixels to trim from raw NAXIS1 end
prescan_y: 0
overscan_y: 64
naxis_x: NAXIS1
naxis_y: NAXIS2

gain: DETGAIN                 # header keyword or literal number
readnoise: 5.5                # header keyword or literal number
dark: 0
sky: 0
exposure_time: EXPTIME

ra: OBJ_RA                    # or RA, etc
dec: OBJ_DEC                  # or DEC, etc
longitude: -17.8792           # observatory coordinates (literal)
latitude: 28.7603
altitude: 2333
target: OBJECT
observation_type: OBSMODE     # header keyword for file type

# File classification: all kw_ fields use the same header keyword
# id_ fields are regex patterns to match against that keyword
kw_bias: OBSMODE
kw_flat: OBSMODE
kw_curvature: OBSMODE
kw_scatter: OBSMODE
kw_orders: OBSMODE
kw_wave: OBSMODE
kw_comb: null
kw_spec: OBSMODE

id_bias: BIAS
id_flat: HRF_FF
id_orders: HRF_FF             # same files used for order tracing
id_curvature: HRF_TH          # or same as flat
id_scatter: HRF_FF
id_wave: HRF_TH
id_comb: null
id_spec: HRF_OBJ
```

**Multi-fiber instruments**: Add a `fibers:` section if the instrument has multiple fibers per order.

```yaml
fibers:
  fibers_per_order: 2          # auto-pairs traces by gap analysis
  groups:
    obj:
      range: [1, 2]            # half-open: fiber_idx 1 only
      merge: center
    sky:
      range: [2, 3]            # half-open: fiber_idx 2 only
      merge: center
  use:
    default: [obj, sky]        # extract each group separately
```

Key points about fiber config:
- `range: [start, end]` is **half-open** (1-based). `[1, 2]` = fiber 1 only, `[2, 3]` = fiber 2 only
- `merge: center` picks the center fiber; `merge: average` averages; `merge: [1]` picks the 1st fiber in the range (1-indexed within range)
- `use.default: [obj, sky]` makes each group extract independently. Without this, all grouped traces are mixed together which causes problems when the extraction height calculation needs overlapping column ranges between neighbors
- Without `use`, the default is `"groups"` which concatenates all grouped traces into one list -- this breaks extraction for dual-fiber instruments because neighboring traces (obj/sky of same order) are only ~6px apart and edge orders may have non-overlapping column ranges

### 4. Create `pyreduce/instruments/{NAME}/settings.json`

Inherit defaults and override what's needed:

```json
{
    "__instrument__": "NAME",
    "__inherits__": "defaults/settings.json",
    "trace": {
        "degree": 6,
        "noise": 50,
        "min_cluster": 3000,
        "filter_y": 120
    },
    "norm_flat": {
        "extraction_height": 6,
        "oversampling": 10
    },
    "wavecal_master": {
        "extraction_height": 6
    },
    "science": {
        "extraction_height": 6,
        "oversampling": 10
    }
}
```

**Important**: For multi-fiber instruments, set `extraction_height` to an explicit pixel value (>= 2) in norm_flat, wavecal_master, and science. A fractional value like `0.5` triggers neighbor-distance calculation which fails when edge orders have non-overlapping column ranges. The pixel value should be less than half the inter-order separation between same-group traces.

### 5. Create `pyreduce/instruments/{NAME}/__init__.py`

Minimal version:

```python
"""Instrument-specific info for {NAME}."""

import logging

from ..common import Instrument

logger = logging.getLogger(__name__)


class NAME(Instrument):
    def add_header_info(self, header, channel, **kwargs):
        header = super().add_header_info(header, channel)
        return header
```

The class name **must** match the directory name (uppercase). Override methods only if needed (e.g. RA unit conversion, JD offset, custom wavecal filename).

### 6. Create `examples/{name}_example.py`

Use `Pipeline.from_instrument()` for instruments with proper config:

```python
from pyreduce.pipeline import Pipeline
import os

base_dir = os.environ.get("REDUCE_DATA", os.path.expanduser("~/REDUCE_DATA"))

data = Pipeline.from_instrument(
    instrument="NAME",
    target="TargetName",
    night="2026-04-12",
    channel="",
    steps=("bias", "flat", "trace", "norm_flat", "wavecal_master", "science"),
    base_dir=base_dir,
    input_dir=os.path.join(base_dir, "NAME"),
    plot=1,
).run()
```

### 7. Verify

Test incrementally:
1. `load_instrument('NAME')` -- config loads without errors
2. `sort_files(...)` -- files classified correctly
3. Run `bias, flat, trace` -- check trace count and positions
4. Run `norm_flat` -- extraction succeeds
5. Run `wavecal_master` -- extraction succeeds
6. Run `science` -- extraction succeeds

Use `PYREDUCE_PLOT=0` for headless testing.

## Common pitfalls

- **Orientation wrong**: traces appear vertical or diagonal instead of horizontal. Try different orientation codes.
- **Prescan/overscan wrong**: images have bright/dark strips at edges. Check flat field column/row means to identify the transition from bias level to signal.
- **File classification misses files**: check that `kw_*` points to the right header keyword and `id_*` patterns match (case-insensitive regex).
- **Edge orders cause extraction failures**: partial orders at detector edges have short column ranges that don't overlap with neighbors. Fix by using explicit pixel extraction heights (>= 2) instead of fractional values.
- **Multi-fiber extraction crashes**: grouped traces from different fibers are interleaved with small separations (~6px). Always set `use.default: [group1, group2]` to extract groups independently.
- **"No instrument channels found"**: harmless warning when `channels` is not set in config. Only needed for instruments with separate detector chips/arms.

## Optional additions

- `order_centers_{channel}.yaml`: Known y-positions for order matching (assigns physical order numbers `m`)
- `wavecal_*.npz`: Pre-computed wavelength solutions
- `mask_*.fits.gz`: Bad pixel masks
- `settings_{channel}.json`: Per-channel setting overrides (uses `"__inherits__": "{NAME}/settings.json"`)
