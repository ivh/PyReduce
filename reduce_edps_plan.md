# PyReduce EDPS Integration Plan

## Overview

Integrate EDPS workflow orchestration into PyReduce, keeping all new code in a top-level `edps/` directory.

## Directory Structure

```
PyReduce/
├── edps/
│   ├── __init__.py
│   ├── workflow/
│   │   ├── __init__.py
│   │   ├── reduce_wkf.py          # Main workflow definition
│   │   ├── reduce_classification.py
│   │   ├── reduce_datasources.py
│   │   ├── reduce_keywords.py
│   │   └── reduce_parameters.yaml
│   └── recipes/
│       ├── __init__.py
│       ├── reduce_bias.py         # Bias recipe wrapper
│       ├── reduce_flat.py
│       ├── reduce_orders.py
│       └── ...
├── pyreduce/
│   └── instruments/
│       ├── crires_plus.yaml       # Already exists
│       └── crires_plus.py         # Already exists
└── pyproject.toml                 # Add edps optional dependency
```

## 1. pyproject.toml Changes

Add after the `[dependency-groups]` section:

```toml
[project.optional-dependencies]
edps = [
    "edps>=1.6.0",
    "pycpl>=1.0.3",
    "pyesorex>=1.0.3",
]

[tool.uv.sources]
pycpl = { index = "pycpl" }

[[tool.uv.index]]
name = "pycpl"
url = "https://ivh.github.io/pycpl/simple/"

[[tool.uv.index]]
name = "eso"
url = "https://ftp.eso.org/pub/dfs/pipelines/libraries/"
```

## 2. Recipe Pattern (from pycr2res example)

```python
# edps/recipes/reduce_bias.py
from typing import Any, Dict
import cpl.core
import cpl.ui

class ReduceBias(cpl.ui.PyRecipe):
    _name = "reduce_bias"
    _version = "0.1"
    _author = "PyReduce"
    _email = "..."
    _copyright = "GPL-3.0-or-later"
    _synopsis = "Create master bias from raw bias frames"
    _description = "Combines bias frames using PyReduce's bias algorithm"

    def __init__(self):
        self.parameters = cpl.ui.ParameterList([
            cpl.ui.ParameterValue(
                name="degree",
                context="reduce_bias",
                description="Polynomial degree for bias fit (0=simple combine)",
                default=0,
            ),
            cpl.ui.ParameterValue(
                name="combine-method",
                context="reduce_bias",
                description="Combination method (median, mean)",
                default="median",
            ),
        ])

    def run(self, frameset: cpl.ui.FrameSet, settings: Dict[str, Any]) -> cpl.ui.FrameSet:
        from pyreduce.combine_frames import combine_bias
        from pyreduce.instruments import load_instrument
        from astropy.io import fits
        import numpy as np

        degree = settings.get("degree", 0)

        # Get input files
        bias_files = [f.file for f in frameset if f.tag == "BIAS"]
        if not bias_files:
            raise ValueError("No BIAS frames provided")

        # Detect instrument from first file header
        with fits.open(bias_files[0]) as hdu:
            header = hdu[0].header
            inst_name = header.get("INSTRUME", "UNKNOWN")

        # Load PyReduce instrument
        instrument = load_instrument(inst_name)

        # Run PyReduce bias combination
        bias, bhead = combine_bias(
            bias_files,
            instrument,
            arm="",  # Will be determined from header
            mask=None,
        )

        # Write output
        output_file = "MASTER_BIAS.fits"
        hdu = fits.PrimaryHDU(data=np.asarray(bias.data, dtype=np.float32), header=bhead)
        hdu.header["HIERARCH ESO PRO CATG"] = "MASTER_BIAS"
        hdu.header["HIERARCH ESO PRO TYPE"] = "REDUCED"
        hdu.writeto(output_file, overwrite=True)

        # Return output frameset
        output = cpl.ui.FrameSet()
        output.append(cpl.ui.Frame(
            file=output_file,
            tag="MASTER_BIAS",
            group=cpl.ui.Frame.FrameGroup.PRODUCT
        ))
        return output
```

## 3. Workflow Files

### edps/workflow/reduce_keywords.py

```python
# Header keywords for classification
instrume = "INSTRUME"
pro_catg = "HIERARCH ESO PRO CATG"
dpr_type = "HIERARCH ESO DPR TYPE"
dpr_catg = "HIERARCH ESO DPR CATG"
mjd_obs = "MJD-OBS"
det_dit = "HIERARCH ESO DET SEQ1 DIT"
ins_wlen_id = "HIERARCH ESO INS WLEN ID"
```

### edps/workflow/reduce_classification.py

```python
from edps import classification_rule
from . import reduce_keywords as kwd

# CRIRES+ specific for now, generalize later
crires = {kwd.instrume: "CRIRES"}

# Raw types
bias_class = classification_rule("BIAS",
    lambda f: f.get(kwd.dpr_type, "").upper() == "DARK")
flat_class = classification_rule("FLAT",
    lambda f: f.get(kwd.dpr_type, "").upper() == "FLAT")
wave_class = classification_rule("WAVE",
    lambda f: "WAVE" in f.get(kwd.dpr_type, "").upper())
science_class = classification_rule("SCIENCE",
    lambda f: f.get(kwd.dpr_catg, "").upper() == "SCIENCE")

# Product types
master_bias_class = classification_rule("MASTER_BIAS", {kwd.pro_catg: "MASTER_BIAS"})
master_flat_class = classification_rule("MASTER_FLAT", {kwd.pro_catg: "MASTER_FLAT"})
orders_class = classification_rule("ORDERS", {kwd.pro_catg: "ORDERS"})
wave_solution_class = classification_rule("WAVE_SOLUTION", {kwd.pro_catg: "WAVE_SOLUTION"})
```

### edps/workflow/reduce_datasources.py

```python
from edps import data_source
from edps.generator.time_range import ONE_DAY, ONE_WEEK, UNLIMITED
from .reduce_classification import *

raw_bias = (data_source("BIAS")
    .with_classification_rule(bias_class)
    .with_grouping_keywords(["mjd-obs"])
    .with_match_keywords(["instrume"], time_range=ONE_WEEK, level=0)
    .with_match_keywords(["instrume"], time_range=UNLIMITED, level=1)
    .build())

raw_flat = (data_source("FLAT")
    .with_classification_rule(flat_class)
    .with_grouping_keywords(["mjd-obs"])
    .with_match_keywords(["instrume"], time_range=ONE_DAY, level=0)
    .with_match_keywords(["instrume"], time_range=ONE_WEEK, level=1)
    .build())

raw_wave = (data_source("WAVE")
    .with_classification_rule(wave_class)
    .with_grouping_keywords(["mjd-obs"])
    .with_match_keywords(["instrume"], time_range=ONE_DAY, level=0)
    .build())

raw_science = (data_source("SCIENCE")
    .with_classification_rule(science_class)
    .with_grouping_keywords(["mjd-obs", "object"])
    .build())
```

### edps/workflow/reduce_wkf.py

```python
from edps import task, SCIENCE, QC1_CALIB
from .reduce_datasources import *
from .reduce_classification import *

__title__ = "PyReduce Generic Workflow"

# Task: Create master bias
bias_task = (task("bias")
    .with_recipe("reduce_bias")
    .with_main_input(raw_bias)
    .with_meta_targets([QC1_CALIB])
    .build())

# Task: Create master flat
flat_task = (task("flat")
    .with_recipe("reduce_flat")
    .with_main_input(raw_flat)
    .with_associated_input(bias_task, [master_bias_class])
    .with_meta_targets([QC1_CALIB])
    .build())

# More tasks to follow...
```

### edps/workflow/reduce_parameters.yaml

```yaml
default_parameters:
  is_default: yes
  recipe_parameters:
    bias:
      reduce.reduce_bias.degree: "0"
      reduce.reduce_bias.combine-method: "median"
    flat:
      reduce.reduce_flat.combine-method: "median"
```

## 4. CRIRES+ Instrument Adapter

The CRIRES+ adapter already exists in PyReduce at:
- `pyreduce/instruments/crires_plus.yaml`
- `pyreduce/instruments/crires_plus.py`

For EDPS, we need to map PyReduce's classification to EDPS classification rules. The key mappings from crires_plus.yaml:

| PyReduce | EDPS | Header Keyword | Pattern |
|----------|------|----------------|---------|
| id_bias | BIAS | ESO DPR TYPE | DARK |
| id_flat | FLAT | ESO DPR TYPE | FLAT |
| id_wave | WAVE | ESO DPR TYPE | WAVE,UNE |
| id_spec | SCIENCE | ESO DPR TYPE | STAR,*,* |

## 5. Implementation Order

1. **pyproject.toml** - Add edps optional dependency group with indexes
2. **Directory structure** - Create edps/workflow/ and edps/recipes/
3. **Keywords + Classification** - reduce_keywords.py, reduce_classification.py
4. **Data sources** - reduce_datasources.py
5. **Bias recipe** - reduce_bias.py (first recipe wrapper)
6. **Workflow** - reduce_wkf.py (with just bias task initially)
7. **Test** - Run `edps -w reduce.reduce_wkf -i <data> -c` to test classification

## 6. Key Files to Reference

- Recipe pattern: `~/pycr2res.git/pyrecipes/cr2res_util_newextract.py`
- EDPS workflow examples: `~/edps/workflows/crires/`
- PyReduce Bias step: `~/PyReduce/pyreduce/reduce.py` (class Bias, line ~499)
- PyReduce combine_bias: `~/PyReduce/pyreduce/combine_frames.py`

## 7. Testing

```bash
cd ~/PyReduce
uv sync --extra edps

# Test classification only
edps -w reduce.reduce_wkf -i /path/to/crires/data -c

# Test bias task
edps -w reduce.reduce_wkf -i /path/to/crires/data -t bias -o /tmp/out
```

## Notes

- Base workflow name is "reduce" (not "echelle" or "generic")
- CRIRES+ uses "DARK" for bias frames (not "BIAS")
- PyReduce's Bias step uses `combine_bias()` from combine_frames.py
- The `instrument` object is needed for FITS I/O quirks (overscan, etc.)
