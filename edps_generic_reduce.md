# Generic Echelle Reduction Workflow with EDPS

## Overview

This document outlines a design for using ESO's EDPS (ESO Data Processing System) to provide robust data classification, calibration association, and workflow orchestration for a generic echelle spectrograph reduction pipeline (like PyReduce).

## The Problem

PyReduce currently has "half-shoddy" data handling:
- Filter-based classification via regex/wildcard matching against FITS headers
- Implicit temporal matching ("closest night" fallback) without formalized quality levels
- No serializable intermediate state - can't checkpoint/resume
- Tuple-passing between steps with no unified data model
- Works fine for single-night reductions but not robust for automated large datasets

## What EDPS Offers

### Core Strengths

1. **Sophisticated classification system**
   - Dictionary-based rules: `classification_rule("BIAS", {kwd.dpr_type: "BIAS"})`
   - Function-based rules: `classification_rule("FLAT", is_flat_func)`
   - Multiple rules per data source

2. **Quality-level calibration association**
   - Time-range matching: `SAME_NIGHT`, `ONE_DAY`, `ONE_WEEK`, `UNLIMITED`
   - Quality levels 0-3 (0 = calibration plan, 3 = risky)
   - Automatic fallback through quality levels

3. **Flexible grouping and matching**
   - Group files by keywords: `.with_grouping_keywords(["mjd-obs", "object"])`
   - Match by setup: `.with_match_keywords([kwd.ins_wlen_id, kwd.ins_slit1_id])`
   - Custom match functions for complex logic

4. **DAG-based workflow execution**
   - Automatic dependency resolution
   - Parallelization via `processes=N` setting
   - Smart reruns (skip unchanged jobs via bookkeeping)

5. **Alternative inputs**
   - Graceful fallback: prefer flat for order tracing, fall back to arc
   - `alternative_associated_inputs()` for multiple valid calibration paths

### EDPS Dependencies

```
edps (core)
├── astropy, fastapi, networkx, pydantic, pyyaml, requests, tinydb, uvicorn
└── NO pycpl dependency

pyesorex (for recipe execution)
└── pycpl (CPL Python bindings)
```

If using `with_recipe()`: need pyesorex + pycpl
If using `with_function()`: pure Python, no pycpl needed

---

## Architecture

```
generic_echelle/
├── __init__.py
├── adapters.py           # Instrument keyword mappings + static calibs
├── keywords.py           # Common keyword definitions
├── classification.py     # Generic classification rules using adapters
├── datasources.py        # Generic data sources with association rules
├── workflow.py           # Main workflow definition
├── subworkflows.py       # Reusable workflow pieces (calibration cascade)
├── functions.py          # Step implementations (for with_function path)
├── parameters.yaml       # Default parameters per task
│
├── instruments/          # Instrument-specific overrides
│   ├── uves_wkf.py      # UVES (multi-arm)
│   ├── harps_wkf.py     # HARPS
│   ├── xshooter_wkf.py  # XSHOOTER (3 arms)
│   └── mcdonald_wkf.py  # Non-ESO instrument
│
├── recipes/              # pyesorex recipe wrappers (default path)
│   ├── pyreduce_bias.py
│   ├── pyreduce_flat.py
│   ├── pyreduce_wavecal.py
│   └── pyreduce_extract.py
│
└── static/               # Static calibration files
    ├── uves/
    │   ├── linelist_thar.fits
    │   └── badpix_blue.fits
    └── harps/
        └── linelist_thar.fits
```

---

## Instrument Adapter Layer

The key to supporting both ESO and non-ESO instruments is an adapter that maps instrument-specific header keywords to generic concepts:

```python
# generic_echelle/adapters.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Callable

STATIC_DIR = Path(__file__).parent / "static"

@dataclass
class InstrumentAdapter:
    name: str

    # Header keyword that identifies instrument
    kw_instrume: str = "INSTRUME"

    # Keyword containing observation type
    kw_obstype: str = "OBSTYPE"  # or "ESO DPR TYPE", "IMAGETYP", etc.

    # Patterns for each file type (matched case-insensitively)
    id_bias: tuple = ("BIAS",)
    id_flat: tuple = ("FLAT", "LAMP,FLAT")
    id_arc: tuple = ("ARC", "WAVE", "WAVE,THAR")
    id_science: tuple = ("SCIENCE", "OBJECT")

    # Grouping keywords
    kw_night: str = "DATE-OBS"
    kw_arm: Optional[str] = None          # e.g., "ESO INS PATH" for UVES
    kw_setting: tuple = ()                # grating, filter, slit, etc.

    # Static calibration files (auto-discovered from static/{name}/)
    linelist: Optional[Path] = None
    badpix_mask: Optional[Path] = None

    def __post_init__(self):
        inst_dir = STATIC_DIR / self.name.lower()
        if inst_dir.exists():
            for name in ["linelist.fits", "linelist_thar.fits"]:
                if (inst_dir / name).exists():
                    self.linelist = inst_dir / name
                    break
            if (inst_dir / "badpix.fits").exists():
                self.badpix_mask = inst_dir / "badpix.fits"


# Registry of known instruments
ADAPTERS = {
    "UVES": InstrumentAdapter(
        name="UVES",
        kw_obstype="ESO DPR TYPE",
        kw_arm="ESO INS PATH",
        kw_setting=("ESO INS GRAT1 WLEN", "ESO INS SLIT2 WID"),
        id_bias=("BIAS",),
        id_flat=("FLAT", "LAMP,FLAT"),
        id_arc=("WAVE,THAR", "WAVE,UNE"),
    ),
    "HARPS": InstrumentAdapter(
        name="HARPS",
        kw_obstype="ESO DPR TYPE",
        id_flat=("FLAT,LAMP", "ORDERDEF,LAMP"),
    ),
    "CRIRES": InstrumentAdapter(
        name="CRIRES",
        kw_obstype="ESO DPR TYPE",
        kw_arm="ESO INS WLEN ID",
        id_flat=("FLAT",),
        id_arc=("WAVE,UNE", "WAVE,FPET"),
    ),
    "MCDONALD": InstrumentAdapter(
        name="McDonald",
        kw_obstype="IMAGETYP",
        id_bias=("zero",),
        id_flat=("flat",),
        id_arc=("comp",),
        id_science=("object",),
    ),
    # Add more instruments as needed
}

def get_adapter(header_or_file) -> Optional[InstrumentAdapter]:
    """Detect instrument from FITS header and return appropriate adapter."""
    # Try each adapter's instrument keyword
    for name, adapter in ADAPTERS.items():
        instrume = header_or_file.get(adapter.kw_instrume, "")
        if instrume and name.upper() in instrume.upper():
            return adapter
    return None
```

---

## Classification Rules

Generic classification functions that use the adapter layer:

```python
# generic_echelle/classification.py
from edps import classification_rule
from .adapters import get_adapter, ADAPTERS

def is_bias(f):
    adapter = get_adapter(f)
    if not adapter:
        return False
    obstype = str(f.get(adapter.kw_obstype, "")).upper()
    return any(pat.upper() in obstype for pat in adapter.id_bias)

def is_flat(f):
    adapter = get_adapter(f)
    if not adapter:
        return False
    obstype = str(f.get(adapter.kw_obstype, "")).upper()
    return any(pat.upper() in obstype for pat in adapter.id_flat)

def is_arc(f):
    adapter = get_adapter(f)
    if not adapter:
        return False
    obstype = str(f.get(adapter.kw_obstype, "")).upper()
    return any(pat.upper() in obstype for pat in adapter.id_arc)

def is_science(f):
    adapter = get_adapter(f)
    if not adapter:
        return False
    obstype = str(f.get(adapter.kw_obstype, "")).upper()
    return any(pat.upper() in obstype for pat in adapter.id_science)

def is_order_source(f):
    """Accept flat OR arc for order tracing."""
    return is_flat(f) or is_arc(f)

# Classification rules
bias_class = classification_rule("BIAS", is_bias)
flat_class = classification_rule("FLAT", is_flat)
arc_class = classification_rule("ARC", is_arc)
science_class = classification_rule("SCIENCE", is_science)
order_def_class = classification_rule("ORDER_DEF", is_order_source)

# Product classifications (by PRO.CATG header)
master_bias_class = classification_rule("MASTER_BIAS", {"pro.catg": "MASTER_BIAS"})
master_flat_class = classification_rule("MASTER_FLAT", {"pro.catg": "MASTER_FLAT"})
orders_class = classification_rule("ORDERS", {"pro.catg": "ORDERS"})
wave_solution_class = classification_rule("WAVE_SOLUTION", {"pro.catg": "WAVE_SOLUTION"})
spectrum_1d_class = classification_rule("SPECTRUM_1D", {"pro.catg": "SPECTRUM_1D"})
```

---

## Data Sources

```python
# generic_echelle/datasources.py
from edps import data_source, match_rules
from edps.generator.time_range import *
from .classification import *

# Raw calibration data sources
raw_bias = (data_source("BIAS")
    .with_classification_rule(bias_class)
    .with_grouping_keywords(["mjd-obs"])
    .with_match_keywords(["instrume"], time_range=ONE_WEEK, level=0)
    .with_match_keywords(["instrume"], time_range=ONE_MONTH, level=1)
    .with_match_keywords(["instrume"], time_range=UNLIMITED, level=2)
    .build())

raw_flat = (data_source("FLAT")
    .with_classification_rule(flat_class)
    .with_grouping_keywords(["mjd-obs"])
    .with_match_keywords(["instrume"], time_range=ONE_DAY, level=0)
    .with_match_keywords(["instrume"], time_range=ONE_WEEK, level=1)
    .with_match_keywords(["instrume"], time_range=UNLIMITED, level=2)
    .build())

# Order definition - prefer flat (level 0), fall back to arc (level 1)
raw_order_def = (data_source("ORDER_DEF")
    .with_classification_rule(flat_class)
    .with_classification_rule(arc_class)
    .with_grouping_keywords(["mjd-obs"])
    .with_match_function(lambda ref, f: is_flat(f), time_range=ONE_DAY, level=0)
    .with_match_function(lambda ref, f: is_arc(f), time_range=ONE_DAY, level=1)
    .build())

raw_arc = (data_source("ARC")
    .with_classification_rule(arc_class)
    .with_grouping_keywords(["mjd-obs"])
    .with_match_keywords(["instrume"], time_range=SAME_NIGHT, level=0)
    .with_match_keywords(["instrume"], time_range=ONE_DAY, level=1)
    .with_match_keywords(["instrume"], time_range=ONE_WEEK, level=2)
    .build())

raw_science = (data_source("SCIENCE")
    .with_classification_rule(science_class)
    .with_grouping_keywords(["mjd-obs", "object"])
    .build())

# Static calibration data sources
linelist = (data_source("LINELIST")
    .with_classification_rule(classification_rule("LINELIST", {"pro.catg": "LINELIST"}))
    .with_match_keywords(["instrume"], time_range=UNLIMITED, level=0)
    .build())
```

---

## Workflow Definition

```python
# generic_echelle/workflow.py
from edps import task, SCIENCE, QC1_CALIB
from .datasources import *
from .classification import *

__title__ = "Generic Echelle Reduction Workflow"

# Task: Create master bias
bias_task = (task("bias")
    .with_recipe("pyreduce_bias")
    .with_main_input(raw_bias)
    .with_meta_targets([QC1_CALIB])
    .build())

# Task: Create master flat + trace orders
flat_task = (task("flat")
    .with_recipe("pyreduce_flat")
    .with_main_input(raw_flat)
    .with_associated_input(bias_task, [master_bias_class])
    .with_meta_targets([QC1_CALIB])
    .build())

# Task: Trace orders (can use flat or arc)
orders_task = (task("orders")
    .with_recipe("pyreduce_orders")
    .with_main_input(raw_order_def)
    .with_associated_input(bias_task, [master_bias_class])
    .with_meta_targets([QC1_CALIB])
    .build())

# Task: Wavelength calibration
wavecal_task = (task("wavecal")
    .with_recipe("pyreduce_wavecal")
    .with_main_input(raw_arc)
    .with_associated_input(bias_task, [master_bias_class])
    .with_associated_input(orders_task, [orders_class])
    .with_associated_input(linelist, min_ret=0)  # optional static calib
    .with_meta_targets([QC1_CALIB])
    .build())

# Task: Science extraction
extract_task = (task("extract")
    .with_recipe("pyreduce_extract")
    .with_main_input(raw_science)
    .with_associated_input(bias_task, [master_bias_class])
    .with_associated_input(flat_task, [master_flat_class])
    .with_associated_input(orders_task, [orders_class])
    .with_associated_input(wavecal_task, [wave_solution_class])
    .with_meta_targets([SCIENCE])
    .build())
```

---

## Recipe Wrappers (Default Path)

Each PyReduce step wrapped as a pyesorex recipe:

```python
# generic_echelle/recipes/pyreduce_bias.py
"""
pyesorex recipe wrapping PyReduce bias step.

Recipe inputs:
  - BIAS: raw bias frames

Recipe outputs:
  - MASTER_BIAS: combined master bias frame
"""
import cpl
from astropy.io import fits

class pyreduce_bias(cpl.Recipe):
    name = "pyreduce_bias"
    version = "1.0"
    author = "PyReduce"
    synopsis = "Create master bias from raw bias frames"
    description = "Combines bias frames using PyReduce's Bias step"

    parameters = cpl.ParameterList([
        cpl.Parameter("pyreduce.pyreduce_bias.combine_method",
                     cpl.Type.STRING, "median",
                     "Method for combining frames (median, mean)"),
        cpl.Parameter("pyreduce.pyreduce_bias.sigma_clip",
                     cpl.Type.DOUBLE, 3.0,
                     "Sigma clipping threshold"),
    ])

    def run(self, frameset):
        from pyreduce.reduce import Bias
        from pyreduce.instruments import instrument_info

        # Get input files
        bias_frames = [f.file for f in frameset.get_frames("BIAS")]
        if not bias_frames:
            raise cpl.CplError("No BIAS frames provided")

        # Detect instrument
        instrument = instrument_info.detect(bias_frames[0])

        # Get parameters
        combine_method = self.parameters["pyreduce.pyreduce_bias.combine_method"].value
        sigma_clip = self.parameters["pyreduce.pyreduce_bias.sigma_clip"].value

        # Run PyReduce
        step = Bias(instrument, method=combine_method, sigma=sigma_clip)
        master_bias, header = step.run(bias_frames)

        # Write output
        output_name = "MASTER_BIAS.fits"
        hdu = fits.PrimaryHDU(master_bias, header=header)
        hdu.header["HIERARCH ESO PRO CATG"] = "MASTER_BIAS"
        hdu.header["HIERARCH ESO PRO TYPE"] = "REDUCED"
        hdu.writeto(output_name, overwrite=True)

        # Return frameset
        output = cpl.FrameSet()
        output.append(cpl.Frame(output_name, tag="MASTER_BIAS"))
        return output
```

Similar wrappers for `pyreduce_flat`, `pyreduce_orders`, `pyreduce_wavecal`, `pyreduce_extract`.

---

## Parameter File

```yaml
# generic_echelle/parameters.yaml
default_parameters:
  is_default: yes
  recipe_parameters:
    bias:
      pyreduce.pyreduce_bias.combine_method: "median"
      pyreduce.pyreduce_bias.sigma_clip: "3.0"
    flat:
      pyreduce.pyreduce_flat.combine_method: "median"
      pyreduce.pyreduce_flat.normalize: "true"
    orders:
      pyreduce.pyreduce_orders.degree: "4"
      pyreduce.pyreduce_orders.min_order_separation: "10"
    wavecal:
      pyreduce.pyreduce_wavecal.degree: "6"
      pyreduce.pyreduce_wavecal.threshold: "100"
    extract:
      pyreduce.pyreduce_extract.extraction_width: "10"
      pyreduce.pyreduce_extract.cosmic_sigma: "5.0"
      pyreduce.pyreduce_extract.method: "optimal"
  workflow_parameters:
    order_source: "flat"  # or "arc"
    wavelength_method: "thar"  # or "comb"

science_parameters:
  <<: *default
  is_default: no
  workflow_parameters:
    order_source: "flat"
```

---

## Instrument-Specific Workflows

For instruments needing customization (e.g., multi-arm):

```python
# generic_echelle/instruments/uves_wkf.py
"""UVES-specific workflow - handles blue/red arms separately."""
from edps import task, data_source, classification_rule
from generic_echelle.workflow import bias_task  # reuse generic bias
from generic_echelle.classification import master_bias_class, master_flat_class

# UVES arm-specific classification
def is_uves_blue(f):
    return f.get("instrume") == "UVES" and f.get("ins.path") == "BLUE"

def is_uves_red(f):
    return f.get("instrume") == "UVES" and f.get("ins.path") == "RED"

# Separate data sources per arm
raw_science_blue = (data_source("SCIENCE_BLUE")
    .with_classification_rule(classification_rule("SCIENCE", is_uves_blue))
    .with_grouping_keywords(["mjd-obs", "object"])
    .build())

raw_science_red = (data_source("SCIENCE_RED")
    .with_classification_rule(classification_rule("SCIENCE", is_uves_red))
    .with_grouping_keywords(["mjd-obs", "object"])
    .build())

# Arm-specific extraction tasks
extract_blue = (task("extract_blue")
    .with_recipe("pyreduce_extract")
    .with_main_input(raw_science_blue)
    .with_associated_input(bias_task, [master_bias_class])
    # ... arm-matched calibrations
    .build())

extract_red = (task("extract_red")
    .with_recipe("pyreduce_extract")
    .with_main_input(raw_science_red)
    .with_associated_input(bias_task, [master_bias_class])
    # ... arm-matched calibrations
    .build())
```

---

## Conditional Execution

For optional steps (e.g., dark subtraction for IR instruments):

```python
from edps import task, get_parameter, JobParameters

def use_darks(params: JobParameters) -> bool:
    return get_parameter(params, "use_darks", "false").lower() == "true"

def is_infrared(params: JobParameters) -> bool:
    return get_parameter(params, "detector_type") == "infrared"

# Conditional dark association
flat_task = (task("flat")
    .with_recipe("pyreduce_flat")
    .with_main_input(raw_flat)
    .with_associated_input(bias_task, [master_bias_class])
    .with_associated_input(dark_task, [master_dark_class],
                          min_ret=0,          # optional
                          condition=use_darks) # only if parameter set
    .build())
```

---

## Caveats and Design Considerations

### EDPS Architecture Trade-offs

**Server Model**: EDPS runs a persistent server process (FastAPI/Uvicorn + TinyDB). This adds complexity but enables:
- Smart reruns (job deduplication via bookkeeping)
- GUI support (edpsgui)
- Continuous operation mode

For batch processing, the server overhead may be unnecessary.

**Dependency Weight**: Full EDPS stack includes FastAPI, Uvicorn, TinyDB, networkx, etc. This is heavier than a minimal classification library would be.

**Abstraction Depth**: The path from workflow definition to recipe execution is:
```
Workflow -> Task -> DataSource -> ClassificationRule
    |
Generator -> JobGraph -> Job -> Action -> Command
    |
Executor -> Invoker -> Scheduler -> subprocess
```

This provides flexibility but can be hard to debug.

### Recipe Wrapping Considerations

**pycpl Dependency**: The recipe path requires pycpl (CPL Python bindings), which:
- Needs compilation against CPL C library
- May have version compatibility issues
- Adds installation complexity for non-ESO users

**Parameter Schema**: CPL recipes define parameter types, ranges, and defaults. This provides:
- Type validation at invocation
- `esorex --man-page` documentation
- Consistent parameter handling

**Standalone Usage**: Wrapped recipes can run outside EDPS:
```bash
esorex pyreduce_bias bias.sof
```

### Instrument Coverage

ESO instruments (UVES, HARPS, CRIRES, XSHOOTER, ESPRESSO) have well-defined header conventions. Non-ESO instruments may need:
- Custom adapter definitions
- Header normalization
- More flexible classification rules

---

## Alternative: `with_function()` Path

EDPS supports pure Python functions instead of recipes via `with_function()`. This provides a simpler path without pycpl/pyesorex dependencies.

### Key Difference

```python
# Recipe path (default)
bias_task = (task("bias")
    .with_recipe("pyreduce_bias")  # calls esorex
    .with_main_input(raw_bias)
    .build())

# Function path (alternative)
bias_task = (task("bias")
    .with_function("generic_echelle.functions.run_bias")  # direct Python call
    .with_main_input(raw_bias)
    .build())
```

### Function Implementation

```python
# generic_echelle/functions.py
from edps import RecipeInvocationArguments, RecipeInvocationResult, InvokerProvider
from edps.executor.renamer import ProductRenamer
from astropy.io import fits
from pathlib import Path

def run_bias(args: RecipeInvocationArguments,
             invoker_provider: InvokerProvider,
             renamer: ProductRenamer) -> RecipeInvocationResult:
    """Pure Python function that runs PyReduce's Bias step."""
    from pyreduce.reduce import Bias
    from pyreduce.instruments import instrument_info

    # Get input files from EDPS
    bias_files = [f.name for f in args.inputs.combined if f.category == "BIAS"]

    # Get parameters (from YAML or runtime overrides)
    params = args.parameters
    combine_method = params.get("combine_method", "median")
    sigma_clip = float(params.get("sigma_clip", "3.0"))

    # Detect instrument and run PyReduce
    instrument = instrument_info.detect(bias_files[0])
    step = Bias(instrument, method=combine_method, sigma=sigma_clip)
    master_bias, header = step.run(bias_files)

    # Write output FITS
    output_path = Path(args.job_dir) / "MASTER_BIAS.fits"
    hdu = fits.PrimaryHDU(master_bias, header=header)
    hdu.header["HIERARCH ESO PRO CATG"] = "MASTER_BIAS"
    hdu.writeto(output_path, overwrite=True)

    # Return result to EDPS
    from edps.client.FitsFile import FitsFile
    return RecipeInvocationResult(
        return_code=0,
        output_files=[FitsFile(name=str(output_path), category="MASTER_BIAS")]
    )
```

### Trade-offs

| Aspect | with_recipe() | with_function() |
|--------|---------------|-----------------|
| **Dependencies** | pycpl + pyesorex | None (pure Python) |
| **Standalone CLI** | `esorex recipe sof` works | Need separate CLI |
| **Parameter schema** | CPL validates types/ranges | Manual validation |
| **Documentation** | `esorex --man-page` | Docstrings only |
| **Installation** | More complex (CPL) | Simpler |
| **ESO compatibility** | Full | Partial |

### Recommendation

- **Use `with_recipe()`** if: ESO ecosystem integration matters, or you want standalone esorex usage
- **Use `with_function()`** if: simplicity is priority, or supporting non-ESO users who won't have pycpl

Both paths get the same EDPS orchestration benefits (classification, association, parallelization, smart reruns).

---

## Implementation Effort

| Component | Effort | Notes |
|-----------|--------|-------|
| Adapter layer | 2-3 days | 10-15 instrument definitions |
| Classification rules | 1-2 days | Generic functions using adapters |
| Data sources | 1-2 days | Standard EDPS patterns |
| Workflow definition | 1 day | Following existing examples |
| Recipe wrappers (5-6 steps) | 3-5 days | If using recipe path |
| Function wrappers (5-6 steps) | 2-3 days | If using function path |
| Testing with real data | 2-3 days | Classification edge cases |
| Instrument-specific workflows | 1-2 days per instrument | For multi-arm etc. |

**Total: ~2-3 weeks for core + 1 week per complex instrument**

---

## Usage

```bash
# Install
pip install edps pyesorex  # or just edps for function path

# Classify data
edps -w generic_echelle.workflow -i /data/raw/ -c

# Run full reduction
edps -w generic_echelle.workflow -i /data/raw/ -o /data/reduced/ -m science

# Run specific task
edps -w generic_echelle.workflow -i /data/raw/ -t wavecal -o /data/reduced/

# With custom parameters
edps -w generic_echelle.workflow -i /data/raw/ \
     -rp extract pyreduce.pyreduce_extract.method "sum" \
     -o /data/reduced/
```
