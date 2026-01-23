# Manual Step Execution

While `Pipeline.from_instrument()` provides a convenient way to run reductions,
sometimes you need more control over the process. The manual approach lets you:

- Access and inspect intermediate results between steps
- Modify data arrays before passing them to subsequent steps
- Reorder or skip steps as needed
- Debug issues by examining individual step outputs
- Experiment when reducing data from a new instrument

## Overview

Each reduction step is implemented as a class in `pyreduce.reduce`. You can
instantiate these classes directly and call their `run()` method with the
required inputs.

```python
from pyreduce.reduce import Bias, Flat, Trace, ...

# Create step instance
bias_step = Bias(instrument, channel, target, night, output_dir, trace_range, **config)

# Run the step
bias_result = bias_step.run(bias_files, mask)

# Use the result in subsequent steps
flat_step = Flat(instrument, channel, target, night, output_dir, trace_range, **config)
flat_result = flat_step.run(flat_files, bias_result, mask)
```

## Complete Example

This example shows how to run a full UVES reduction step by step:

```python
import os
from os.path import join

from pyreduce import datasets, util
from pyreduce.configuration import load_config
from pyreduce.instruments.instrument_info import load_instrument
from pyreduce.reduce import (
    Bias,
    ContinuumNormalization,
    Finalize,
    Flat,
    Mask,
    NormalizeFlatField,
    Trace,
    ScienceExtraction,
    SlitCurvatureDetermination,
    WavelengthCalibrationFinalize,
    WavelengthCalibrationInitialize,
    WavelengthCalibrationMaster,
)

# Parameters
instrument_name = "UVES"
target = "HD132205"
night = "2010-04-01"
channel = "middle"
trace_range = (1, 21)
plot = 1

# Paths
base_dir = datasets.UVES()
input_dir = join(base_dir, "raw/")
output_dir = join(base_dir, f"reduced/{night}/{channel}")
os.makedirs(output_dir, exist_ok=True)

# Load instrument and configuration
instrument = load_instrument(instrument_name)
config = load_config(None, instrument_name, 0)

# Common arguments for all steps
step_args = (instrument, channel, target, night, output_dir, trace_range)

# Find and classify files
file_groups = instrument.sort_files(
    input_dir,
    target,
    night,
    channel=channel,
    **config["instrument"],
)
settings, files = file_groups[0]

# Extract file lists
bias_files = files.get("bias", [])
flat_files = files.get("flat", [])
trace_files = files.get("orders", flat_files)
curvature_files = files.get("curvature", files.get("wavecal_master", []))
wavecal_files = files.get("wavecal_master", [])
science_files = files.get("science", [])
```

### Running Each Step

```python
def step_config(name):
    """Get step config with plot level override."""
    cfg = config.get(name, {}).copy()
    cfg["plot"] = plot
    return cfg

# Step 1: Load bad pixel mask
mask_step = Mask(*step_args, **step_config("mask"))
mask = mask_step.run()

# Step 2: Create master bias
bias_step = Bias(*step_args, **step_config("bias"))
bias = bias_step.run(bias_files, mask)

# Step 3: Create master flat
flat_step = Flat(*step_args, **step_config("flat"))
flat = flat_step.run(flat_files, bias, mask)

# Step 4: Trace
trace_step = Trace(*step_args, **step_config("trace"))
traces = trace_step.run(trace_files, mask, bias)

# Step 5: Determine slit curvature
curvature_step = SlitCurvatureDetermination(*step_args, **step_config("curvature"))
curvature = curvature_step.run(curvature_files, traces, mask, bias)

# Step 6: Normalize flat field
norm_flat_step = NormalizeFlatField(*step_args, **step_config("norm_flat"))
scatter = None  # Optional background scatter
norm_flat = norm_flat_step.run(flat, traces, scatter, curvature)

# Step 7: Wavelength calibration (three sub-steps)
wavecal_master_step = WavelengthCalibrationMaster(*step_args, **step_config("wavecal_master"))
wavecal_master = wavecal_master_step.run(
    wavecal_files, traces, mask, curvature, bias, norm_flat
)

wavecal_init_step = WavelengthCalibrationInitialize(*step_args, **step_config("wavecal_init"))
wavecal_init = wavecal_init_step.load(config, wavecal_master)

wavecal_step = WavelengthCalibrationFinalize(*step_args, **step_config("wavecal"))
wavecal = wavecal_step.run(wavecal_master, wavecal_init)
wave, coef, linelist = wavecal

# Step 8: Extract science spectra
science_step = ScienceExtraction(*step_args, **step_config("science"))
science = science_step.run(
    science_files, bias, traces, norm_flat, curvature, scatter, mask
)

# Step 9: Continuum normalization
continuum_step = ContinuumNormalization(*step_args, **step_config("continuum"))
continuum = continuum_step.run(science, wave, norm_flat)

# Step 10: Write final output
finalize_step = Finalize(*step_args, **step_config("finalize"))
finalize_step.run(continuum, wave, config)
```

## Step Dependencies

Each step requires outputs from previous steps. Here's the dependency graph:

| Step | Inputs |
|------|--------|
| `Mask` | (none) |
| `Bias` | files, mask |
| `Flat` | files, bias, mask |
| `Trace` | files, mask, bias |
| `SlitCurvatureDetermination` | files, trace, mask, bias |
| `NormalizeFlatField` | flat, trace, scatter, curvature |
| `WavelengthCalibrationMaster` | files, trace, mask, curvature, bias, norm_flat |
| `WavelengthCalibrationInitialize` | config, wavecal_master |
| `WavelengthCalibrationFinalize` | wavecal_master, wavecal_init |
| `ScienceExtraction` | files, bias, trace, norm_flat, curvature, scatter, mask |
| `ContinuumNormalization` | science, wave, norm_flat |
| `Finalize` | continuum, wave, config |

## Inspecting Intermediate Results

The main advantage of manual execution is access to intermediate data:

```python
# After tracing
traces, column_range = trace_step.run(trace_files, mask, bias)

# Inspect traces array
print(f"Found {len(traces)} traces")
print(f"Trace polynomial coefficients shape: {traces.shape}")
print(f"Column range: {column_range}")

# Modify traces if needed (e.g., exclude problematic traces)
traces = traces[2:-2]  # Skip first and last 2 traces

# Continue with modified traces
curvature = curvature_step.run(curvature_files, traces, mask, bias)
```

## Loading Previous Results

Each step can save and load its results:

```python
# Run a step and save
traces = trace_step.run(trace_files, mask, bias)
# Results are automatically saved to output_dir

# Later, load without re-running
traces = trace_step.load()
```

## See Also

- [Pipeline API](examples.md) - For automated reductions
- [CLI](cli.md) - Command-line interface
- [Configuration](configuration_file.md) - Settings reference
