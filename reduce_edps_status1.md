# EDPS Integration Status - Session 1

## What Was Built

### Directory Structure
```
PyReduce/
├── edps_wkf/                    # EDPS workflow definitions
│   ├── __init__.py
│   ├── parameters_config.yaml   # Points EDPS to parameter files
│   ├── reduce_classification.py # Classification rules (CRIRES+ specific)
│   ├── reduce_datasources.py    # Data sources with time-based matching
│   ├── reduce_keywords.py       # Header keyword definitions
│   ├── reduce_parameters.yaml   # Recipe parameters
│   └── reduce_wkf.py            # Main workflow (bias + flat tasks)
├── recipes/                     # PyRecipe wrappers for pyesorex
│   ├── __init__.py
│   └── reduce_bias.py           # Working bias recipe
├── tools/
│   └── pyesorex-wrapper.sh      # Wrapper to run pyesorex via uv
└── pyproject.toml               # Added edps optional dependencies
```

### Changes to Existing Code
- `pyreduce/instruments/crires_plus.py`: Added `get_arm_from_header()` method to derive arm string from FITS header

### pyproject.toml Additions
```toml
[project.optional-dependencies]
edps = ["edps>=1.6.0", "pycpl>=1.0.3", "pyesorex>=1.0.3"]

[tool.uv.sources]
pycpl = { index = "pycpl" }

[[tool.uv.index]]
name = "pycpl"
url = "https://ivh.github.io/pycpl/simple/"

[[tool.uv.index]]
name = "eso"
url = "https://ftp.eso.org/pub/dfs/pipelines/libraries/"
```

## EDPS Configuration (~/.edps/application.properties)

Key settings that need to be configured:
```ini
workflow_dir=...,/Users/tom/PyReduce/edps_wkf
esorex_path=/Users/tom/PyReduce/tools/pyesorex-wrapper.sh
parameters_config_file=/Users/tom/PyReduce/edps_wkf/parameters_config.yaml
```

## What Works

1. **Classification**: EDPS correctly classifies CRIRES+ files as BIAS (DARK), FLAT based on `ESO DPR TYPE` header
2. **Bias recipe**: Successfully combines bias frames using PyReduce's `combine_bias()`
3. **Arm detection**: `get_arm_from_header()` correctly determines arm from CRIRES+ headers (e.g., "J1228_Open_det1")
4. **EDPS orchestration**: Jobs are scheduled, executed, and tracked

### Test Run Results
- 8 of 9 bias jobs completed successfully
- 1 failed due to mixing previously-reduced file with raw data (classification issue)

## Lessons Learned

### EDPS Architecture
1. **Parameter discovery**: EDPS needs explicit `parameters_config.yaml` mapping workflow names to parameter files
2. **Recipe discovery**: Set `PYESOREX_PLUGIN_DIR` env var to point to recipes directory
3. **Workflow naming**: Workflow module path becomes the workflow name (e.g., `edps_wkf.reduce_wkf`)
4. **Classification functions**: Receive EDPS `FitsFile` object, use `f[key]` not `f.get(key)`

### pyesorex Integration
1. pyesorex needs to run in the PyReduce venv to access pyreduce modules
2. Wrapper script with `uv run pyesorex` solves this
3. Recipe outputs go to pyesorex's cwd (set by wrapper)

### Instrument Handling
1. Header `INSTRUME` value may differ from PyReduce module name (CRIRES vs crires_plus)
2. Recipe needs mapping dict or instrument should provide this
3. `arm` parameter is required for CRIRES+ - must be derived from headers

## Known Issues / TODOs

1. **Output location**: Recipe outputs to wrapper's cwd, not job directory
2. **Reduced file classification**: Previously-reduced files get classified as raw BIAS
3. **Instrument mapping**: `CRIRES` -> `crires_plus` mapping is in recipe, should be in config
4. **WAVE/SCIENCE classification**: Not yet connected to tasks (rules exist but unused)
5. **Flat recipe**: Referenced in workflow but not implemented

## Usage

```bash
# Install with EDPS support
uv sync --extra edps

# Set env var for recipe discovery
export PYESOREX_PLUGIN_DIR=/path/to/PyReduce/recipes

# Classify data
uv run edps -w edps_wkf.reduce_wkf -i /path/to/data -c

# Run bias reduction
uv run edps -w edps_wkf.reduce_wkf -i /path/to/data -t bias -o /tmp/out

# Reset workflow state
uv run edps -w edps_wkf.reduce_wkf -r
```
