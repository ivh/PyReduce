# Instrument Architecture Redesign

## Motivation

The original instrument configuration system had several issues:

- **JSON/Python split unclear**: ~50% of instruments override methods like `sort_files()`, `get_extension()`. Logic was scattered.
- **Mode-indexed lists fragile**: Properties like `extension`, `orientation`, `gain` were parallel lists that had to stay in sync.
- **No validation**: Config errors only discovered at runtime deep in the pipeline.
- **No unified API**: Users had to understand `reduce.main()` with many parameters.

## Completed Changes

### 1. YAML Instrument Configs
All instrument definitions moved from JSON to YAML format. JSON files deleted.
- `pyreduce/instruments/*.yaml` - One file per instrument
- Pydantic validation via `InstrumentConfig` model in `models.py`

### 2. Pydantic Validation
- `pyreduce/instruments/models.py` - Validates instrument config at load time
- Type-safe access via `self.config.field` instead of `self.info["field"]`
- Default values defined in model, not scattered in code

### 3. Pipeline API
New fluent API in `pyreduce/pipeline.py`:

```python
from pyreduce.pipeline import Pipeline

# Recommended: auto-discover files
result = Pipeline.from_instrument(
    instrument="UVES",
    target="HD132205",
    night="2010-04-01",
    arm="middle",
    steps=("bias", "flat", "orders", "science"),
).run()

# Or with explicit files
result = (
    Pipeline("UVES", output_dir)
    .bias(bias_files)
    .flat(flat_files)
    .trace_orders()
    .extract(science_files)
    .run()
)
```

### 4. Click CLI
New command-line interface in `pyreduce/__main__.py`:

```bash
# Run full pipeline
uv run reduce run UVES HD132205 --steps bias,flat,orders,science

# Run individual steps
uv run reduce step bias UVES HD132205
uv run reduce step flat UVES HD132205

# List available steps
uv run reduce list-steps

# Download sample data
uv run reduce download UVES
```

### 5. Mode → Arm Terminology
Renamed "mode" to "arm" throughout codebase for clarity:
- `mode` parameter → `arm` parameter
- `modes` config field → `arms` config field

### 6. Deprecation of reduce.main()
`pyreduce.reduce.main()` now shows deprecation warning pointing to `Pipeline.from_instrument()`.

## Architecture Summary

```
CLI (reduce command)     →  pyreduce/__main__.py (Click)
     ↓
Pipeline.from_instrument →  pyreduce/pipeline.py
     ↓
Step classes             →  pyreduce/reduce.py
     ↓
Core algorithms          →  pyreduce/extract.py, trace_orders.py, etc.
     ↓
C extensions (CFFI)      →  pyreduce/clib/*.c
```

### Configuration Separation

- **Instrument configs** (`instruments/*.yaml`): What the instrument IS - detector, headers, arms
- **Reduction settings** (`settings/*.json`): HOW to reduce - polynomial degrees, thresholds, iterations

---

## Future Work

### V2 Hardware Model (Not Implemented)

The original design proposed explicit modeling of:
- Detectors with amplifier regions
- Optical paths (fibers, beam-splitters)
- Dimension system for mode explosion (CRIRES+ 261 combinations)

This was deferred because:
- Current flat YAML structure works for all existing instruments
- Would require rewriting all YAML files
- Can be added incrementally when a concrete need arises

V2 Pydantic models are defined in `models.py` but not used.

### CharSlit Extraction Backend

Replace CFFI extraction with nanobind-based CharSlit:
- Better build system (scikit-build-core)
- Unified API for vertical/curved extraction
- Richer output (model image, diagnostics)

Integration phases:
1. Add as optional backend
2. Validate results match CFFI
3. Switch default
4. Remove CFFI

### Other Ideas

- Hierarchical FITS output for multi-fiber instruments
- Config file for full reduction (instrument + files + steps in one YAML)
- Per-amplifier calibration properties
