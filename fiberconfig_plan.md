# Plan: Fiber Grouping Configuration for Multi-Fiber Instruments

**Supersedes:** `docs/fiber_bundle_tracing.md` (partial implementation in settings.json)

## Goal
Extend `config.yaml` to declaratively specify fiber grouping and extraction selection, eliminating manual work in example scripts like `aj_example.py` and `mosaic_example.py`.

## Terminology
- **Order**: Spectral order (echelle order)
- **Trace**: Polynomial describing one fiber's path across detector
- **Fiber**: Physical fiber in bundle (numbered 1-N)
- **Group**: Named collection of fibers (e.g., "A", "cal", "B")
- **Bundle**: Repeating pattern of fibers (e.g., 7 fibers per target)

## Current State
Both examples perform manual fiber organization after tracing:
- **AJ**: Groups fibers into `A: [1,36), cal: [37,40), B: [40,76)`, extracts average of each group
- **MOSAIC**: Groups 630 fibers into 90 bundles of 7, extracts center fiber from each bundle

## Proposed Config Structure

### Option A: Explicit Named Groups (AJ-style)
```yaml
fibers:
  groups:
    A:
      range: [1, 36]      # 1-based, [start, end) half-open: fibers 1-35
      merge: average      # average all fiber traces into one group trace
    cal:
      range: [37, 40]     # fibers 37-39
      merge: average
    B:
      range: [40, 76]     # fibers 40-75
      merge: average

  # Per-step trace selection: which group traces to use
  use:
    curvature: [A]        # just fiber A for curvature
    norm_flat: all        # all individual fibers (not grouped)
    science: [A, B]       # science uses A and B, not cal
    wavecal: [cal]        # wavelength cal uses cal fiber
```

### Option B: Bundle Pattern (MOSAIC-style)
```yaml
fibers:
  bundles:
    size: 7               # fibers per bundle
    count: 90             # number of bundles (optional, validated if given)
    merge: center         # use middle fiber trace for each bundle
    # OR: merge: average  # average all 7 into one trace
    # OR: merge: [3]      # specific index within bundle (1-based)

  use:
    curvature: groups     # all 90 bundle traces
    science: groups
```

### Merge methods
- `average` - fit polynomial to mean y-positions of all fibers in group
- `center` - select the middle fiber's trace
- `[i]` or `[i, j, ...]` - select specific fiber index(es) within group (1-based)

### `use` section
- `all` - use raw individual fiber traces (ignores grouping)
- `groups` - use all merged group traces
- `[A, B]` - use specific named groups only

## Implementation Plan

### 1. Add `FibersConfig` model to `models.py`
```python
class FiberGroupConfig(BaseModel):
    range: tuple[int, int]  # [start, end) half-open, 1-based
    merge: str | list[int] = "center"  # "average", "center", or [indices]

class FiberBundleConfig(BaseModel):
    size: int               # fibers per bundle
    count: int | None = None  # validated if provided
    merge: str | list[int] = "center"

# Type for use section values
TraceSelection = Literal["all", "groups"] | list[str]

class FibersConfig(BaseModel):
    groups: dict[str, FiberGroupConfig] | None = None
    bundles: FiberBundleConfig | None = None
    use: dict[str, TraceSelection] | None = None  # step_name -> selection
```

Add to `InstrumentConfig`:
```python
fibers: FibersConfig | None = None
```

### 2. Add fiber organization functions to `trace.py`

New function `organize_fibers()`:
```python
def organize_fibers(traces, column_range, fibers_config):
    """Organize traced fibers into groups according to config.

    Parameters
    ----------
    traces : ndarray (n_fibers, degree+1)
    column_range : ndarray (n_fibers, 2)
    fibers_config : FibersConfig

    Returns
    -------
    group_traces : dict[str, ndarray]
        {group_name: traces} - merged trace(s) per group
    group_column_range : dict[str, ndarray]
    group_fiber_counts : dict[str, int]
        Number of physical fibers in each group (for extraction height calc)
    """
```

Merge logic (applied per group or bundle):
- `merge: average` → fit polynomial to mean y-positions of all fibers
- `merge: center` → select middle fiber's trace
- `merge: [i, j, ...]` → select specific fiber indices (1-based within group)

New function `select_traces_for_step()`:
```python
def select_traces_for_step(raw_traces, raw_cr, group_traces, group_cr,
                           fibers_config, step_name):
    """Select which traces to use for a given step.

    Looks up fibers_config.use[step_name] to determine selection.
    Returns (traces, column_range) for the step to use.
    """
```

### 3. Modify `OrderTracing` step in `reduce.py`

The step should:
1. Run standard tracing (unchanged)
2. If `fibers` config present, call `organize_fibers()`
3. Save both raw traces (`orders`, `column_range`) and group traces (`group_*`)
4. Return a new structure that includes both

### 4. Modify steps to use `select_traces_for_step()`

Each step that uses traces should call `select_traces_for_step()` to get the appropriate trace set based on `fibers.use`:
- `ScienceExtraction`
- `SlitCurvatureDetermination`
- `NormalizeFlatField`
- `WavelengthCalibrationMaster`

If `fibers.use` doesn't specify the step, fall back to:
- `groups` if groups/bundles are defined
- `all` otherwise

### 5. Update example scripts

After implementation, `mosaic_example.py` would become:
```python
# Config handles fiber selection - no manual grouping needed
result = Pipeline.from_instrument(
    instrument="MOSAIC",
    channel="NIR",
    steps=("trace", "curvature", "science"),
).run()
```

## Files to Modify

| File | Changes |
|------|---------|
| `pyreduce/instruments/models.py` | Add `FibersConfig`, `FiberGroupConfig`, `FiberBundleConfig` |
| `pyreduce/trace.py` | Add `organize_fibers()`, `select_traces_for_step()` |
| `pyreduce/reduce.py` | Modify `OrderTracing`, `ScienceExtraction`, `SlitCurvatureDetermination`, `NormalizeFlatField` to use fiber config |
| `pyreduce/instruments/AJ/config.yaml` | Add `fibers:` section with groups |
| `pyreduce/instruments/MOSAIC/config.yaml` | Add `fibers:` section with bundles |
| `docs/fiber_bundle_tracing.md` | Update to reflect new design (remove settings.json approach) |

## Design Decisions

1. **Indexing**: 1-based fiber indices (matches astronomer convention, AJ example)
2. **Output**: Preserve both raw traces AND organized/selected traces in output files
3. **Bundles**: Require exact division - fail if `n_traces % bundle_size != 0`

## Verification

1. Run `aj_example.py` with new config - should produce same output with less code
2. Run `mosaic_example.py` with new config - should auto-select center fibers
3. Unit tests for `organize_fibers()` with both group and bundle patterns
