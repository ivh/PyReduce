# Fiber Bundle Tracing for Multi-Fiber Instruments

## Status: Work in Progress

This document describes the fiber bundle tracing feature for instruments with many physical fibers per spectral order (e.g., 75 fibers reformatted as a pseudo-slit).

## Implemented

### New Functions in `pyreduce/trace_orders.py`

1. **`merge_traces()`** - Merges traces from even/odd illuminated flats, assigns to spectral orders based on `order_centers` config, and assigns fiber IDs within each order.

2. **`group_and_refit()`** - Groups physical fiber traces into logical fibers (A, B, cal) and refits polynomials by averaging y-positions across member fibers.

### Extended `OrderTracing` in `pyreduce/reduce.py`

- Added `fiber_bundle` config parameter
- Added `_run_fiber_bundle()` method for multi-step tracing
- Added `save_fiber_bundle()` for extended output format

### Configuration in `pyreduce/settings/settings_aj.json`

```json
{
  "orders": {
    "fiber_bundle": {
      "count": 75,
      "order_centers": [377, 1217, 1442, ...],  // y-position at x=ncols/2
      "illumination_sets": {
        "even": {"pattern": "0::2"},
        "odd": {"pattern": "1::2"}
      },
      "logical_fibers": {
        "A": {"range": [0, 36]},
        "cal": {"range": [36, 38]},
        "B": {"range": [38, 75]}
      }
    }
  }
}
```

### Instrument Config in `pyreduce/instruments/aj.yaml`

Added file classification for even/odd flats:
```yaml
kw_flat_even: FILENAME
kw_flat_odd: FILENAME
id_flat_even: ".*even.*"
id_flat_odd: ".*odd.*"
```

## Remaining Work

### 1. Trace Detection Parameters

Current cluster-based detection doesn't work well for thin fiber traces:
- With min_cluster=100, only ~30-40 traces detected per flat (expected ~400+)
- Traces are fragmenting into small clusters
- Per-order distribution is uneven

**Options to explore:**
- Use smaller min_cluster (50 or less)
- Row-by-row peak finding instead of cluster detection
- Trace seeding from known fiber positions

### 2. Output Format Testing

The extended npz format saves:
- `traces_order_N` - physical traces per order
- `traces_A`, `traces_B`, `traces_cal` - logical fiber traces
- `orders` - backward compat alias for first logical fiber

Need to verify extraction code works with new format.

### 3. CLI Integration

Currently requires manual file specification. Need to:
- Wire up file classification for even/odd flats
- Pass `files_even`/`files_odd` through pipeline

### 4. Extraction Support

Modify extraction to:
- Accept fiber name parameter (A, B, cal)
- Load correct trace from extended npz format

## Test Data

Located at `~/REDUCE_DATA/AJ/raw/`:
- `J_FF_even_1s.fits` - 37 even-numbered fibers illuminated
- `J_FF_odd_1s.fits` - 38 odd-numbered fibers illuminated

Image: 4096x4096, 13 spectral orders, 75 fibers per order (simulated pyechelle data).

## Architecture Notes

### Terminology
- **Order**: Spectral order (echelle order) - 13 on detector
- **Trace**: Single fiber path across detector within an order
- **Fiber**: Physical fiber in the bundle (0-74)
- **Logical fiber**: Group of physical fibers (A, B, cal)

### Workflow
```
Even flat (37 fibers) ─┐
                       ├─> Assign to orders ─> Merge ─> Group & refit ─> A, B, cal
Odd flat (38 fibers)  ─┘
```

1. Trace each flat separately
2. Assign traces to spectral orders via `order_centers`
3. Merge even/odd traces within each order
4. Group by fiber range, average positions, refit polynomial
