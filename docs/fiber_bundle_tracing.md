# Fiber Bundle Configuration for Multi-Fiber Instruments

## Overview

PyReduce supports multi-fiber instruments where each spectral order contains multiple physical fibers. The `fibers` section in `config.yaml` defines how these fibers are organized into groups and which traces to use for each reduction step.

## Terminology

- **Trace**: Polynomial describing one fiber's path across detector
- **Fiber**: Physical fiber in bundle (numbered 1-N)
- **Group**: Named collection of fibers (e.g., "A", "cal", "B")
- **Bundle**: Repeating pattern of fibers (e.g., 7 fibers per IFU target)
- **Spectral Order**: Wavelength range; multi-order instruments have same fiber pattern repeated across orders

## Configuration

### Named Groups (single-order instruments)

For instruments with explicitly defined fiber groups:

```yaml
fibers:
  groups:
    A:
      range: [1, 36]      # Fibers 1-35 (1-based, half-open interval)
      merge: average      # Average all fiber traces into one
    cal:
      range: [37, 40]     # Fibers 37-39
      merge: average
    B:
      range: [40, 76]     # Fibers 40-75
      merge: average

  use:
    science: [A, B]       # Science extraction uses A and B groups
    wavecal: [cal]        # Wavelength calibration uses cal fiber
    norm_flat: all        # Flat normalization uses all individual fibers
```

### Bundle Pattern (MOSAIC-style)

For instruments with repeating fiber bundles:

```yaml
fibers:
  bundles:
    size: 7               # 7 fibers per bundle
    count: 90             # 90 bundles total (optional validation)
    merge: center         # Select middle fiber from each bundle

  use:
    curvature: groups     # Use all 90 bundle centers
    science: groups
```

### Handling Missing/Broken Fibers

When some fibers are broken or missing, the trace count won't be divisible by bundle size. Use `bundle_centers_file` to assign traces to bundles by proximity rather than fixed division:

```yaml
fibers:
  bundles:
    size: 7
    bundle_centers_file: bundle_centers.yaml  # y-position of each bundle center
    merge: center

  use:
    curvature: groups
    science: groups
```

The `bundle_centers.yaml` file maps bundle IDs to y-positions at detector center:

```yaml
# bundle_centers.yaml
1: 3975.0
2: 3932.4
3: 3889.8
# ... etc for all 90 bundles
```

Each detected trace is assigned to its nearest bundle center. When merging:

- **merge: center** with all fibers present: picks middle index (e.g., fiber 4 of 7)
- **merge: center** with missing fibers: picks trace closest to bundle_center
- **merge: average**: averages all present fibers (no extrapolation for missing)

This approach handles arbitrary patterns of missing fibers without requiring the trace count to be divisible by bundle size.

### Per-Order Grouping (echelle multi-fiber)

For echelle instruments where fiber groups repeat across spectral orders (e.g., AJ with 75 fibers per order across 18 orders):

```yaml
fibers:
  per_order: true
  fibers_per_order: 75              # Expected fibers per order (validation)
  order_centers_file: order_centers.yaml  # Or inline with order_centers:

  groups:
    A:
      range: [1, 36]
      merge: average
    cal:
      range: [37, 40]
      merge: average
    B:
      range: [40, 76]
      merge: average

  use:
    science: [A, B]
    wavecal: [cal]
    norm_flat: all
```

The `order_centers.yaml` file maps spectral order numbers to y-positions at detector center:

```yaml
# order_centers.yaml
90: 3868.1
91: 3609.0
92: 3356.4
# ... etc
```

Or inline for instruments with few orders:

```yaml
fibers:
  per_order: true
  order_centers:
    1: 150.5
    2: 320.3
    3: 490.1
```

### Multi-Channel Instruments

For instruments with multiple channels (detectors/arms), per-order fields can be lists indexed by channel:

```yaml
channels: [UVB, VIS, NIR]

fibers:
  per_order: true
  order_centers_file: [uvb_centers.yaml, vis_centers.yaml, nir_centers.yaml]
  fibers_per_order: [75, 75, 60]  # Can vary per channel

  groups:  # Same structure across all channels
    A: {range: [1, 36], merge: average}
    B: {range: [40, 76], merge: average}
```

If fiber arrangements differ fundamentally between channels, use separate instrument configs.

## Merge Methods

- `average` - Fit polynomial to mean y-positions of all fibers in group
- `center` - Select the middle fiber's trace
- `[i]` or `[i, j, ...]` - Select specific 1-based indices within group

## Per-Step Trace Selection

The `use` section specifies which traces each reduction step receives:

- `all` - All individual fiber traces (ignores grouping)
- `groups` - All merged group/bundle traces stacked
- `[A, B]` - Specific named groups (kept separate in output)

Steps not listed in `use` default to `groups` when groups/bundles are defined.

## Output Format

Order tracing saves both raw and grouped traces to the `.ord_default.npz` file.

For per_order=False:
```
orders          - Raw traces (n_fibers, degree+1)
column_range    - Raw column ranges (n_fibers, 2)
group_A_traces  - Merged traces for group A
group_A_cr      - Column ranges for group A
```

For per_order=True:
```
orders          - Raw traces (n_total_fibers, degree+1)
column_range    - Raw column ranges
A_order_90      - Merged trace for group A in order 90
A_cr_90         - Column range for group A in order 90
...
```

## Example Instruments

### AJ (75 fibers x 18 orders)

Simulated echelle with science fibers A/B and calibration fiber:
- Fibers 1-35: Science fiber A
- Fibers 37-39: Calibration fiber
- Fibers 40-75: Science fiber B
- Uses `per_order: true` with `order_centers_file`

### MOSAIC (630 fibers, single order)

ELT multi-object spectrograph with 90 IFU targets:
- 7 fibers per target bundle
- Extract center fiber from each bundle for reduction
- Uses `bundles` pattern with `bundle_centers_file` to handle broken fibers
