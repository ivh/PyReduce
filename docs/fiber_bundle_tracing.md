# Fiber Bundle Configuration for Multi-Fiber Instruments

## Overview

PyReduce supports multi-fiber instruments where each spectral order contains multiple physical fibers. The `fibers` section in `config.yaml` defines how these fibers are organized into groups and which traces to use for each reduction step.

## Terminology

- **Trace**: Polynomial describing one fiber's path across detector
- **Fiber**: Physical fiber in bundle (numbered 1-N)
- **Group**: Named collection of fibers (e.g., "A", "cal", "B")
- **Bundle**: Repeating pattern of fibers (e.g., 7 fibers per IFU target)

## Configuration

### Named Groups (AJ-style)

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

## Merge Methods

- `average` - Fit polynomial to mean y-positions of all fibers in group
- `center` - Select the middle fiber's trace
- `[i]` or `[i, j, ...]` - Select specific 1-based indices within group

## Per-Step Trace Selection

The `use` section specifies which traces each reduction step receives:

- `all` - All individual fiber traces (ignores grouping)
- `groups` - All merged group/bundle traces
- `[A, B]` - Specific named groups only

Steps not listed in `use` default to `groups` when groups/bundles are defined.

## Output Format

Order tracing saves both raw and grouped traces to the `.ord_default.npz` file:

```
orders          - Raw traces (n_fibers, degree+1)
column_range    - Raw column ranges (n_fibers, 2)
group_names     - List of group names
group_A_traces  - Merged traces for group A
group_A_cr      - Column ranges for group A
group_A_count   - Number of physical fibers in group A
...
```

## Example Instruments

### AJ (75 fibers)

Simulated echelle with science fibers A/B and calibration fiber:
- Fibers 1-35: Science fiber A
- Fibers 37-39: Calibration fiber
- Fibers 40-75: Science fiber B

### MOSAIC (630 fibers)

ELT multi-object spectrograph with 90 IFU targets:
- 7 fibers per target bundle
- Extract center fiber from each bundle for reduction
