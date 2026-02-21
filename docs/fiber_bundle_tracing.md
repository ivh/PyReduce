# Fiber Bundle Configuration for Multi-Fiber Instruments

## Overview

PyReduce supports multi-fiber instruments where each spectral order contains multiple physical fibers. The `fibers` section in `config.yaml` defines how these fibers are organized into groups and which traces to use for each reduction step.

## Terminology

- **Trace**: Polynomial describing one fiber's path across detector
- **Fiber**: Physical fiber in bundle (numbered 1-N within each order/group)
- **Group**: Named collection of fibers with semantic meaning (e.g., "A", "cal", "B"). Use groups when fibers have distinct roles - science targets, calibration sources, sky fibers, etc.
- **Bundle**: Numbered groups of identical fibers (e.g., "bundle_1", "bundle_2", ...). Use bundles for repeating patterns like IFU spaxels where each bundle has the same structure.
- **Spectral Order**: Wavelength range; multi-order instruments have same fiber pattern repeated across orders
- **fiber_idx**: Index of a fiber within its group/order (1-indexed). Used for per-fiber wavelength calibration.

In short: **groups** are for named, semantically distinct fiber sets; **bundles** are for numbered, structurally identical fiber sets.

Each `Trace` object has:
- `m`: Spectral order number (physical diffraction order)
- `group`: Group/bundle identifier ("A", "B", "cal", or bundle index)
- `fiber_idx`: Fiber index within the group (1-indexed), or None if merged

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

For echelle instruments where fiber groups repeat across spectral orders (e.g., ANDES_YJH with 75 fibers per order across 18 orders):

```yaml
fibers:
  fibers_per_order: 75              # Enables per-order organization
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

Setting `fibers_per_order` enables per-order organization (no separate `per_order: true` needed).

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
  fibers_per_order: 3
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
  fibers_per_order: [75, 75, 60]  # Can vary per channel
  order_centers_file: [uvb_centers.yaml, vis_centers.yaml, nir_centers.yaml]

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
- `per_fiber` - Traces grouped by `fiber_idx` for per-fiber processing

Steps not listed in `use` default to `groups` when groups/bundles are defined.

### Per-Fiber Wavelength Calibration

For 2D wavelength calibration (`dimensionality: "2D"`), all traces in a fit must share the same optical path. When multiple fibers exist per order:

```yaml
fibers:
  use:
    wavecal: [A]        # One 2D fit for group A across all orders
    # OR
    wavecal: [A, B]     # Separate 2D fits for A and B
    # OR
    wavecal: per_fiber  # Separate 2D fit per fiber_idx
```

With `per_fiber`, each unique `fiber_idx` gets its own 2D polynomial fit. This is useful when fibers have slightly different wavelength solutions and you want to calibrate each independently.

## Output Format

Tracing saves all traces to a FITS binary table (`.traces.fits`) with one row per trace:

| Column | Type | Description |
|--------|------|-------------|
| M | int16 | Spectral order number (-1 if N/A) |
| GROUP | 16A | Group/bundle identifier ('A', 'B', 'cal', etc.) |
| FIBER_IDX | int16 | Fiber index within group (1-indexed, -1 if N/A) |
| POS | float64[deg+1] | Position polynomial coefficients |
| COL_RANGE | int32[2] | Valid x range [start, end) |
| HEIGHT | float32 | Extraction aperture height in pixels |
| SLIT | float64[...] | Curvature coefficients (if available) |
| SLITDELTA | float32[...] | Per-row curvature residuals (if available) |
| WAVE | float64[...] | Wavelength polynomial (if available) |

The `GROUP` column identifies which group/bundle each trace belongs to. The `FIBER_IDX` column identifies the fiber within the group (1-indexed). For merged groups, multiple raw fibers become a single trace with the group name and `FIBER_IDX = -1`.

### Automatic Extraction Heights

The `heights` array stores per-trace extraction heights computed from trace geometry:
- For middle traces: half the distance between neighboring traces
- For edge traces: distance to the single neighbor
- Measured at multiple reference columns; maximum is used

These heights are used automatically when `extraction_height` is set to `null` in settings.json. This provides optimal per-trace apertures without manual tuning.

For groups/bundles, heights are derived from fiber spacing within each group (span + fiber diameter).

## Example Instruments

### ANDES_UBV / ANDES_RIZ (66 fibers per order)

Simulated echelle spectrographs for visible wavelengths:
- Fibers 1-31: Slit A
- Fibers 33-35: Calibration
- Fibers 36-66: Slit B
- ANDES_UBV: channels U, B, V (selected by `BAND` header)
- ANDES_RIZ: channels R, R1, R2, IZ (selected by `HDFMODEL` header, since R variants all have `BAND=R`)

### ANDES_YJH (75 fibers per order)

Simulated NIR echelle with science fibers A/B, calibration, and IFU:
- Fibers 1-35: Slit A
- Fibers 37-39: Calibration
- Fibers 40-75: Slit B
- Additional groups: ifu, ring0-4 (subsets of the fiber bundle)
- Channels Y, J, H (selected by `BAND` header)

### MOSAIC (630 fibers, single order)

ELT multi-object spectrograph with 90 IFU targets:
- 7 fibers per target bundle
- Extract center fiber from each bundle for reduction
- Uses `bundles` pattern with `bundle_centers_file` to handle broken fibers
