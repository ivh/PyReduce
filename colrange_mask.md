# Column Range and Mask: Current State and Options

## What is `column_range`?

A compact representation of valid pixel ranges per order:

```python
column_range : array of shape (ntrace, 2)
    For each order: [start_column, end_column] of valid signal
```

Example for 5 orders on a 4096-pixel detector:
```python
column_range = [
    [  50, 4000],  # order 0: columns 50-3999 have signal
    [  30, 4050],  # order 1
    [  10, 4080],  # order 2
    [   0, 4090],  # order 3
    [   0, 4096],  # order 4
]
```

## Where `column_range` is Stored

| File | Contents | Purpose |
|------|----------|---------|
| `trace.npz` | Original from tracing | Authoritative source |
| `.science.fits` | Possibly modified by `fix_column_range()` | Self-contained output |
| `.final.fits` | Same as science | Self-contained output |

## How `column_range` Flows Through the Pipeline

```
Trace step
    ↓
mark_orders() determines column_range from where traces have signal
    ↓
Saved to trace.npz (authoritative source)
    ↓
Extraction step
    ↓
Trace.load() → column_range from trace.npz
    ↓
fix_column_range() clips based on extraction_height
    (ensures aperture stays within image bounds)
    ↓
Modified column_range saved to .science.fits
    ↓
Continuum/Finalize steps
    ↓
Pass through to .final.fits
```

**Key point:** Extraction always reads `column_range` from `trace.npz`, never from echelle files.

## The Mask

The `mask` is just the expanded 2D boolean version of `column_range`:

```python
mask = np.full((ntrace, ncol), True)   # all masked
for i in range(ntrace):
    mask[i, column_range[i,0]:column_range[i,1]] = False  # unmask valid range
```

## Current Problems (Issue #34)

### Problem 1: Mask Generated on Load, Ignored if Present

In `echelle.py` lines 194-209:
```python
if "columns" in ech:
    ech["mask"] = np.full((nord, ncol), True)  # Creates fresh mask
    for iord in range(nord):
        ech["mask"][iord, ech["columns"][iord, 0]:ech["columns"][iord, 1]] = False
```

If both `MASK` and `COLUMNS` exist in the file, the saved `MASK` is loaded but immediately **overwritten** with a freshly generated mask from `COLUMNS`.

### Problem 2: Round-Trip Saves Both

```python
ech = Echelle.read("file.fits")  # Generates mask from columns, stores in _data
ech.save("file_copy.fits")       # Saves **self._data including both mask AND columns
```

Result: ~10% file size growth from redundant mask storage.

### Current Workaround

`tools/combine.py` manually handles this:
```python
e.mask = (snew == 0) | (cnew == 0)
del e["columns"]   # Remove columns to avoid saving both
e.save(output)
```

## Options to Resolve

### Option A: Stop Saving `columns` in Echelle Files

**Rationale:**
- `column_range` is authoritative in `trace.npz`
- Extraction never reads it from echelle files
- The mask (or masked arrays) already encode valid pixels

**Changes:**
1. Remove `columns=column_range` from `ScienceExtraction.save()` and `Finalize.save()`
2. Downstream consumers use mask instead of columns

**Pros:** Simpler, no redundancy
**Cons:** Breaking change for anyone parsing `COLUMNS` from output files

### Option B: Stop Saving/Generating `mask`, Keep `columns`

**Rationale:**
- `columns` is more compact
- Can derive mask when needed

**Changes:**
1. In `Echelle.read()`: generate mask from columns but don't store in `_data`
2. In `Echelle.save()`: don't save mask, only columns
3. Masked arrays use the generated mask without persisting it

**Pros:** Smaller files, backwards compatible
**Cons:** Still have columns in two places (trace.npz and echelle)

### Option C: Save Only What's Needed Per File Type

**Rationale:** Different files have different purposes

| File | Save | Rationale |
|------|------|-----------|
| `trace.npz` | `column_range` | Authoritative source |
| `.science.fits` | Neither | Intermediate file, mask implicit in masked arrays |
| `.final.fits` | `columns` | User-facing, compact representation for consumers |

**Changes:**
1. Science extraction: don't save columns
2. Finalize: save columns only
3. On load: if columns present, generate mask; if mask present, use it

**Pros:** Clean separation of concerns
**Cons:** More complex logic

### Option D: Explicit Mask Priority

**Changes in `Echelle.read()`:**
```python
if "mask" in ech:
    # Use saved mask directly
    pass
elif "columns" in ech:
    # Generate mask from columns
    ech["mask"] = ...
```

**Changes in `Echelle.save()`:**
```python
# If mask can be represented as column ranges, save only columns
# Otherwise save mask
if mask_is_simple_column_ranges(self.mask):
    save columns, not mask
else:
    save mask, not columns
```

**Pros:** Handles both simple (column-based) and complex (arbitrary) masks
**Cons:** Most complex to implement

## Echelle File Structure

Echelle files store **full detector width** arrays:

```
SPEC: shape=(26, 4096)  # full width, not trimmed
SIG:  shape=(26, 4096)
WAVE: shape=(26, 4096)
COLUMNS: shape=(26, 2)  # which part of each row is valid
```

Pixels outside `column_range` contain zeros or garbage and must be ignored.

## Current Mask Limitations

**The mask currently only encodes column range, NOT bad pixels within the valid region.**

During extraction:
- Outlier rejection happens in slit decomposition (`cwrappers.py` lines 250-254)
- But rejected pixels are **not propagated** to the output mask
- The only mask modifications are edge trimming due to curvature

```python
# This is ALL the mask does currently:
mask[: xrange[0]] = True   # outside left edge
mask[xrange[1] :] = True   # outside right edge
```

**Implication:** The current mask is 100% redundant with `columns`. However, bad pixel propagation would be useful - it's rare that all pixels contributing to a spectral bin are bad, but it can happen.

## FITS Conventions for Masked Data

The standard approach for storing masked arrays in FITS is **NaN for masked values**:

| Approach | Pros | Cons |
|----------|------|------|
| **NaN in data** | Simple, widely understood, single array | Loses original bad pixel values |
| Separate MASK extension | Preserves original values | More complex, larger files |
| Bit mask (DQ array) | Rich info (why pixels are bad) | Overkill for simple cases |

**NaN is the de facto standard:**
- Astropy uses NaN for float columns by default when writing masked tables
- Specutils expects NaN for masked flux values
- MPDAF replaces masked data with NaN when saving

```python
# Save
spec_data = np.ma.filled(spec, np.nan)

# Load
spec = np.ma.masked_invalid(spec_data)
```

Astropy has a mode that saves both data+mask as separate columns, but the docs warn it "goes outside of the established FITS standard" and may confuse non-astropy readers.

## Updated Recommendation

**Use NaN for masked pixels, drop `columns` from echelle files:**

1. **During extraction:** Propagate bad pixel info to the output mask (future enhancement)
2. **When saving:** Use `np.ma.filled(spec, np.nan)` - mask becomes implicit in data
3. **When loading:** Use `np.ma.masked_invalid()` to recover mask
4. **Remove `COLUMNS`** from echelle files - no longer needed

**Benefits:**
- No redundancy between mask and columns
- Works with any FITS reader
- Downstream tools (specutils, etc.) handle NaN naturally
- Supports future bad pixel propagation without format changes

**Migration:**
- New files: NaN only, no COLUMNS
- Loading old files: If COLUMNS present, generate mask from it (backwards compat)
