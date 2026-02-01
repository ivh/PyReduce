# Output File Formats

PyReduce produces FITS files containing extracted spectra. This page documents
the file structure.

## Spectra Format (v2)

The current format stores spectra in a FITS binary table with one row per trace.
Files are identified by header keyword `E_FMTVER = 2`.

### Header Keywords

| Keyword | Description |
|---------|-------------|
| `E_FMTVER` | Format version (2 for current format) |
| `E_STEPS` | Comma-separated list of pipeline steps run |
| `E_OSAMPLE` | Extraction oversampling factor |
| `E_LAMBDASF` | Slit function smoothing parameter |
| `E_LAMBDASP` | Spectrum smoothing parameter |
| `E_SWATHW` | Swath width (if set) |
| `barycorr` | Barycentric velocity correction (km/s) |

### Table Columns

The binary table extension (named `SPECTRA`) contains:

| Column | Format | Description |
|--------|--------|-------------|
| `SPEC` | `{ncol}E` | Extracted spectrum (float32). NaN for masked pixels. |
| `SIG` | `{ncol}E` | Uncertainty (float32). NaN for masked pixels. |
| `M` | `I` | Spectral order number (-1 if unknown) |
| `FIBER` | `16A` | Fiber identifier (string) |
| `EXTR_H` | `E` | Extraction height used for this trace |
| `WAVE` | `{ncol}D` | Wavelength in Angstroms (float64, optional) |
| `CONT` | `{ncol}E` | Continuum level (float32, optional) |
| `SLITFU` | `{len}E` | Slit function (float32, optional, NaN-padded) |

Each row corresponds to one extracted trace/order.

### Masking

Invalid pixels are marked with `NaN` in the `SPEC` and `SIG` columns. This
replaces the separate `COLUMNS` array used in the legacy format.

### Reading Spectra

```python
from pyreduce.spectra import Spectra

# Load spectra (handles both v2 and legacy formats)
spectra = Spectra.read("observation.science.fits")

# Access individual spectra
for s in spectra.data:
    print(f"Order {s.m}, fiber {s.fiber}")
    print(f"  Wavelength range: {s.wave[~s.mask].min():.1f} - {s.wave[~s.mask].max():.1f} A")

# Get stacked arrays
arrays = spectra.get_arrays()
spec_2d = arrays["spec"]  # shape (ntrace, ncol)
```

## Legacy Echelle Format (v1)

Files without `E_FMTVER` or with `E_FMTVER < 2` use the legacy format.

### Structure

The binary table has a single row containing flattened 2D arrays:

| Column | Format | Description |
|--------|--------|-------------|
| `SPEC` | `{ntrace*ncol}E` | Flattened spectrum array |
| `SIG` | `{ntrace*ncol}E` | Flattened uncertainty array |
| `WAVE` | `{ntrace*ncol}D` | Flattened wavelength array |
| `CONT` | `{ntrace*ncol}E` | Flattened continuum array |
| `COLUMNS` | `{ntrace*2}I` | Column range [start, end] per trace |

The `TDIM` keyword stores the original shape as `(ncol, ntrace)`.

### Key Differences from v2

| Aspect | Legacy (v1) | Current (v2) |
|--------|-------------|--------------|
| Table rows | 1 (flattened) | ntrace (one per spectrum) |
| Masking | Separate `COLUMNS` array | NaN in data |
| Order info | Not stored | `M` column |
| Fiber info | Not stored | `FIBER` column |
| Extraction height | Not stored | `EXTR_H` column |
| Slit function | Separate files | `SLITFU` column |

### Reading Legacy Files

`Spectra.read()` automatically detects and handles legacy files:

```python
from pyreduce.spectra import Spectra

# Works for both formats - auto-detects via E_FMTVER header
spectra = Spectra.read("old_file.fits")

# Access data the same way regardless of original format
for s in spectra.data:
    print(f"Order {s.m}: {len(s.spec)} pixels")
```

## Migration

To convert legacy files to the new format:

```python
from pyreduce.migration import convert_echelle_to_spectra

# Convert single file
spectra = convert_echelle_to_spectra("old.fits", "new.fits")

# Batch convert a directory
from pyreduce.migration import migrate_directory
results = migrate_directory("/path/to/data", dry_run=False)
```
