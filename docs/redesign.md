# Architecture Changes (v0.7+)

This document explains the architectural changes in PyReduce v0.7 and the rationale behind them.

## Summary

PyReduce v0.7 introduced two major interface changes:

1. **Trace dataclass** - Replaces parallel arrays with a single object containing all trace metadata
2. **Spectra class** - Replaces the old Echelle class with a cleaner per-trace data model

These changes improve code clarity, enable multi-fiber support, and consolidate calibration data into fewer files.

---

## Part 1: The Trace Dataclass

### Problem: Parallel Arrays

Previously, trace data was scattered across multiple parallel arrays:

```python
# Old interface - easy to get out of sync
traces = np.array([...])        # shape (ntrace, degree+1) - position polynomials
column_range = np.array([...])  # shape (ntrace, 2) - valid x range
curvature = (coeffs, deltas)    # separate tuple from curvature step
wave_coef = np.array([...])     # separate array from wavecal step
```

This caused problems:
- Arrays could get out of sync (different lengths, wrong alignment)
- Curvature had to be passed as a separate parameter to every extraction step
- No place to store per-trace metadata (order number, fiber ID)
- Three separate files: `*.traces.npz`, `*.curve.npz`, `*.thar.npz`

### Solution: Trace Dataclass

All trace data is now bundled in a single dataclass:

```python
@dataclass
class Trace:
    # Identity
    m: int | None           # spectral order number (None if unknown)
    fiber: str | int        # fiber identifier ('A', 'B', 0, 1, etc.)

    # Geometry
    pos: np.ndarray         # y(x) position polynomial coefficients
    column_range: tuple[int, int]  # valid x range (start, end)
    height: float | None    # extraction aperture height

    # Calibration (filled by later steps)
    slit: np.ndarray | None      # curvature polynomial coefficients
    slitdelta: np.ndarray | None # per-row curvature residuals
    wave: np.ndarray | None      # wavelength polynomial coefficients

    def wlen(self, x: np.ndarray) -> np.ndarray:
        """Evaluate wavelength at column positions."""
        return np.polyval(self.wave, x)
```

### Data Flow

Pipeline steps now work with `list[Trace]`:

```
Trace step      → creates list[Trace] with pos, column_range, m, fiber
Curvature step  → updates traces in-place with slit, slitdelta
Wavecal step    → updates traces in-place with wave coefficients
Extraction      → reads all needed data from traces
```

### File Format

All trace data is stored in a single FITS binary table (`*.traces.fits`):

| Column | Type | Description |
|--------|------|-------------|
| M | int16 | Spectral order number (-1 if unknown) |
| GROUP | 16A | Group identifier ('A', 'B', 'cal', etc.) |
| FIBER_IDX | int16 | Fiber index within group (1-indexed, -1 if N/A) |
| POS | float64[deg+1] | Position polynomial coefficients |
| COL_RANGE | int32[2] | Valid x range |
| HEIGHT | float32 | Extraction height |
| SLIT | float64[...] | Curvature coefficients (variable length) |
| SLITDELTA | float32[...] | Per-row curvature residuals |
| WAVE | float64[deg+1] | Wavelength polynomial coefficients |

Legacy `.npz` files and old FITS files with `FIBER` column are still readable.

---

## Part 2: The Spectra Class

### Problem: Lost Metadata

The old `Echelle` class stored spectra as flattened 2D arrays:

```python
# Old format - which row is which order/fiber?
spec = np.array([...])  # shape (ntrace, ncol) - but no labels!
```

After extraction, there was no way to know which array row corresponded to which spectral order or fiber.

### Solution: Spectrum and Spectra Classes

Each extracted spectrum is now a `Spectrum` object with full metadata:

```python
@dataclass
class Spectrum:
    # Identity (copied from source Trace)
    m: int | None
    fiber: str | int

    # Extracted data
    spec: np.ndarray        # flux (NaN for masked pixels)
    sig: np.ndarray         # uncertainty

    # Optional
    wave: np.ndarray | None # wavelength array
    cont: np.ndarray | None # continuum level
    slitfu: np.ndarray | None  # slit function

@dataclass
class Spectra:
    header: fits.Header
    data: list[Spectrum]

    @staticmethod
    def read(fname) -> "Spectra": ...
    def save(self, fname): ...
```

### File Format (v2/v3)

Spectra files use a FITS binary table with one row per trace:

| Column | Format | Description |
|--------|--------|-------------|
| SPEC | float32[ncol] | Flux (NaN for masked) |
| SIG | float32[ncol] | Uncertainty |
| M | int16 | Spectral order number |
| GROUP | 16A | Group identifier |
| FIBER_IDX | int16 | Fiber index within group (v3) |
| EXTR_H | float32 | Extraction height used |
| WAVE | float64[ncol] | Wavelength (optional) |
| CONT | float32[ncol] | Continuum (optional) |
| SLITFU | float32[len] | Slit function (optional) |

Header keyword `E_FMTVER` identifies the format version (2 or 3).

### Masking

Invalid pixels use NaN instead of a separate mask array:

```python
# Old: separate COLUMNS array defined valid range
# New: NaN in SPEC and SIG columns

# Reading (both formats supported)
spectra = Spectra.read("file.fits")  # auto-detects format
```

---

## Part 3: Naming Conventions

### Terminology Changes

| Old | New | Reason |
|-----|-----|--------|
| `nord` | `ntrace` | "order" conflated spectral order (m) with trace index |
| `iord` | `idx` | Same reason |
| `orders` (array) | `traces` or `trace` | Clarity |

The field `m` on Trace/Spectrum is the actual spectral order number.

### Variable Names in Code

| Name | Type | Meaning |
|------|------|---------|
| `trace` | `list[Trace]` | Step run() parameter |
| `Trace.wave` | `np.ndarray` | Wavelength polynomial coefficients |
| `Trace.wlen(x)` | method | Evaluate wavelength at positions |
| `wlen` | `np.ndarray` | Evaluated wavelength array |

---

## Part 4: Removed Code

### Deleted Files

| File | Replacement |
|------|-------------|
| `echelle.py` | `spectra.py` (Spectra class) |

### Deleted Functions

| Function | Replacement |
|----------|-------------|
| `traces_to_arrays()` | Use `list[Trace]` directly |

### Simplified Step Signatures

The `curvature` parameter was removed from extraction step signatures:

```python
# Old
def run(self, files, bias, trace, norm_flat, curvature, scatter, mask): ...

# New - curvature data is in trace[i].slit
def run(self, files, bias, trace, norm_flat, scatter, mask): ...
```

---

## Part 5: Migration

### Reading Old Files

All readers auto-detect format:

```python
from pyreduce.trace_model import load_traces
from pyreduce.spectra import Spectra

# Works for both old and new formats
traces, header = load_traces("file.traces.fits")  # or .npz
spectra = Spectra.read("file.science.fits")
```

---

## Part 6: Benefits

1. **Single source of truth** - All trace data in one object, one file
2. **No sync errors** - Can't have mismatched array lengths
3. **Multi-fiber ready** - Each trace knows its fiber ID
4. **Simpler APIs** - Steps don't need curvature parameter
5. **Self-documenting** - Output files contain order/fiber labels
6. **Standard masking** - NaN works with any FITS reader

---

## Future Work

The following features from the original design are not yet implemented:

### Multi-Detector Model

Explicit `Detector` and `Amplifier` classes for instruments with:
- Multiple readout amplifiers (different gains per quadrant)
- Multiple detectors (CRIRES+ det1/2/3, XSHOOTER UVB/VIS/NIR)

Currently handled via the `channels` config parameter.

### Dimension System

Declarative config for instruments with mode explosion:
- CRIRES+: 29 bands × 3 deckers × 3 detectors
- Currently uses mode string parsing

### Fiber Bundle Extraction

For many-fiber instruments (60+ fiber pseudo-slit):
- Grouped extraction (sum fiber subsets)
- Slit-range extraction (arbitrary apertures)

Currently each fiber is extracted individually.
