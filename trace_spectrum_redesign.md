# Trace and Spectrum Redesign

Issue #34 and multi-fiber support.

## Summary of Changes

| Current | New |
|---------|-----|
| `order`, `nord`, `iord` | `trace`, `ntrace`, `itrace` (where it means trace index) |
| `Echelle` class | `Spectra` class (contains `list[Spectrum]`) |
| Parallel arrays (traces, column_range, ...) | `Trace` dataclass |
| COLUMNS + MASK in FITS | NaN for masked pixels |
| Lost order/fiber identity | Preserved via `m` and `fiber` fields (`m` can be None) |
| No extraction metadata | `ExtractionParams` in header, per-trace height in column |
| Spectrum normalized on load | Stored un-normalized, `cont` separate, `.normalized()` method |
| Three separate files (traces.npz, curve.npz, thar.npz) | Single `traces.fits` with all calibration data |
| NPZ for traces | FITS binary table for traces (consistency with spectrum files) |
| No format versioning | `E_FMTVER` header keyword for backwards compat detection |
| No provenance tracking | `E_STEPS` header keyword lists pipeline steps run |
| Separate initial/refined wavecal | Single `wave` field, refined in place if wavecal runs |
| Slitdeltas in curvature tuple | `slitdelta` field on Trace (per-row correction, shape matches height) |

---

## Part 1: The Mask/Column Range Problem

### What is `column_range`?

A compact representation of valid pixel ranges per trace:

```python
column_range : array of shape (ntrace, 2)
    For each trace: [start_column, end_column] of valid signal
```

### Current Problem

1. **Redundancy**: Both `COLUMNS` and `MASK` saved to FITS (10% file bloat)
2. **Mask overwrites**: On load, saved mask is regenerated from columns anyway
3. **Mask is limited**: Only encodes column range, not bad pixels within valid region

### Solution: NaN Masking

Use NaN for invalid pixels (FITS de facto standard):

```python
# Save
spec_data = np.ma.filled(spec, np.nan)

# Load
spec = np.ma.masked_invalid(spec_data)
```

**Benefits:**
- No COLUMNS needed in output files
- Works with any FITS reader
- Specutils and other tools expect NaN
- Supports future bad pixel propagation

**Migration:**
- New files: NaN only
- Loading old files: If COLUMNS present, generate mask from it

---

## Part 2: Multi-Fiber Support

### Current Problem: Lost Metadata

```python
# Trace file has order/fiber info:
group_traces = {
    'A': {87: coeffs, 88: coeffs},  # order number is dict key
    'B': {87: coeffs, 88: coeffs},
}

# But extraction flattens to arrays, discarding it:
traces, _ = _stack_per_order_traces(...)  # orders DISCARDED
selected_traces = np.vstack(all_traces)   # flat array

# Output file has no idea which row is which:
SPEC: (6, 9232)   # which is order 87? which is fiber A?
```

### Solution: Trace Dataclass

Bundle all trace metadata together:

```python
@dataclass
class Trace:
    # Identity
    m: int | None                   # spectral order number (None if unknown, e.g. MOSAIC)
    fiber: str | int                # 'A', 'B', 'cal' or bundle index

    # Geometry
    pos: np.ndarray                 # y(x) trace position polynomial, shape (deg+1,)
    column_range: tuple[int, int]   # valid x range
    height: float | None = None     # extraction aperture
    slit: np.ndarray | None = None  # slit curvature, shape (deg_y+1, deg_x+1)
                                    # x_offset = P(y) where coeffs vary with x
    slitdelta: np.ndarray | None = None  # per-row slit correction, shape (height,)

    # Calibration
    wave: np.ndarray | None = None  # wavelength(x) polynomial, shape (deg+1,)

    def slit_at_x(self, x: float) -> np.ndarray:
        """Evaluate slit polynomial coefficients at position x."""
        if self.slit is None:
            return None
        return np.array([np.polyval(c, x) for c in self.slit])
```

**Naming rationale:**
- `m` not `order` - avoids confusion with sort order
- `fiber` is `str | int` - string for named groups ('A', 'B'), int for bundles
- `pos`, `wave`, `slit` - short, consistent polynomial field names
- `slit` under geometry - describes physical detector geometry

**Pipeline fills incrementally:**
```
Trace step      → pos, column_range, m, fiber, height  → save traces.fits
Load init guess → wave (from instrument's wavecal_*.npz) → update traces.fits
Curvature step  → slit                                 → update traces.fits
Wavecal step    → refine wave (optional)               → update traces.fits
```

This consolidates what was previously three separate files (`*.traces.npz`, `*.curve.npz`, `*.thar.npz`) into a single `*.traces.fits`.

**Wavecal workflow**: The initial wavelength guess is loaded into `Trace.wave` early. If the wavecal step runs, it refines this in place. If wavecal is skipped, extraction still has approximate wavelengths. No separate "initial" vs "refined" tracking - check `E_STEPS` to see if wavecal was run.

---

## Part 3: Spectrum Output

### Spectrum Dataclass

Output of extraction for one trace:

```python
@dataclass
class Spectrum:
    # Identity (copied from Trace)
    m: int | None                   # spectral order number (None if unknown)
    fiber: str | int                # fiber identifier

    # Extracted data (NaN for masked pixels)
    spec: np.ndarray                # flux, un-normalized
    sig: np.ndarray                 # uncertainty

    # Optional data
    wave: np.ndarray | None = None  # wavelength (evaluated from poly)
    cont: np.ndarray | None = None  # continuum (full array, not polynomial)
    slitfu: np.ndarray | None = None  # slit function (shape depends on osample)

    # Per-trace extraction param (can vary by fiber group)
    extraction_height: float | None = None

    @classmethod
    def from_trace(cls, trace: Trace, spec, sig, **kwargs) -> "Spectrum":
        """Factory method ensures identity copied from Trace."""
        return cls(m=trace.m, fiber=trace.fiber, spec=spec, sig=sig, **kwargs)

    def normalized(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (spec/cont, sig/cont)."""
        if self.cont is None:
            raise ValueError("No continuum available")
        return self.spec / self.cont, self.sig / self.cont
```

**Notes:**
- `spec` stored un-normalized; `cont` stored separately for flexibility
- `cont` is full array (not polynomial) - computed via iterative smoothing, not poly fit
- `slitfu` shape varies with `osample` - length is `extraction_height * osample + 1`
- `extraction_height` stored per-trace since it can vary by fiber group

### ExtractionParams Dataclass

Global extraction parameters (same for all traces in a file):

```python
@dataclass
class ExtractionParams:
    osample: int                    # oversampling factor
    lambda_sf: float                # slitfunction smoothing
    lambda_sp: float                # spectrum smoothing
    swath_width: int | None = None  # swath width for extraction

    def to_header(self, header: fits.Header):
        """Write params to FITS header."""
        header["E_OSAMPLE"] = (self.osample, "Extraction oversampling")
        header["E_LAMBDASF"] = (self.lambda_sf, "Slitfunction smoothing")
        header["E_LAMBDASP"] = (self.lambda_sp, "Spectrum smoothing")
        if self.swath_width is not None:
            header["E_SWATHW"] = (self.swath_width, "Swath width")

    @classmethod
    def from_header(cls, header: fits.Header) -> "ExtractionParams":
        """Read params from FITS header."""
        return cls(
            osample=header.get("E_OSAMPLE"),
            lambda_sf=header.get("E_LAMBDASF"),
            lambda_sp=header.get("E_LAMBDASP"),
            swath_width=header.get("E_SWATHW"),
        )
```

### Spectra Container

Replaces `Echelle` class:

```python
@dataclass
class Spectra:
    header: fits.Header
    data: list[Spectrum]
    params: ExtractionParams | None = None  # global params, stored in header

    @staticmethod
    def read(fname) -> "Spectra": ...

    def save(self, fname): ...

    @property
    def ntrace(self) -> int:
        return len(self.data)

    def select(self, m=None, fiber=None) -> list[Spectrum]:
        """Filter spectra by order and/or fiber."""
        result = self.data
        if m is not None:
            result = [s for s in result if s.m == m]
        if fiber is not None:
            result = [s for s in result if s.fiber == fiber]
        return result
```

**Naming rationale:**
- `Spectra` not `Echelle` - describes what it is, not the instrument type
- `Spectrum` (singular) for one trace's data
- Mirrors `Trace` (input) → `Spectrum` (output)

### Connection: Trace → Spectrum

Each Spectrum comes from exactly one Trace. The connection is:
1. **Identity**: `(m, fiber)` copied from Trace to Spectrum
2. **Factory method**: `Spectrum.from_trace()` ensures correct copying
3. **Implicit link**: match by `(m, fiber)` when needed

```python
# Extraction creates Spectrum from Trace
for t in traces:
    spec, sig, slitfu, cr = extract(t.pos, t.column_range, t.slit, ...)
    result = Spectrum.from_trace(t, spec, sig, slitfu=slitfu,
                                  extraction_height=actual_height)
```

---

## Part 4: Terminology Renaming

Where "order" currently means "trace index" (not spectral order), rename:

| Current | New | Context |
|---------|-----|---------|
| `nord` | `ntrace` | Number of traces |
| `iord` | `itrace` | Loop variable |
| `for iord in range(nord)` | `for itrace in range(ntrace)` | Iteration |
| `orders` (array of traces) | `traces` | Variable name |

**Keep unchanged:**
- `m` field in Trace/Spectrum - this IS the spectral order number
- Header keywords like `OBASE` - external interface

---

## Part 5: File Formats

All files use FITS for consistency and inspectability with standard tools.

**Format version**: All files include header keyword `E_FMTVER = 2` to distinguish from old format.

**Pipeline steps**: Each file records which steps were run: `E_STEPS = 'bias,flat,science'`

### Trace File (*.traces.fits)

FITS binary table, one row per trace. Replaces separate `*.traces.npz`, `*.curve.npz`, and `*.thar.npz` files.

**Binary table columns**:

| Column | Type | Description |
|--------|------|-------------|
| M | int16 | Spectral order number (-1 if unknown) |
| FIBER | str/int | Fiber identifier |
| POS | float64[deg+1] | y(x) position polynomial |
| COL_RANGE | int32[2] | Valid x range (start, end) |
| HEIGHT | float32 | Extraction height |
| SLIT | float64[deg_y+1, deg_x+1] | Slit curvature coefficients |
| SLITDELTA | float32[height] | Per-row slit correction (optional) |
| WAVE | float64[deg+1] | Wavelength polynomial |

Variable-length arrays (VLA) used where polynomial degree varies.

### Spectrum Files (*.science.fits, *.final.fits)

**Header keywords** (all use `E_` prefix for PyReduce-specific keys):

| Keyword | Type | Description |
|---------|------|-------------|
| E_FMTVER | int | Format version (2 = new format) |
| E_STEPS | str | Pipeline steps run (e.g. 'bias,flat,science') |
| E_OSAMPLE | int | Oversampling factor |
| E_LAMBDASF | float | Slitfunction smoothing |
| E_LAMBDASP | float | Spectrum smoothing |
| E_SWATHW | int | Swath width (optional) |

**Binary table columns**:

| Column | Shape | Type | Description |
|--------|-------|------|-------------|
| SPEC | (ntrace, ncol) | float32 | Flux, NaN for masked |
| SIG | (ntrace, ncol) | float32 | Uncertainty, NaN for masked |
| M | (ntrace,) | int16 | Spectral order number (-1 if unknown) |
| FIBER | (ntrace,) | str/int | Fiber identifier |
| EXTR_H | (ntrace,) | float32 | Extraction height used (per-trace) |
| SLITFU | (ntrace, nslitfu) | float32 | Slit function (science.fits only) |
| WAVE | (ntrace, ncol) | float64 | Wavelength (final.fits only) |
| CONT | (ntrace, ncol) | float32 | Continuum (final.fits only) |

No COLUMNS. No MASK. Masking implicit in NaN.

---

## Part 6: Extraction Flow

### Current (loses metadata)

```python
for i in range(nord):
    spec[i] = extract(traces[i], column_range[i], ...)
# Which order/fiber is spec[3]? Unknown.

echelle.save(fname, spec=spec, sig=sig, columns=column_range)
```

### New (preserves metadata)

```python
# Extraction with full metadata
params = ExtractionParams(osample=10, lambda_sf=1.0, lambda_sp=0.0, swath_width=300)

results = []
for t in traces:  # list[Trace]
    # Get actual height (per-trace default or override from settings)
    height = settings_height or t.height

    spec, sig, slitfu, cr = extract(
        img, t.pos, t.column_range, t.slit,
        extraction_height=height, osample=params.osample, ...
    )

    results.append(Spectrum.from_trace(
        t, spec, sig,
        slitfu=slitfu,
        extraction_height=height,
    ))

spectra = Spectra(header=head, data=results, params=params)
spectra.save(fname)  # saves global params to header, per-trace to columns
```

### Later: Adding Wavelength and Continuum

```python
# After wavecal step - add wavelength
for spectrum, trace in zip(spectra.data, traces):
    spectrum.wave = np.polyval(trace.wave, x)  # evaluate polynomial

# After continuum step - add continuum
for spectrum, cont in zip(spectra.data, continua):
    spectrum.cont = cont

spectra.save(fname)  # now includes WAVE and CONT columns
```

---

## Part 7: Migration / Backwards Compatibility

### Format Detection

```python
def detect_format(header):
    fmtver = header.get("E_FMTVER", 1)
    if fmtver >= 2:
        return "new"
    return "old"
```

### Reading Old Files

```python
def read(fname):
    # ... load data ...

    if detect_format(header) == "old":
        # Old format: has COLUMNS, no M/FIBER
        mask = columns_to_mask(data["columns"])
        # Assume sequential order numbers, single fiber
        m = np.arange(ntrace)
        fiber = np.zeros(ntrace, dtype=int)
    else:
        # New format: has M/FIBER, NaN masking
        mask = np.isnan(data["spec"])
        m = data["m"]
        fiber = data["fiber"]
```

### Reading Old Trace Files

```python
def read_traces(fname):
    if fname.endswith(".npz"):
        # Old format - convert
        data = np.load(fname)
        # Also check for separate curve.npz and thar.npz
        ...
    else:
        # New format - FITS binary table
        ...
```

### Writing New Files

Always write new format (E_FMTVER=2, E_STEPS, NaN, M, FIBER). Old readers that expect COLUMNS will fail - this is a breaking change but necessary for multi-fiber support.

---

## Part 8: Implementation Order

1. **Trace dataclass** - define in `models.py` or `trace_model.py`
2. **Trace FITS I/O** - save/load list[Trace] to FITS binary table
3. **Update trace.py** - use Trace dataclass, preserve m/fiber
4. **Update curvature step** - write slit coeffs to Trace, no separate curve.npz
5. **Update wavecal step** - write wave coeffs to Trace, no separate thar.npz
6. **Spectrum/Spectra classes** - define in `spectra.py`
7. **Update reduce.py** - extraction uses Trace, outputs Spectrum
8. **Rename echelle.py → spectra.py** - with new save/load
9. **Update tools/combine.py** - use new Spectra format
10. **Terminology rename** - nord→ntrace, iord→itrace throughout
11. **Update tests**
12. **Migration helpers** - functions to convert old files to new format
