# Instrument Architecture Redesign

## Problem Statement

The current instrument configuration system conflates several distinct concepts into a flat structure with mode-indexed lists and ad-hoc Python overrides. This creates problems when instruments have:

- Multiple fibers per order (HARPS A/B, NEID)
- Beam-splitters duplicating fiber images
- Multiple detectors with different properties (CRIRES+ det1/2/3)
- Multiple amplifiers with different gains per detector
- Mode explosion (CRIRES+: 29 bands × 3 deckers × 3 detectors = 261 combinations)
- Unclear boundary between JSON config and Python class logic

## Current Architecture Issues

### 1. Mode-Indexed Lists Are Fragile
```json
"extension": [1, 2],
"orientation": [1, 0],
"gain": ["ESO DET OUT1 CONAD", "ESO DET OUT2 CONAD"]
```
The alignment between these lists is implicit. Adding a mode requires updating multiple lists in lockstep.

### 2. Orthogonal Dimensions Conflated
HARPS needs to track: mode (BLUE/RED) × fiber (A/B/AB) × polarization
CRIRES+ needs: band × decker × detector

These are independent axes but get flattened into a single "mode" concept.

### 3. No Explicit Detector/Amplifier Model
Multi-amplifier readouts (different gains/readnoise per quadrant) are handled via template strings like `"ESO DET OUT{id[0]} CONAD"` - no clear model of what amplifiers exist.

### 4. Fibers and Traces Are Implicit
No explicit representation of:
- How many fibers feed the spectrograph
- How many traces each fiber produces (beam-splitter = 2)
- How traces are arranged on the detector (interleaved vs separate)

### 5. Python Overrides Are Opaque
~50% of instruments override `sort_files()`, `get_extension()`, or `add_header_info()`. The logic is scattered and hard to follow.

---

## Proposed Architecture

### Core Principle: Explicit Component Hierarchy

```
Instrument
├── Detectors[]
│   ├── name, dimensions, orientation
│   └── Amplifiers[]
│       └── gain, readnoise, region
├── OpticalPaths[] (fibers, beam arms)
│   ├── name
│   └── traces_per_order (1 normally, 2 with beam-splitter)
├── Modes[] (operational configurations)
│   └── detector_mapping, wavelength_range
└── Dimensions (what varies and how)
```

### 1. Hardware Model (Static)

**Key distinction: Detector vs Amplifier**
- **Multi-amplifier detector**: Multiple readout regions, but after gain/bias correction → single unified image. Amplifiers just differ in calibration properties.
- **Multi-detector instrument**: Each detector is a separate image with its own traces, wavelength coverage, orientation. Must be processed as independent reduction units (detectors never perfectly aligned).

```python
@dataclass
class Amplifier:
    """Readout amplifier - a region of a detector with its own calibration"""
    id: str
    region: Region  # Which pixels this amp reads
    gain: float | HeaderRef
    readnoise: float | HeaderRef
    linearity: LinearityModel | None  # Per-amp non-linearity correction
    bad_pixel_mask: Path | None  # Per-amp bad pixels (or None if shared)

@dataclass
class Detector:
    """Physical detector - after calibration, becomes one unified image"""
    name: str
    naxis: tuple[int, int]
    orientation: int  # 0-7 rotation/flip code
    prescan: Region | None
    overscan: Region | None
    amplifiers: list[Amplifier]
    bad_pixel_mask: Path | None  # Detector-wide bad pixels (in addition to per-amp)

@dataclass
class BeamArm:
    """One arm of a beam-splitter or polarimeter"""
    name: str  # "ordinary", "extraordinary", "beam_upper", etc.
    polarization: str | None  # "O", "E", "circular_L", "circular_R", etc.
    wavelength_shift: float = 0.0  # If beam-splitter introduces wavelength offset
    trace_offset: float = 0.0  # Pixel offset from nominal trace position

@dataclass
class OpticalPath:
    """A light path through the instrument (fiber, slit position, etc.)"""
    name: str  # "fiber_a", "fiber_b", "slit_center", etc.
    beam_arms: list[BeamArm] | None  # None = no beam-splitter, just 1 trace
    # If beam_arms is set, each arm produces its own trace per order
```

**Processing hierarchy:**
```
Instrument
├── Detector 1 (→ single image after amp calibration)
│   ├── Amplifier A (gain, readnoise, linearity, bad pixels)
│   ├── Amplifier B
│   └── [unified image with optical paths below]
├── Detector 2 (→ separate image, own traces)
│   └── ...
└── OpticalPaths (which detectors they illuminate)
    ├── fiber_a → illuminates detector 1
    ├── fiber_b → illuminates detector 1
    └── ...
```

For multi-detector instruments (CRIRES+ with det1/2/3, XSHOOTER with UVB/VIS/NIR):
- Each detector is a separate reduction unit
- Loop over detectors, each has own trace solution, wavelength solution
- Results combined at the end (but images never merged)

### 2. Dimension System (Replaces Mode-Indexed Lists)

Instead of implicit list indexing, declare dimensions explicitly:

```yaml
dimensions:
  mode:
    values: [BLUE, RED]
    header_key: "ESO INS MODE"
  fiber:
    values: [A, B, AB]
    header_key: "ESO INS FIBER"
    optional: true  # Not all exposures use all fibers

# Properties declare what they vary with
properties:
  detector:
    varies_by: [mode]
    mapping:
      BLUE: detector_blue
      RED: detector_red

  wavelength_range:
    varies_by: [mode]
    mapping:
      BLUE: [380, 530]
      RED: [530, 680]
```

For CRIRES+ with mode explosion:
```yaml
dimensions:
  band:
    values: [Y1029, J1232, ..., M4187]
    header_key: "ESO INS WLEN ID"
  decker:
    values: [Open, pos1, pos2]
    header_key: "ESO INS SLIT NAME"
  detector:
    values: [det1, det2, det3]
    header_key: "ESO DET CHIP NAME"

# Mode is computed from dimensions, not enumerated
mode_pattern: "{band}_{decker}_{detector}"
```

### 3. Trace Configuration

Explicitly model how spectra appear on detector:

```yaml
# Simple single-fiber instrument (e.g., UVES)
optical_paths:
  - name: science
    beam_arms: null  # No beam-splitter, 1 trace per order

---
# HARPS dual-fiber
optical_paths:
  - name: fiber_a
    beam_arms: null
  - name: fiber_b
    beam_arms: null
trace_arrangement:
  layout: interleaved  # A and B alternate on detector
  separation: 15.5  # Pixels between A and B traces

---
# Polarimetric instrument with beam-splitter
optical_paths:
  - name: science_fiber
    beam_arms:
      - name: ordinary
        polarization: "O"
        trace_offset: 0.0
      - name: extraordinary
        polarization: "E"
        trace_offset: 48.5  # Pixels between O and E beams
trace_arrangement:
  layout: paired  # O and E traces are paired per order
```

**Trace counting:**
- Total traces per order = sum of (1 if no beam_arms else len(beam_arms)) for each optical_path
- HARPS: 2 paths × 1 beam = 2 traces/order
- Polarimeter: 1 path × 2 beams = 2 traces/order
- HARPS + polarimeter: 2 paths × 2 beams = 4 traces/order

### Extension: Fiber Bundles / IFU (Many-Fiber Instruments)

For instruments with many fibers (e.g., 60-fiber pseudo-slit), need:
1. Compact fiber geometry definition
2. Multi-frame trace calibration (even/odd illumination)
3. Flexible extraction apertures (individual, grouped, arbitrary slit portions)

**Instrument config** (geometry - static):
```yaml
optical_paths:
  type: fiber_bundle
  count: 60
  arrangement: pseudo_slit
  spacing: 2.5  # pixels between fiber centers
  slit_coords: [0, 1]  # map fiber 0 → 0.0, fiber 59 → 1.0

trace_calibration:
  method: alternating_illumination
  sets:
    - name: even
      pattern: "0::2"  # fibers 0, 2, 4, ...
      file_keyword: "FLAT_EVEN"
    - name: odd
      pattern: "1::2"  # fibers 1, 3, 5, ...
      file_keyword: "FLAT_ODD"
  merge: true  # combine into single 60-trace set
```

**Extraction config** (runtime - flexible):
```yaml
# Mode 1: Extract all 60 fibers individually
extraction:
  mode: individual
  # → 60 spectra per order

# Mode 2: Two groups centered on specific fibers
extraction:
  mode: grouped
  apertures:
    - center_fiber: 7
      width_fibers: 15  # covers fibers 0-14
      name: "group_A"
    - center_fiber: 22
      width_fibers: 15  # covers fibers 15-29
      name: "group_B"
  # → 2 spectra per order

# Mode 3: Arbitrary slit portion via normalized coords
extraction:
  mode: slit_range
  apertures:
    - range: [0.0, 0.125]   # bottom 1/8th of slit
      name: "bottom"
    - range: [0.4, 0.6]     # middle 20%
      name: "center"
  # → 2 spectra per order

# Mode 4: Full slit as single spectrum
extraction:
  mode: slit_range
  apertures:
    - range: [0.0, 1.0]
      name: "full_slit"
  # → 1 spectrum per order (sum of all fibers)
```

**Implementation:**
```python
class FiberBundle:
    """Represents a bundle of fibers along a pseudo-slit."""

    def __init__(self, count: int, spacing: float):
        self.count = count
        self.spacing = spacing
        self.traces = []  # populated after tracing

    def fiber_to_slit(self, fiber: int) -> float:
        """Convert fiber number to normalized slit position [0,1]."""
        return fiber / (self.count - 1)

    def slit_to_fibers(self, slit_range: tuple[float, float]) -> list[int]:
        """Convert slit range to list of fiber indices."""
        lo, hi = slit_range
        return [i for i in range(self.count)
                if lo <= self.fiber_to_slit(i) <= hi]

    def get_aperture(self, spec: dict) -> Aperture:
        """Create extraction aperture from config."""
        if "center_fiber" in spec:
            center = spec["center_fiber"]
            width = spec["width_fibers"]
            fibers = list(range(center - width//2, center + width//2 + 1))
        elif "range" in spec:
            fibers = self.slit_to_fibers(spec["range"])
        return Aperture(fibers=fibers, name=spec.get("name"))
```

**Order tracing flow:**
```
1. Load even-illuminated flat → trace 30 fibers (well-separated)
2. Load odd-illuminated flat → trace 30 fibers (interleaved)
3. Merge by position → 60 traces, each tagged with fiber_id
4. Store as single trace set with fiber_id metadata
```

**Extraction flow:**
```
1. Load trace set (60 traces per order)
2. Parse extraction config (individual / grouped / slit_range)
3. For each aperture:
   - Identify which fibers contribute
   - Sum/combine those traces' extraction regions
   - Run slit decomposition on combined region
4. Output: N spectra per order (N = number of apertures)
```

### 4. Configuration Structure

**Single YAML file per instrument** (consolidates hardware, dimensions, headers):

```
instruments/
  harps.yaml         # All instrument config in one file
  harps.py           # Only truly custom logic (minimal, optional)
  uves.yaml
  crires_plus.yaml
  ...

settings/
  settings_HARPS.json    # Reduction parameters (unchanged)
  settings_default.json  # Fallback defaults
```

**Separation of concerns:**
- `instruments/*.yaml` - What the instrument IS (hardware, dimensions, header mapping)
- `settings/*.json` - HOW to reduce (polynomial degrees, iterations, thresholds)

The settings system stays as-is. It's orthogonal to instrument definition and already works well for per-instrument reduction tuning.

**harps.yaml** - Complete instrument definition (single file):
```yaml
# === METADATA ===
instrument: HARPS
telescope: ESO-3.6m
id_instrument: "HARPS"

# === DETECTORS ===
detectors:
  - name: blue_ccd
    naxis: [4096, 4096]
    orientation: 1
    prescan: {x: [0, 50], y: null}
    overscan: {x: [4046, 4096], y: null}
    bad_pixel_mask: "masks/blue_ccd_badpix.fits"
    amplifiers:
      - id: amp1
        region: {x: [0, 2048], y: [0, 4096]}
        gain: 1.2  # or header ref: {key: "ESO DET OUT1 CONAD"}
        readnoise: 3.5
        linearity: {model: polynomial, coefficients: [1.0, -2.3e-6, 1.1e-11]}
      - id: amp2
        region: {x: [2048, 4096], y: [0, 4096]}
        gain: 1.18
        readnoise: 3.6
        linearity: {model: polynomial, coefficients: [1.0, -2.1e-6, 0.9e-11]}

  - name: red_ccd
    naxis: [4096, 4096]
    orientation: 0
    # ... similar structure

# === OPTICAL PATHS ===
optical_paths:
  - name: fiber_a
    beam_arms: null  # No beam-splitter
  - name: fiber_b
    beam_arms: null

trace_arrangement:
  layout: interleaved
  separation: 15.5

# === DIMENSIONS ===
dimensions:
  mode:
    values: [BLUE, RED]
    header_key: "ESO INS MODE"
  fiber:
    values: [A, B, AB]
    # Derived via custom parser (see Python class)
    optional: true

property_mapping:
  detector:
    BLUE: blue_ccd
    RED: red_ccd

# === HEADER KEYWORDS ===
headers:
  target: "ESO OBS TARG NAME"
  date: "DATE-OBS"
  exposure_time: "ESO DET WIN1 DIT1"
  ra: "RA"
  dec: "DEC"

# File classification
file_types:
  bias:
    keyword: "ESO DPR TYPE"
    pattern: "BIAS"
  flat:
    keyword: "ESO DPR TYPE"
    pattern: "FLAT.*"
  wave:
    keyword: "ESO DPR TYPE"
    pattern: "WAVE.*"
  science:
    keyword: "ESO DPR TYPE"
    pattern: "OBJECT.*"
```

### 5. Observation Context

Runtime context extracted from headers:

```python
@dataclass
class ObservationContext:
    """Everything needed to process one frame"""
    instrument: str
    detector: str
    mode: str | None
    fiber: str | None
    exposure_type: str  # bias, flat, wave, science
    exposure_time: float

    # Resolved hardware
    active_amplifiers: list[Amplifier]
    active_optical_paths: list[OpticalPath]

    @classmethod
    def from_header(cls, header: fits.Header, instrument: Instrument) -> Self:
        """Extract context from FITS header using instrument config"""
        ...
```

### 6. Minimal Python Classes

Python classes only for:
1. Custom header parsing that can't be declarative
2. Instrument-specific file sorting logic
3. Complex derived values

```python
class HARPS(Instrument):
    """HARPS-specific logic only"""

    def parse_fiber(self, header: fits.Header) -> str:
        """Custom fiber parsing from DPR TYPE"""
        dpr_type = header.get("ESO DPR TYPE", "")
        # ... complex logic that doesn't fit in YAML
        return fiber

    # Everything else comes from config files
```

---

## Key Design Benefits

### 1. Explicit Over Implicit
- Dimension relationships are declared, not assumed from list alignment
- Clear what varies with what

### 2. Composable
- Build complex instruments from simple components
- Reuse detector/amplifier definitions

### 3. Handles All Complexity Sources

| Challenge | Solution |
|-----------|----------|
| Multiple fibers | `optical_paths[]` with explicit names |
| Beam-splitters | `beam_arms[]` per optical path, with polarization/offset |
| Mode explosion | Computed from orthogonal dimensions |
| Multi-detector | `detectors[]` - each is separate reduction unit |
| Multi-amplifier | `amplifiers[]` per detector - merged after calibration |
| Amp properties | Per-amp gain, readnoise, linearity, bad pixels |
| JSON/Python split | YAML layers (hardware, dimensions, headers) + minimal Python |

### 4. Self-Documenting
Configuration files serve as documentation of instrument capabilities.

### 5. Validation at Load Time
Schema can enforce:
- All dimensions referenced in mappings exist
- All detectors referenced are defined
- Required properties have values for all dimension combinations

---

## Example: CRIRES+ Redesigned

Current problem: 261 mode combinations, mode string parsing in Python, scattered config.

**crires_plus.yaml**:
```yaml
instrument: CRIRES+
telescope: VLT

# Three independent detectors - each is own reduction unit
detectors:
  - name: det1
    naxis: [2048, 2048]
    extension: "CHIP1.INT1"
    orientation: 0
    amplifiers:
      - id: amp1
        region: {x: [0, 2048], y: [0, 2048]}
        gain: {key: "ESO DET CHIP1 GAIN"}
        readnoise: {key: "ESO DET CHIP1 RON"}
  - name: det2
    naxis: [2048, 2048]
    extension: "CHIP2.INT1"
    # ... similar
  - name: det3
    naxis: [2048, 2048]
    extension: "CHIP3.INT1"
    # ... similar

optical_paths:
  - name: science
    beam_arms: null

# Orthogonal dimensions - NOT enumerated as 261 combinations
dimensions:
  band:
    values: [Y1029, J1232, J1228, H1559, K2148, L3262, M4187]  # etc
    header_key: "ESO INS WLEN ID"
  decker:
    values: [Open, pos1, pos2]
    header_key: "ESO INS SLIT NAME"
  detector:
    values: [det1, det2, det3]
    # Determined by which FITS extension is being processed

# Mode string is computed, not enumerated
mode_format: "{band}_{decker}_{detector}"

# Properties that vary - only need N values, not N×M×K
property_mapping:
  detector:
    det1: det1
    det2: det2
    det3: det3
  # Wavelength range comes from header, not config
  wavelength_range:
    source: header
    keys: ["ESO INS WLEN MIN", "ESO INS WLEN MAX"]

headers:
  # ... standard header mappings
```

**Key improvement**: Instead of enumerating 261 modes, we declare 3 orthogonal dimensions. The mode string is computed, and properties only need to vary along the axes that matter (detector varies with detector dimension, wavelength from header).

---

## Reduction Flow with New Model

### Single-Detector Instrument (e.g., UVES RED)
```
1. Load frame → apply amp corrections → unified 2D image
2. Trace all optical paths on this detector
3. Extract spectra for each (path, beam_arm) combination
4. Output: 1 spectrum per (order, path, beam_arm)
```

### Multi-Detector Instrument (e.g., CRIRES+ or XSHOOTER)
```
For each detector in instrument.detectors:
    1. Load frame for this detector
    2. Apply amp corrections → unified 2D image for this detector
    3. Trace optical paths that illuminate THIS detector
    4. Extract spectra
    5. Wavelength calibrate independently

Combine: Merge wavelength-calibrated spectra across detectors
         (overlap handling, stitching where wavelengths meet)
```

### Multi-Fiber + Beam-Splitter (e.g., polarimetric dual-fiber)
```
For each detector:
    For each optical_path illuminating detector:
        For each beam_arm in optical_path.beam_arms:
            - Trace this specific (path, arm) combination
            - Extract spectrum
            - Tag with (order, path.name, arm.name, arm.polarization)

Output: 4 spectra per order (fiber_a_O, fiber_a_E, fiber_b_O, fiber_b_E)
        Pipeline can then compute Stokes parameters from O/E pairs
```

### Key Principle: Detector as Reduction Unit
- Each detector is reduced independently up to wavelength calibration
- Only combine at the spectrum level, never at the image level
- This naturally handles misaligned detectors, different orientations, etc.

---

## Implementation Plan

### Guiding Principle
Each step should result in a working state. Add new code alongside old, validate it works, then switch over.

---

### Step 1: Pipeline Class (wrapper around existing)
**Goal:** New API available, old code unchanged.

```python
# pyreduce/pipeline.py (NEW FILE)
class Pipeline:
    """Fluent API wrapping existing Step classes."""

    def __init__(self, instrument, output_dir, config=None):
        self.instrument = instrument
        self.output_dir = output_dir
        self.config = config or {}
        self._steps = []
        self._data = {}

    def bias(self, files): ...
    def flat(self, files): ...
    def trace_orders(self): ...
    # ... each method appends to _steps, returns self

    def run(self):
        # Internally uses existing Step classes (Bias, Flat, etc.)
        for name, files in self._steps:
            step_class = STEP_CLASSES[name]
            step = step_class(*self._make_args(), **self.config.get(name, {}))
            result = step.run(files, **self._get_dependencies(name))
            self._data[name] = result
        return self._data
```

**Test:** Write `test_pipeline.py`, run existing tests still pass.

**Files changed:** Add `pyreduce/pipeline.py`, add `test/test_pipeline.py`

---

### Step 2: YAML Instrument Loader (parallel to JSON)
**Goal:** Can load instruments from YAML, JSON still works.

```python
# pyreduce/instruments/loader.py
def load_instrument(name_or_path):
    if name_or_path.endswith('.yaml'):
        return Instrument.from_yaml(name_or_path)
    elif name_or_path.endswith('.json'):
        return Instrument.from_json(name_or_path)  # existing
    else:
        # Load by name from package (existing behavior)
        return load_instrument_by_name(name_or_path)
```

**Convert ONE instrument** (UVES - simple) to YAML format:
- Keep same flat structure initially (just format change)
- `instruments/uves.yaml` alongside `instruments/uves.json`

**Test:** Run UVES tests with YAML config, verify identical results.

**Files changed:** Add `loader.py`, add `uves.yaml`, modify `instrument_info.py`

---

### Step 3: Convert All Instruments to YAML
**Goal:** All instruments in YAML, delete JSON files.

For each instrument:
1. Convert JSON → YAML (automated script)
2. Run that instrument's tests
3. Commit

Order (simple → complex):
1. UVES (already done)
2. LICK_APF, MCDONALD (simple)
3. XSHOOTER, HARPS (modes)
4. CRIRES+, NIRSPEC (complex)
5. JWST_NIRISS, JWST_MIRI (space)

**Delete:** All `*.json` instrument files, `instrument_schema.json`

---

### Step 4: Pydantic Models for Instrument Config
**Goal:** Type-safe instrument loading with validation.

```python
# pyreduce/instruments/models.py
from pydantic import BaseModel

class Amplifier(BaseModel):
    id: str
    gain: float | dict  # float or {key: "HEADER_KEY"}
    readnoise: float | dict
    region: dict | None = None

class Detector(BaseModel):
    name: str
    naxis: tuple[int, int]
    orientation: int
    amplifiers: list[Amplifier] = []

class OpticalPath(BaseModel):
    name: str
    beam_arms: list[BeamArm] | None = None

class InstrumentConfig(BaseModel):
    instrument: str
    detectors: list[Detector]
    optical_paths: list[OpticalPath]
    dimensions: dict = {}
    headers: dict = {}
    # ... etc
```

**Migrate YAML schema** to use new structure (detectors[], optical_paths[], etc.)

**Test:** Load each instrument, validate Pydantic catches errors.

---

### Step 5: Update Instrument Class to Use Models
**Goal:** `Instrument` class uses Pydantic models internally.

```python
class Instrument:
    def __init__(self, config: InstrumentConfig):
        self.config = config
        self.detectors = config.detectors
        self.optical_paths = config.optical_paths
        # ...

    @classmethod
    def from_yaml(cls, path: str) -> "Instrument":
        data = yaml.safe_load(open(path))
        config = InstrumentConfig(**data)
        return cls(config)
```

**Refactor** methods like `get_extension()`, `load_fits()` to use new model.

**Test:** All instrument tests pass.

---

### Step 6: CLI with Click
**Goal:** `uv run reduce trace ...` works.

```python
# pyreduce/__main__.py
import click
from .pipeline import Pipeline
from .instruments import load_instrument

@click.group()
def cli():
    """PyReduce echelle spectrograph reduction."""
    pass

@cli.command()
@click.argument('instrument')
@click.option('--files', '-f', multiple=True)
@click.option('--output', '-o', default='.')
def trace(instrument, files, output):
    inst = load_instrument(instrument)
    Pipeline(inst, output).flat(list(files)).trace_orders().run()

# ... other commands

if __name__ == '__main__':
    cli()
```

**Add** `[project.scripts]` to pyproject.toml:
```toml
[project.scripts]
reduce = "pyreduce.__main__:cli"
```

**Test:** `uv run reduce --help` works.

---

### Step 7: Deprecate Old Reducer Class
**Goal:** Pipeline is the primary API, Reducer is thin wrapper.

```python
class Reducer:
    """Legacy interface. Use Pipeline instead."""

    def __init__(self, ...):
        warnings.warn("Reducer is deprecated, use Pipeline", DeprecationWarning)
        self.pipeline = Pipeline(instrument, output_dir, config)
        # ...
```

**Update** examples to use Pipeline.
**Update** documentation.

---

### Summary: What's Working When

| After Step | What Works |
|------------|------------|
| 1 | New Pipeline API + all existing code |
| 2 | YAML loading for UVES + all existing |
| 3 | All instruments in YAML, JSON deleted |
| 4 | Pydantic validation on load |
| 5 | New instrument model throughout |
| 6 | CLI commands |
| 7 | Pipeline is primary, Reducer deprecated |

### Time Estimate
Not providing time estimates per instructions, but ordering by risk/complexity:
- Steps 1-3: Low risk, can be done quickly
- Steps 4-5: Medium risk, need careful testing
- Steps 6-7: Low risk, mostly additive

---

## Reduction Step Architecture

### Current Pattern
```python
# String-based step names, Reducer holds all state
reducer = Reducer(files, output_dir, target, instrument, mode, night, config)
reducer.run_steps(["bias", "flat", "orders", "wavecal", "science"])

# Each step is a class with run/save/load methods
class Bias(CalibrationStep):
    _dependsOn = ["mask"]
    def run(self, files, mask): ...
    def save(self, bias, head): ...
    def load(self): ...
```

**Issues:**
- String-based step names are error-prone
- Data flow hidden in `self.data` dict
- Lots of boilerplate per step (class with run/save/load)
- Hard to understand dependencies without reading code
- Awkward for custom pipelines (toes_example.py pattern)

### Proposed: Pipeline Builder

```python
# Fluent API - chain steps, explicit data flow
result = (
    Pipeline(instrument, output_dir)
    .bias(files["bias"])           # → instrument_bias.fits
    .flat(files["flat"])           # → instrument_flat.fits
    .trace_orders()                # → instrument_orders.npz
    .normalize_flat()              # → instrument_norm_flat.npz
    .wavelength_calibration(files["wavecal"])
    .extract(files["science"])     # → instrument_science_001.ech
    .run()
)

# For prototyping - start from existing intermediate products
result = (
    Pipeline(instrument, output_dir)
    .load_orders("instrument_orders.npz")  # load instead of compute
    .extract(files["science"])
    .run()
)

# Skip steps that have existing outputs
result = pipeline.run(skip_existing=True)
```

### Design Principles

1. **Disk-based persistence** - Each step saves output to disk with consistent naming. No in-memory caching needed.

2. **Explicit over implicit** - You see exactly which steps run and in what order. Dependencies visible in method signatures.

3. **Composable** - Easy to skip steps, substitute data, or build custom pipelines.

4. **Type-safe** - Pipeline methods have typed signatures, IDE autocompletion works.

### Implementation Sketch

```python
class Pipeline:
    def __init__(self, instrument: Instrument, output_dir: str, config: dict = None):
        self.instrument = instrument
        self.output_dir = output_dir
        self.config = config or {}
        self._steps: list[tuple[str, Callable, dict]] = []
        self._data: dict[str, Any] = {}

    def bias(self, files: list[str]) -> "Pipeline":
        """Combine bias frames."""
        self._steps.append(("bias", reduce_bias, {"files": files}))
        return self

    def flat(self, files: list[str]) -> "Pipeline":
        """Combine flat frames, requires bias."""
        self._steps.append(("flat", reduce_flat, {"files": files}))
        return self

    def trace_orders(self, **kwargs) -> "Pipeline":
        """Trace orders on flat field."""
        self._steps.append(("orders", trace_orders, kwargs))
        return self

    def extract(self, files: list[str], **kwargs) -> "Pipeline":
        """Extract spectra from science frames."""
        self._steps.append(("science", extract_science, {"files": files, **kwargs}))
        return self

    def load_orders(self, path: str) -> "Pipeline":
        """Load orders from existing file instead of tracing."""
        self._data["orders"] = load_orders_from_disk(path)
        return self

    def run(self, skip_existing: bool = False) -> dict:
        """Execute all queued steps."""
        for name, func, kwargs in self._steps:
            if skip_existing and self._output_exists(name):
                self._data[name] = self._load_output(name)
                continue
            # Inject dependencies from previous steps
            deps = self._resolve_dependencies(func, self._data)
            result = func(self.instrument, **deps, **kwargs, **self.config.get(name, {}))
            self._save_output(name, result)
            self._data[name] = result
        return self._data
```

### Backward Compatibility

Keep `Reducer` class as a thin wrapper:

```python
class Reducer:
    """Legacy interface - wraps Pipeline for backward compatibility."""

    def __init__(self, files, output_dir, target, instrument, mode, night, config, ...):
        self.pipeline = Pipeline(instrument, output_dir, config)
        self.files = files
        # ...

    def run_steps(self, steps="all"):
        if "bias" in steps:
            self.pipeline.bias(self.files.get("bias", []))
        if "flat" in steps:
            self.pipeline.flat(self.files.get("flat", []))
        # ... etc
        return self.pipeline.run()
```

### Step Functions

Steps become pure functions (or thin wrappers around existing modules):

```python
def reduce_bias(instrument: Instrument, files: list[str], mask: np.ndarray = None) -> tuple[np.ndarray, fits.Header]:
    """Combine bias frames into master bias."""
    # ... implementation
    return bias, header

def reduce_flat(instrument: Instrument, files: list[str], mask: np.ndarray = None, bias: np.ndarray = None) -> tuple[np.ndarray, fits.Header]:
    """Combine flat frames into master flat."""
    # ... implementation
    return flat, header

def trace_orders(flat: np.ndarray, **config) -> Orders:
    """Detect and trace echelle orders."""
    # ... implementation
    return orders
```

### Benefits for New Instruments

The toes_example.py pattern becomes cleaner:

```python
# Before: verbose Reducer setup
instrument = create_custom_instrument("TOES", extension=0)
instrument.info["gain"] = 1.1
# ... 30 lines of config
reducer = Reducer(files, output_dir, target, instrument, mode, night, config)
reducer.run_steps(steps)

# After: fluent pipeline
instrument = Instrument.from_yaml("toes.yaml")  # or create_custom_instrument()
result = (
    Pipeline(instrument, output_dir)
    .config(orders={"degree": 4, "min_cluster": 3000})  # override specific settings
    .flat(flat_files)
    .trace_orders()
    .wavelength_calibration(wavecal_files)
    .extract(science_files)
    .run()
)
```

---

## Output Format

### Current: .ech Files

Currently outputs `.ech` files (FITS with binary table extension). One file per science frame with columns for wavelength, flux, uncertainty per order.

### Challenge: Multiple Fibers/Beams

With multi-fiber instruments or beam-splitters, we have multiple spectra per order:
- HARPS: fiber_a, fiber_b (or combined AB)
- Polarimeter: ordinary beam, extraordinary beam
- Combined: 4 spectra per order (fiber_a_O, fiber_a_E, fiber_b_O, fiber_b_E)

### Proposed Structure

Option A: **Hierarchical FITS extensions**
```
HDU 0: Primary (header with observation metadata)
HDU 1: FIBER_A (binary table with order, wavelength, flux, uncertainty)
HDU 2: FIBER_B (binary table)
# or for polarimetry:
HDU 1: FIBER_A_O
HDU 2: FIBER_A_E
HDU 3: FIBER_B_O
HDU 4: FIBER_B_E
```

Option B: **Wide table with columns per trace**
```
HDU 1: SPECTRA
  - ORDER (int)
  - WAVELENGTH (float)
  - FLUX_A, FLUX_B, FLUX_A_O, FLUX_A_E, ... (floats)
  - ERR_A, ERR_B, ... (floats)
```

Option C: **Separate files with consistent naming**
```
output/
  harps_science_001_fiber_a.ech
  harps_science_001_fiber_b.ech
  # or
  harps_science_001_ord_O.ech  # ordinary beam
  harps_science_001_ext_E.ech  # extraordinary beam
```

**Recommendation**: Option A (hierarchical extensions) - keeps related data together, self-documenting via HDU names, standard FITS viewers can browse structure.

### Metadata

Each spectrum should carry:
- `OPTICAL_PATH`: fiber_a, fiber_b, etc.
- `BEAM_ARM`: ordinary, extraordinary (or null if no beam-splitter)
- `POLARIZATION`: O, E, L, R (or null)
- `DETECTOR`: which detector this came from
- `ORDER`: echelle order number

---

## CLI Runner

### Current: Script-Based

```bash
# Run example script
uv run python examples/uves_example.py

# Or use main() with lots of arguments
uv run python -m pyreduce --instrument UVES --target HD12345 ...
```

### Proposed: Step-Based CLI

```bash
# Run individual steps
uv run reduce bias instrument.yaml --files bias/*.fits --output output/
uv run reduce flat instrument.yaml --files flat/*.fits --output output/
uv run reduce trace instrument.yaml --output output/
uv run reduce wavecal instrument.yaml --files thar/*.fits --output output/
uv run reduce extract instrument.yaml --files science/*.fits --output output/

# Run full pipeline
uv run reduce run instrument.yaml --steps bias,flat,trace,extract --output output/

# Or with config file
uv run reduce run reduction.yaml  # contains instrument, files, steps, output
```

### Implementation

```python
# pyreduce/__main__.py
import click

@click.group()
def cli():
    """PyReduce echelle spectrograph reduction pipeline."""
    pass

@cli.command()
@click.argument("instrument", type=click.Path(exists=True))
@click.option("--files", "-f", multiple=True, help="Input FITS files")
@click.option("--output", "-o", default=".", help="Output directory")
@click.option("--config", "-c", type=click.Path(), help="Step config overrides")
def trace(instrument, files, output, config):
    """Trace echelle orders on flat field."""
    inst = Instrument.from_yaml(instrument)
    pipe = Pipeline(inst, output)
    if files:
        pipe.flat(list(files))
    pipe.trace_orders()
    pipe.run()

@cli.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--steps", "-s", default="all", help="Steps to run (comma-separated)")
def run(config, steps):
    """Run full reduction pipeline from config file."""
    # Load config YAML with instrument, files, steps, output
    ...
```

### Config File for Full Reduction

```yaml
# reduction.yaml
instrument: instruments/harps.yaml
output: /data/reduced/HD12345/

files:
  bias: /data/raw/bias/*.fits
  flat: /data/raw/flat/*.fits
  wavecal: /data/raw/thar/*.fits
  science: /data/raw/science/*.fits

steps: [bias, flat, trace, wavecal, extract]

# Optional: override default settings
config:
  trace:
    degree: 4
    min_cluster: 3000
  extract:
    oversampling: 8
```

---

## Testing Strategy

### Current Test Structure

```
test/
  test_extract.py    # Unit tests - pure algorithms (PRESERVE)
  test_cwrappers.py  # Unit tests - C wrappers (PRESERVE)
  test_clipnflip.py  # Unit tests - utilities (PRESERVE)
  test_bias.py       # Integration tests using Step classes
  test_flat.py       # Integration tests using Step classes
  test_orders.py     # Integration tests using Step classes
  test_wavecal.py    # Integration tests (slow)
  conftest.py        # Fixtures - tightly coupled to Step classes
```

### Test-Driven Redesign Insight

**Unit tests ARE the stable API contract.** The redesign must preserve:

```python
# Core extraction functions (test_extract.py) - PRESERVE SIGNATURES
extract.extend_orders(orders, height) -> array
extract.fix_column_range(cr, orders, ew, nrow, ncol) -> (cr, orders)
extract.make_bins(swath_width, xlow, xhigh, ycen) -> (nbin, starts, ends)
extract.arc_extraction(img, orders, ew, cr, tilt, shear) -> (spec, unc)
extract.optimal_extraction(img, orders, xwd, cr, tilt, shear) -> (spec, slitf, unc)
extract.extract(img, orders, ...) -> (spec, unc, slitf, ...)

# C wrappers (test_cwrappers.py) - PRESERVE SIGNATURES
slitfunc(img, ycen, lambda_sp, lambda_sf, osample) -> (spec, slitf, ...)
slitfunc_curved(img, ycen, tilt, shear, ...) -> (spec, slitf, ...)
```

**Integration fixtures need refactoring** - Currently coupled to Step classes:
```python
# Current (conftest.py)
step = Bias(*step_args, **settings)
bias = step.run(files, mask)

# After redesign
pipe = Pipeline(instrument, output_dir)
bias = pipe.bias(files).run()["bias"]
```

### Test Categories

**Unit tests** (`@pytest.mark.unit`) - **Keep unchanged**:
- Test pure functions with synthetic data
- Fast (<1s per test)
- No file I/O, no instrument downloads
- ~40 tests, run in parallel

**Integration tests** (`@pytest.mark.instrument`) - **Refactor fixtures**:
- Test full pipeline steps with real instrument data
- Download sample datasets on first run
- Parametrized across instruments (UVES, XSHOOTER, NIRSPEC, JWST_NIRISS)
- ~70 tests, run sequentially (shared datasets)

### Testing Pipeline Builder

```python
# test/test_pipeline.py

def test_pipeline_bias_only(tmp_path, synthetic_bias_files):
    """Test running just bias step."""
    inst = create_test_instrument()
    result = (
        Pipeline(inst, tmp_path)
        .bias(synthetic_bias_files)
        .run()
    )
    assert "bias" in result
    assert (tmp_path / "test_bias.fits").exists()

def test_pipeline_skip_existing(tmp_path, synthetic_files):
    """Test that skip_existing works."""
    inst = create_test_instrument()

    # First run
    Pipeline(inst, tmp_path).bias(synthetic_files["bias"]).run()

    # Second run with skip_existing
    result = (
        Pipeline(inst, tmp_path)
        .bias(synthetic_files["bias"])  # should skip
        .flat(synthetic_files["flat"])
        .run(skip_existing=True)
    )
    # bias should be loaded, not recomputed

def test_pipeline_load_intermediate(tmp_path, existing_orders_file):
    """Test loading from intermediate file."""
    inst = create_test_instrument()
    result = (
        Pipeline(inst, tmp_path)
        .load_orders(existing_orders_file)
        .extract(science_files)
        .run()
    )
    assert "science" in result

@pytest.mark.instrument
@pytest.mark.parametrize("instrument", ["UVES", "HARPS"])
def test_full_reduction(instrument, sample_dataset):
    """Integration test: full pipeline on real data."""
    inst = load_instrument(instrument)
    result = (
        Pipeline(inst, sample_dataset.output_dir)
        .bias(sample_dataset.files["bias"])
        .flat(sample_dataset.files["flat"])
        .trace_orders()
        .extract(sample_dataset.files["science"])
        .run()
    )
    assert "science" in result
    # Check output file exists and has expected structure
```

### Testing Step Functions

```python
# test/test_steps.py

def test_reduce_bias_synthetic():
    """Test bias combination with synthetic data."""
    files = create_synthetic_bias_frames(n=5, shape=(100, 100), value=1000)
    inst = create_test_instrument()

    bias, header = reduce_bias(inst, files)

    assert bias.shape == (100, 100)
    assert np.isclose(bias.mean(), 1000, rtol=0.01)

def test_trace_orders_synthetic():
    """Test order tracing on synthetic flat."""
    flat = create_synthetic_flat_with_orders(n_orders=10)

    orders = trace_orders(flat, degree=4, min_cluster=100)

    assert len(orders) == 10
    for order in orders:
        assert order.polynomial_degree == 4
```

### Fixtures for New Architecture

```python
# test/conftest.py

@pytest.fixture
def test_instrument():
    """Create a minimal test instrument."""
    return Instrument.from_dict({
        "name": "TEST",
        "detectors": [{
            "name": "ccd",
            "naxis": [100, 100],
            "amplifiers": [{"id": "amp1", "gain": 1.0, "readnoise": 5.0}]
        }],
        "optical_paths": [{"name": "science", "beam_arms": None}]
    })

@pytest.fixture
def synthetic_bias_files(tmp_path):
    """Create synthetic bias FITS files."""
    files = []
    for i in range(5):
        data = np.random.normal(1000, 10, (100, 100))
        path = tmp_path / f"bias_{i}.fits"
        fits.writeto(path, data.astype(np.float32))
        files.append(str(path))
    return files
```

---

## Future: Extraction Backend (CharSlit)

### Current State

PyReduce uses CFFI-wrapped C code for optimal extraction:
- `pyreduce/clib/slit_func_bd.c` - Vertical slit function decomposition
- `pyreduce/clib/slit_func_2d_xi_zeta_bd.c` - Curved 2D extraction
- `pyreduce/cwrappers.py` - Python interface returning tuples

### CharSlit Project

A separate project (`CharSlit.git`) provides an improved extraction with:
- **nanobind** wrapper (faster, cleaner than CFFI)
- **scikit-build-core** + CMake (modern build system)
- **Improved algorithm** with slit curvature and per-row x-offsets (slitdeltas)
- **Own test suite** (24 fast + 8 visualization tests)
- **Dict-based API**: Returns `{spectrum, slitfunction, model, uncertainty, info}`

### Integration Plan

**Phase 1: Complete Pipeline redesign first**
- Keep current CFFI extraction stable
- Don't compound risk with extraction changes during architecture overhaul

**Phase 2: Add CharSlit as optional backend**
```python
# pyreduce/extraction/__init__.py
from .cffi_backend import slitfunc, slitfunc_curved  # Current
from .charslit_backend import slitdec  # New (optional)

# In Pipeline or settings
config:
  extract:
    backend: "charslit"  # or "cffi" (default for now)
```

**Phase 3: Validation**
- Run both backends on same data
- Compare spectrum, slitfunction, uncertainties
- Verify results match within tolerance
- Benchmark performance

**Phase 4: Switch default**
- Once validated, make CharSlit the default
- Keep CFFI as fallback for edge cases or platforms without CMake

**Phase 5: Deprecate CFFI backend**
- After one release cycle with CharSlit default
- Remove CFFI code and hatch_build.py complexity

### Benefits of CharSlit

1. **Better build system** - scikit-build-core handles cross-platform compilation better than custom hatch_build.py
2. **nanobind performance** - Faster Python↔C boundary than CFFI
3. **Unified API** - Single `slitdec()` function handles both vertical and curved cases
4. **Richer output** - Returns model image and info dict, useful for diagnostics
5. **Own tests** - Extraction algorithm tested independently of PyReduce

### Risks

- API change (dict vs tuple returns) requires `extract.py` refactoring
- Algorithm improvements may change results slightly (need revalidation)
- Build dependency on CMake (most systems have it, but adds requirement)

---

## Open Questions

1. **Schema validation** - Pydantic models (recommended) vs JSON Schema vs both?
2. **Backward compatibility** - How long to support old JSON format during migration?
3. **Header keyword abstraction** - Should we have an ESO/non-ESO keyword adapter layer?
4. **Wavecal files** - Should wavelength solution paths be in instrument config or discovered at runtime?
5. **CLI tool name** - `pyreduce` vs `reduce` vs something else?
6. **CharSlit integration** - As git submodule, or separate PyPI package dependency?
