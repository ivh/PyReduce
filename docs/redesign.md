# Architecture Changes (v0.7 / v0.8)

This document summarizes the major changes for users coming back to PyReduce after the 0.7/0.8 series.

---

## Trace Dataclass

Previously, trace data was scattered across parallel arrays (`traces`, `column_range`, `curvature`, `wave_coef`) stored in separate `.npz` files. These could easily get out of sync.

Now all trace data lives in a single `Trace` dataclass:

```python
@dataclass
class Trace:
    m: int | None              # spectral order number
    group: str | int | None    # fiber group ('A', 'B', 'cal', ...)
    fiber_idx: int | None      # fiber index within group

    pos: np.ndarray            # y(x) position polynomial
    column_range: tuple[int, int]
    height: float | None       # extraction aperture height

    slit: np.ndarray | None    # curvature coefficients (filled by curvature step)
    slitdelta: np.ndarray | None
    wave: np.ndarray | None    # wavelength polynomial (filled by wavecal step)
```

Pipeline steps update traces in-place as they run. All trace data is saved to a single `.traces.fits` file. Old `.npz` files are still readable.

See [trace_model.py](../pyreduce/trace_model.py) for the full implementation.

---

## Spectra Format

The old `Echelle` class stored spectra as unlabeled 2D arrays — after extraction there was no way to know which row was which order or fiber.

Now each extracted spectrum is a `Spectrum` object with full metadata (order number, group, extraction parameters), saved as a FITS binary table with one row per trace. Invalid pixels use NaN instead of a separate mask array.

```python
spectra = Spectra.read("file.science.fits")
for s in spectra:
    print(s.m, s.group, s.spec.shape)
```

See [output_formats.md](output_formats.md) for the file format specification.

A deprecated `echelle.py` shim still exists for backward compatibility but will be removed in a future version.

---

## Multi-Fiber Support

PyReduce now handles instruments with fiber bundles (MOSAIC, ANDES, HARPSPOL):

- **Fiber groups** defined in instrument YAML (`fibers` config)
- **Bundle detection** for repeating fiber patterns (IFU, multi-fiber pseudo-slits)
- **Per-step trace selection** (`science: [A, B]`, `wavecal: [cal]`)
- **Merged traces** — fibers within a group can be averaged for extraction

See [fiber_bundle_tracing.md](fiber_bundle_tracing.md) for details.

---

## Pipeline API

The old `pyreduce.reduce.main()` is replaced by a fluent Pipeline API:

```python
from pyreduce.pipeline import Pipeline

Pipeline.from_instrument(
    instrument="UVES",
    target="HD132205",
    steps=("bias", "flat", "trace", "science"),
).run()
```

The old `main()` still works but emits a deprecation warning.

---

## CLI

The argparse CLI is replaced with Click. Individual steps are now top-level commands:

```bash
uv run reduce run UVES -t HD132205 --steps bias,flat,trace,science
uv run reduce trace UVES -t HD132205
uv run reduce download UVES
```

See [cli.md](cli.md) for the full reference.

---

## Configuration

- Instrument configs use YAML (validated by Pydantic), organized into `instruments/{NAME}/` directories
- Settings cascade: `defaults/settings.json` < `{INSTRUMENT}/settings.json` < `{INSTRUMENT}/settings_{channel}.json` < runtime overrides
- `mode` is now called `channel`

See [configuration_file.md](configuration_file.md) for details.

---

## Naming Changes

| Old | New | Reason |
|-----|-----|--------|
| `nord` / `iord` | `ntrace` / `idx` | "order" conflated spectral order (m) with trace index |
| `orders` step | `trace` step | |
| `extraction_width` | `extraction_height` | dispersion is horizontal, extraction is vertical |
| `mode` | `channel` | |
| `.ech` output | `.fits` output | standard format |
| Mask 0=bad, 1=good | 1=bad, 0=good | numpy convention |

---

## Extraction Backends

Two extraction backends are available:

- **CFFI** (default) — the original C slit function decomposition, supports curvature degree 1-2
- **Charslit** (optional) — a newer backend supporting curvature up to degree 5 and per-row slitdelta corrections. Install with `uv sync --extra charslit` and enable with `PYREDUCE_USE_CHARSLIT=1`.

---

## Future Work

### Multi-Detector Model

Explicit `Detector` and `Amplifier` classes for instruments with multiple readout amplifiers or detectors. Currently handled via the `channels` config parameter.

### Dimension System

Declarative config for instruments with mode explosion (e.g. CRIRES+: 29 bands x 3 deckers x 3 detectors).
