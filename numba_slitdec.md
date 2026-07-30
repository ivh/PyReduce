# A pure-Python slitdec: assessment, port, and results

Consolidates two independent assessment sessions (July 2026) and the port that
followed. Question was: master's C extraction got a large speedup (`slitdec`);
the old `numba` branch has a pure-Python reimplementation benchmarked against
the *previous* C code. How far behind is it, could the speedups be ported, and
is a numpy-only version viable instead?

**Outcome: ported. `pyreduce/numba_slitdec.py` runs at 1.38x the C, agreeing
with it to 1.3e-14.** Details below.

## Where things live

- **`master` / `numba` branches**: `pyreduce/clib/slitdec.c` (commit `676d785`,
  "use slitdec (copied from charslit) as the sole CFFI backend"), wrapper
  `cwrappers.slitdec`. This is the fast version and the reference.
- **`numba` branch**: `pyreduce/numba_slitdec.py`, the new port. Selected with
  `PYREDUCE_USE_NUMBA=1`, tested by `test/test_numba_slitdec.py`.
- **`numba_old` branch** (was `numba`, base `4db3619`, pre-`steps/`-refactor):
  `pyreduce/numba_extract.py` (~919 lines) + the *old* CFFI
  `pyreduce/clib/slit_func_2d_xi_zeta_bd.c`. Superseded; kept for reference only.

## Starting point: the old numba code was a port of the old algorithm

Benchmark on identical straight synthetic swaths (spectra agreed to ~1e-11),
each backend timed from its own venv. Median per swath:

| Swath | `slitdec` | old CFFI | old Numba |
|---|---|---|---|
| 1000×15, os=8 | 2.3 ms | ~27 ms | 8.7 ms |
| 4096×25, os=10 | 31 ms | — | 160 ms |
| 4096×25, os=10, +hotpix | 54 ms | — | 300 ms |

- `slitdec` vs old CFFI, small swath: **27 / 2.3 ≈ 11–12x**.
- old Numba vs old CFFI: **~1.25x** slower — same algorithm, numba paying a
  JIT/dispatch tax.
- old Numba vs `slitdec`: **~3.7x (small) → 5.6x (large, dirty)**. The gap
  widened with columns, osample and rejection-iteration count.

`numba_extract.build_sL_system` / `build_sP_system` were line-for-line
transliterations of the old `slit_func_2d_xi_zeta_bd.c` fill loops: full **xi
tensor**, subpixel-centric fill with the 4-corner `for n in range(4)` expansion,
inner `m_zeta` scan with per-iteration bounds checks (`if 0 <= col_idx < ...`)
in the innermost body.

It had adopted **none** of the slitdec restructuring. `slitdec` is the *same*
xi/zeta math (same `bandsol`, same index helpers) — just assembled differently.
So the 5x gap was algorithmic structure the numba code never received, not an
inherent numba ceiling. Note also that a separate measurement (`speed.md`) put
the C optimization rounds at 34x cumulative rather than the 11–12x seen here;
different swath configs and different round counts. Either way the conclusion
holds: **the win came from reformulation and data layout, not from C being C.**

### What slitdec changed

`slitdec.c` documents it: *"the SLE fill loops became pixel-centric, only zeta
is needed."*

| # | Change | Buys | Ported? |
|---|---|---|---|
| 1 | **Pixel-centric fill, xi tensor dropped** — loop over `nrows×ncols` detector pixels, each touching its short zeta list, instead of subpixels × 4 xi corners | Removes ×4 corner loop, halves working set | **Yes**, fully |
| 2 | **Precomputed per-pixel key ranges** (`z_rng`) — merge into a dense window instead of searching unique keys; inner loops run over populated band entries only | Kills the `if 0 <= col_idx < …` branches in the hot loop | **Yes**, fully |
| 3 | **`restrict` pointers + hoisted const index base**, contiguous `arow`/`zrow` walks | SIMD vectorization, no repeated `*_index()` arithmetic | **Partially** — `fastmath=True` plus fresh contiguous arrays capture most of it; no explicit aliasing guarantee |
| 4 | Cache-friendly contiguous access | locality | **Yes** — falls out of #1's loop order |

## The numpy-vs-numba question

The second assessment asked whether numpy/scipy alone could get within 5x,
mapping each stage to a `np.bincount` scatter over the COO zeta arrays:

| Stage | numpy equivalent | est. cost |
|---|---|---|
| sL merge | weighted `bincount` over ~5M zeta entries into a padded (npix, K) window | 5–10 ms |
| sL pair fill | per band-offset d: `W[:,j]*W[:,j+d]` + `bincount` into precomputed band bins | 15–30 ms |
| sP fill | same pattern, x-window Kx ≤ 3–4 | 2–5 ms |
| solves | `scipy.linalg.solve_banded` | ≪1 ms |
| model + unc | gather `sL[k0[:,None]+j]`, multiply, `sum(axis=1)` | 3–5 ms |
| convergence | `np.partition` percentile | 2–5 ms |

Reference was a CRIRES swath, 176×2048 = 360k pixels, ny = 1063, C ~20 ms/iter
→ ~40–80 ms/iter, i.e. 2–4x, bad end ~6x. The enabler either way: everything
geometric is **fixed across iterations** and built once per call (COO zeta,
padded windows, scatter bin indices); only the sP/sL-weighted values change.

**Decision: numba, not numpy.** numba was explicitly allowed, and it wins on
every axis that mattered — a mechanical transliteration (low correctness risk,
diffable against the C) landing at 1.2–2x, versus a novel bincount design at
2–6x that might miss the target and only tells you after it's built. Both drop
the build toolchain, which was the actual motivation. The bincount version
remains a cheap follow-up experiment if the numba dependency ever becomes a
problem: the geometry construction is shared, and the two fill kernels sit
behind a clean seam.

The old numba branch was **not** used as a base. Its hot code was exactly what
had to be replaced, and it sat on a pre-`steps/` base — rebasing a year of
pipeline churn to reuse code slated for deletion. It was branched aside as
`numba_old` and the port written fresh against `slitdec.c`.

## Results

Structure mirrors `slitdec.c`: `_zeta_tensors` (zeta only), pixel-centric
`_fill_sl`/`_fill_sp` with dense merge windows off `z_rng`, `_bandsol`, and the
whole iteration loop inside one `@njit` so there is no per-iteration Python.
Zeta is three parallel arrays (`zx`, `ziy`, `zw`) rather than the C
struct-of-three, which lets LLVM vectorize the contiguous walks.
`fastmath=True` on the fills (element-wise, nothing to reassociate); off for
`_bandsol`, which is unpivoted Gaussian elimination.

### Speed

| Case | `slitdec` (C) | `numba_slitdec` | Ratio |
|---|---|---|---|
| Raw swath, 4096×25, os=10, +hotpix | 51.8 ms | 71.4 ms | **1.38x** |
| `extract_spectrum`, 6 orders × 9 swaths, curvature + rejection | 0.27 s | 0.37 s | **1.37x** |

At the good end of the predicted 1.5–2x, and a 2.7–4x improvement on the old
numba code's 3.7–5.6x.

### Correctness

Worst relative deviation vs the C oracle: **1.3e-14**. Identical masks,
identical iteration counts, identical `delta_x` and status codes. Verified
across straight and curved geometry, `lambda_sP > 0`, `kappa = 0`, preset slit
function, and the `nx > ncols` bail-out; at both the raw-swath level and through
`extract_spectrum` (swath splitting, overlap merging, outlier rejection). Full
unit suite (724 tests) passes on both backends.

`test/test_numba_slitdec.py` diffs the two backends directly and is the gate
when `slitdec.c` changes: **the C stays the reference — port changes forward,
don't let the two drift.**

### JIT cost

| | |
|---|---|
| `import pyreduce.numba_slitdec` (mostly numba itself) | ~1.8 s |
| First call, cold compile | ~4.9 s |
| First call, `cache=True` hit on disk | ~180 ms |
| Warm call (small swath) | 0.3 ms |

Irrelevant on a full order-by-order reduction; noticeable on a one-swath script,
and paid once per worker process when extraction is parallelized.

### Two caveats found on the way

- **The UVES example is broken on master**, independent of this work.
  `reduce run UVES -t HD132205 --steps bias,flat,trace,norm_flat,science` dies in
  `extract.fix_extraction_height` with `ValueError: Check your column ranges.
  Traces 16 and 17 are weird`. Confirmed by stashing the `extract.py` change and
  re-running. That is why end-to-end verification used a synthetic frame.
- **Installing numba pins numpy down** 2.5.1 → 2.4.6 (numba 0.66 / llvmlite 0.48
  don't support 2.5 yet). Opt-in extra, but `uv sync --extra numba` silently
  downgrades numpy for the whole venv.

## Standalone use (outside PyReduce)

`numba_slitdec.py` imports only `math`, `numpy` and `numba` — no PyReduce
imports at all. Copy the single file next to your script and it works.

```python
# /// script
# dependencies = ["numpy", "numba"]
# ///
import numpy as np
from numba_slitdec import slitdec

nrows, ncols, osample = 25, 512, 8          # slit height, dispersion length
y = np.arange(nrows)[:, None]
x = np.arange(ncols)

ycen = nrows / 2 + 0.4 * np.sin(2 * np.pi * x / ncols)   # trace, absolute rows
spec = 1000 * (1 + 0.3 * np.sin(2 * np.pi * x / 60))
img = spec * np.exp(-0.5 * ((y - ycen) / 4.0) ** 2)      # fake order
img += np.random.default_rng(0).normal(0, 5, img.shape)

slitcurve = np.zeros((ncols, 6))            # d_x = sum_k c[k] * d_y**k
slitcurve[:, 1] = 0.12                      # tilt;  c[0] is ignored
slitcurve[:, 2] = 0.002                     # shear

res = slitdec(
    im=img,                                 # (nrows, ncols), dispersion along x
    pix_unc=np.sqrt(np.abs(img)),           # accepted but unused by the algorithm
    mask=np.ones(img.shape, np.uint8),      # 1 = good, 0 = bad  (charslit convention)
    ycen=ycen,
    slitcurve=slitcurve,                    # (ncols, n) with 1 <= n <= 6
    slitdeltas=np.zeros(nrows),             # per-row extra x-offsets; nrows or ny
    osample=osample,
    lambda_sL=1.0,                          # slit-function smoothing, usually > 0
    lambda_sP=0.0,                          # spectrum smoothing; > 0 forces delta_x >= 1
    maxiter=20,
    kappa=10.0,                             # sigma clip on residuals; 0 disables
)

print(res["spectrum"].shape)                # (ncols,)
print(res["slitfunction"].shape)            # (ny,) with ny = osample*(nrows+1)+1
```

Returns a dict: `spectrum` (ncols), `slitfunction` (ny), `model` (nrows, ncols),
`uncertainty` (ncols), `mask` (updated copy), `info`, `return_code`.

Things that bite:

- **Orientation.** Dispersion must run along x (columns = wavelength, rows =
  slit). PyReduce's `clipnflip()` does this rotation upstream; standalone, do it
  yourself.
- **`ycen` is absolute** row position within the swath, not an offset. It is
  split internally into an integer row shift and a sub-pixel remainder.
- **Mask polarity is charslit's**: 1 = good. The opposite of numpy masked arrays.
- **`slitcurve[:, 0]` is ignored** — the polynomial starts at the linear term.
- **`pix_unc` is accepted and never used.** The C carries a "Should pix_unc
  contribute here?" comment; uncertainties come from data − model. Passing zeros
  changes nothing.
- **`return_code == -1` with `info[2] == -2`** means the curvature implied a
  horizontal span wider than the swath (`nx > ncols`) — usually a bad curvature
  fit, not a real geometry.
- **Edge columns are zeroed**: the outermost `delta_x` columns of `spectrum` and
  `uncertainty` are set to 0, since their support is incomplete. `info[4]` is
  `delta_x`.
- `preset_slitfunc=` skips the slit-function solve entirely and fits only the
  spectrum against the supplied profile (single-pass). Length `nrows` or `ny`.
