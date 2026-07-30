# A numpy-only slitdec: port and results

Follow-up to `numba_slitdec.md`, which ported `clib/slitdec.c` to Numba and left the
numpy/scipy variant as an open experiment. Motivation for doing it: colleagues want
PyReduce extraction inside a new ESO pipeline whose dependency policy may exclude
numba. Target was to stay within 5x of the C.

**Outcome: shipped. `pyreduce/numpy_slitdec.py` runs at 3.0–4.2x the C (~3.5x
typical), agreeing with it to 1.4e-14, with identical masks and iteration counts.**
No extra to install: numpy and scipy are already core PyReduce dependencies.

## Where things live

- **Reference**: `pyreduce/clib/slitdec.c`, wrapped by `cwrappers.slitdec`.
- **This port**: `pyreduce/numpy_slitdec.py`, selected with `PYREDUCE_USE_NUMPY=1`,
  tested by `test/test_numpy_slitdec.py` (9 cases diffed against the C oracle).
- **Numba port**: `pyreduce/numba_slitdec.py`, `PYREDUCE_USE_NUMBA=1`. The
  structural template for this one.
- `numba_old` branch holds the superseded pre-slitdec numba code. Do not consult it.

## Design

Everything geometric is fixed across iterations and built **once per call**. The zeta
tensor is held in flat COO form — `pix`, `src_x`, `src_iy`, `w` — with
`pix = xx * nrows + yy`, i.e. images are transposed internally so the flat pixel index
runs along the slit first, matching the C's zeta layout. Consecutive COO entries then
hit consecutive pixels and the per-iteration scatters stay cache friendly.

| C construct | numpy equivalent |
|---|---|
| `zeta_tensors` triple loop | vectorised over `(x, y, k)`: `osample+1` subpixels per `(x, y)`, since `iy1`/`iy2` depend on `x` only and both step by `osample` per row |
| three `zeta_add` branches | one uniform pair, `A = (x+ix1, w - frac*w)` and `B = (x+ix2, frac*w)`; `B` is dropped by the `w > 0` test when `delta == 0` |
| per-pixel `z_rng` scan | `np.minimum.at` for the window base `k0`, width from the shifted keys |
| dense merge window `zw[iy - k0]` | one `np.bincount` into a `(K, npix)` array |
| pair fill `arow[n] += um * uv[n]` | per `(m, d)`: one multiply plus `np.bincount(k0, ...)`, shifted into the band |
| `bandsol` | `scipy.linalg.solveh_banded` on the upper bands |
| model triple loop | contract the sP window: `model[p] = sum_m W[m,p] * sP[k0_x[p]+m]` |
| `quick_select_percentile/median` | `np.partition(arr, k)[k]` — same, no interpolation |

Only the upper bands are ever built, since the matrix is symmetric and that is exactly
what `solveh_banded` consumes: the C's mirroring step disappears. `Aup[i, d]` holds
`A[i, i+d]`, so the smoothing penalty and the diagonal floor become two-line slice
operations.

Three details that took measurement rather than reasoning:

- **`dy` must be accumulated, not evaluated in closed form.** The closed form
  `dy0 + y + k*step` drifts up to 6.8e-13 from the C's sequential `+=`/`-=`, enough to
  flip `ix1 = int(delta)` when `delta` lands on an integer — a discrete difference, not
  rounding. `np.cumsum` over the same `±step` sequence in the same order is bit-exact.
- **A window wider than a pixel's own key range is free.** The extra slots stay zero
  and add exactly `0.0`, which is what lets a single uniform `K` replace the C's
  per-pixel `rng`. Rows `k0[p] + m` past the last one can then only receive zero
  slots, so the band slices simply truncate — no padding needed.
- **Uniform `(d, m)` column bincounts beat the flat per-`d` form 2:1** (14.9 vs
  29.2 ms for 36 pairs), because the output stays in L1 and the index array is a
  single reused `k0`. The original plan proposed the flat form with precomputed
  row-index arrays; that also wanted ~100 MB of index memory.

Masked pixels are multiplied out (`W *= maskf`) rather than skipped, so a heavily
masked frame costs the same as a clean one. The sP window is kept **unmasked** because
the model needs every pixel; the mask is applied to a copy for the fill.

## Speed

CRIRES-like swath 2048x176, osample=6 (npix=360k, ny=1183, 5.02M zeta entries, K=8,
Kx=3, 14.0 entries per pixel), 4 iterations, timed by fitting 2- and 4-iteration runs:

| | per iteration | once per call | total |
|---|---|---|---|
| `slitdec` (C, CFFI) | **22.9 ms** | 36.2 ms | 128 ms (1.0x) |
| `numba_slitdec` | 25.6 ms (1.1x) | 67.3 ms (1.9x) | 170 ms (**1.33x**) |
| `numpy_slitdec` | 55.1 ms (2.4x) | 168.7 ms (4.7x) | 389 ms (**3.04x**) |

**The plan's ~2.2x projection was right about the iteration and silent about the
rest.** Per-iteration comes in at 2.4x; the once-per-call work — geometry, window
bases, the closing uncertainty pass — is 4.7x the C's and is 43% of the numpy total.
Per iteration, ~26 ms goes to the two SLE fills, ~22 ms to the two 5M-entry merge
scatters, and ~7 ms to the gathers, model contraction and rejection.

Across swath shapes:

| Swath | osample | C | numba | numpy |
|---|---|---|---|---|
| 400x40 | 6 | 3.7 ms | 1.36x | **4.17x** |
| 800x40 | 6 | 7.8 ms | 1.26x | 3.83x |
| 400x100 | 6 | 11.7 ms | 1.37x | 3.61x |
| 2048x40 | 6 | 19.8 ms | 1.33x | 3.89x |
| 2048x176 | 6 | 130.7 ms | 1.31x | **3.00x** |
| 1000x25 | 10 | 10.3 ms | 1.41x | 4.13x |
| 4096x25 | 10 | 51.4 ms | 1.28x | 3.54x |

Small swaths are the worst case: with ~50 numpy calls per iteration, per-call overhead
stops being negligible. PyReduce's default `swath_width` is 400, so expect the 4x end
in practice. Through `extract_spectrum` (6 orders x 7 swaths of 400 columns): C 0.21 s,
numba 0.35 s, numpy 0.70 s.

**Known headroom, not taken:** the pair-fill bincounts are 3.7x faster as
`np.add.reduceat` segment sums (4.1 vs 15.4 ms for 36 pairs), but that requires the
pixel axis sorted by `k0` — two different permutations, threaded through every
per-pixel array. Worth ~10% overall, at the cost of the module's auditability. The
same applies to skipping the COO compaction in `_geometry` (~34 ms of the setup) by
routing dropped candidates to a dummy pixel with zero weight.

## Correctness

Worst relative deviation vs the C oracle: **1.4e-14** (numba: 1.3e-14). Identical
masks, identical iteration counts, identical `delta_x` and status codes. Verified on
straight and curved geometry, `lambda_sP > 0`, `kappa = 0`, preset slit function, the
`nx > ncols` bail-out, tilt up to 1.3, negative tilt, and nonzero `slitdeltas`.

The geometry was checked separately and harder: the COO tensor is **bit-exact**
against the numba transliteration of `zeta_tensors` — same entry set, zero weight
deviation — which is what pins down the `dy` accumulation and the collapsed branches.

Full unit suite (742 tests) passes with `PYREDUCE_USE_NUMPY=1`.

Two places where the numpy version could in principle diverge from the C, neither
observed: `np.sum` is pairwise where the C accumulates sequentially, so `dev` differs
at ~1e-16 relative and a residual sitting that close to `kappa*dev` would flip a mask
pixel; and `solveh_banded` is Cholesky where `bandsol` is unpivoted Gaussian
elimination. `test/test_numpy_slitdec.py` therefore keeps exact assertions on the mask
and iteration count deliberately — if either ever starts failing, this is why.

`clib/slitdec.c` stays the reference: **port changes forward, don't let them drift.**

## Two deliberate differences from the C

- **Non-positive-definite matrices.** `bandsol` has no pivoting and no singularity
  check; on a degenerate system it divides by zero and propagates inf/nan. Here
  `solveh_banded` raises and the code falls back to a pivoted LU (`solve_banded`) on
  the mirrored band. Better answer, not just a different one.
- **Over-wide merge windows.** When a pixel's key span exceeds `2*osample+1`, the C
  switches to a key-search fallback that can write past the end of its band array.
  Here the band simply widens, which is the correct normal-equations solve. Not
  reachable with realistic geometry (K=8 vs the cap of 13 on CRIRES-like curvature).

## Memory

~50 bytes per zeta entry, i.e. ~700 bytes per detector pixel at 14 entries each:
`pix`, `src_x`, `src_iy` and the two merge-bin index arrays as int64 plus `w` as
float64. For 2048x176 at osample=6 that is ~240 MB, against ~120 MB for the C's zeta
tensor. The geometry is built in chunks of ~2M candidates so intermediates stay small,
but the COO arrays themselves are allocated whole. Relevant because extraction is
parallelised over orders (`n_jobs`): peak memory scales with worker count.

## Standalone use (outside PyReduce)

`numpy_slitdec.py` imports only `numpy` and `scipy.linalg` — no PyReduce imports at
all. Copy the single file next to your script and it works; verified in a clean venv
with nothing but numpy and scipy installed.

```python
# /// script
# dependencies = ["numpy", "scipy"]
# ///
import numpy as np
from numpy_slitdec import slitdec

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
    ycen=ycen,                              # absolute row position, not an offset
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

Things that bite (identical to the numba backend — same signature, same conventions):

- **Orientation.** Dispersion must run along x (columns = wavelength, rows = slit).
  PyReduce's `clipnflip()` does this rotation upstream; standalone, do it yourself.
- **`ycen` is absolute** row position within the swath, not an offset. It is split
  internally into an integer row shift and a sub-pixel remainder.
- **Mask polarity is charslit's**: 1 = good. The opposite of numpy masked arrays.
- **`slitcurve[:, 0]` is ignored** — the polynomial starts at the linear term.
- **`pix_unc` is accepted and never used.** The C carries a "Should pix_unc contribute
  here?" comment; uncertainties come from data - model. Passing zeros changes nothing.
- **`return_code == -1` with `info[2] == -2`** means the curvature implied a horizontal
  span wider than the swath (`nx > ncols`) — usually a bad curvature fit.
- **Edge columns are zeroed**: the outermost `delta_x` columns of `spectrum` and
  `uncertainty` are set to 0, since their support is incomplete. `info[4]` is
  `delta_x`, `info[3]` the iteration count.
- **`ycen` and `mask` are not mutated** — unlike the C, this wrapper copies both.
- `preset_slitfunc=` skips the slit-function solve entirely and fits only the spectrum
  against the supplied profile (single-pass). Length `nrows` or `ny`.
