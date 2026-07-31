# A numpy-only slitdec: port and results

Follow-up to `numba_slitdec.md`, which ported `clib/slitdec.c` to Numba and left the
numpy/scipy variant as an open experiment. Motivation for doing it: colleagues want
PyReduce extraction inside a new ESO pipeline whose dependency policy may exclude
numba. Target was to stay within 5x of the C.


> **Re-measured 2026-07-30:** the ratios below were taken against the C as it
> stood when this port was written. `clib/slitdec.c` has since gained 1.1-1.4x
> (post-0.9b2), so the current figures are 1.8-2.1x the C. The analysis is kept as written.

**Outcome: shipped. `pyreduce/numpy_slitdec.py` runs at 1.4–2.2x the C on realistic
swaths, agreeing with it to 2e-14, with identical masks and iteration counts.**
No extra to install: numpy and scipy are already core PyReduce dependencies.

## Where things live

- **Reference**: `pyreduce/clib/slitdec.c`, wrapped by `cwrappers.slitdec`.
- **This port**: `pyreduce/numpy_slitdec.py`, selected with `PYREDUCE_EXTRACTION=numpy`,
  tested by `test/test_numpy_slitdec.py` (9 cases diffed against the C oracle).
- **Numba port**: `pyreduce/numba_slitdec.py`, `PYREDUCE_EXTRACTION=numba`. The
  structural template for this one.
- `numba_old` branch holds the superseded pre-slitdec numba code. Do not consult it.

## Design

Everything geometric is fixed across iterations and built **once per call**, then
collapsed into one dense tensor

```
T[m, j, p] = total zeta weight reaching detector pixel p
             from slit position k0_iy[p] + m and source column k0_x[p] + j
```

with `K = max_p` window width in slit position and `Kx` in source column (8 and 3
on CRIRES-like geometry at osample=6). **Both merge windows are contractions of
`T`**, so nothing per-iteration ever touches the five million individual zeta
entries again:

```
W_sL[m, p] = sum_j T[m, j, p] * sP[k0_x[p] + j]      einsum mjp,jp->mp
W_sP[j, p] = sum_m T[m, j, p] * sL[k0_iy[p] + m]     einsum mjp,mp->jp
```

Pixels are held **permuted into runs of equal `k0`**, which turns the two normal
equation fills from indexed scatters (`np.bincount` on the window base) into
`np.add.reduceat` segment sums — 3.8x faster, and the reason the per-iteration
cost now lands within 10% of the C.

| C construct | numpy equivalent |
|---|---|
| `zeta_tensors` triple loop | vectorised over `(x, y, k)`: `osample+1` subpixels per `(x, y)`, since `iy1`/`iy2` depend on `x` only and both step by `osample` per row |
| three `zeta_add` branches | one uniform pair, `A = (x+ix1, w - frac*w)` and `B = (x+ix2, frac*w)`; `B` is dropped by the `w > 0` test when `delta == 0` |
| per-pixel `z_rng` scan | one `np.minimum.at` for the sP window base `k0_x`; the sL base is then derived analytically (below), width from the shifted keys |
| dense merge window `zw[iy - k0]` | one `np.einsum` contraction of `T` |
| pair fill `arow[n] += um * uv[n]` | per `(m, d)`: one multiply plus `np.add.reduceat` over the `k0` runs |
| `bandsol` | `scipy.linalg.solveh_banded` on the upper bands |
| model triple loop | contract the sP window: `model[p] = sum_j W_sP[j,p] * sP[k0_x[p]+j]` |
| `quick_select_percentile/median` | `np.partition(arr, k)[k]` — same, no interpolation |

Only the upper bands are ever built, since the matrix is symmetric and that is exactly
what `solveh_banded` consumes: the C's mirroring step disappears. `Aup[d, i]` holds
`A[i, i+d]`, so the smoothing penalty and the diagonal floor become two-line slice
operations.

**Only one of the two window bases needs a scatter-min.** Every zeta entry obeys,
exactly in integer arithmetic,

```
src_iy = (yy + ycen_offset[xx] + 1) * osample + k - q[src_x],   k in [0, osample]
```

with `q[x] = floor(ycen[x] * osample)` on absolute `ycen` — which follows from
`iy1_init = -floor(frac(ycen) * osample)` and the row invariant
`y + ycen_offset[x] == yy + ycen_offset[xx]`. So a pixel's smallest `src_iy` is that
first term minus the largest `q` over its source columns, and those all lie inside the
sP window `[k0_x, k0_x + Kx)`. Maximising `q` over the whole window rather than the
contributing subset can only *lower* the base, which is free, and measures the same
`K` on every geometry tried. `k0_iy` therefore costs a rolling max over `ncols` values
plus one gather instead of a second `np.minimum.at` over five million entries.

Details that took measurement rather than reasoning:

- **`dy` must be accumulated, not evaluated in closed form.** The closed form
  `dy0 + y + k*step` drifts up to 6.8e-13 from the C's sequential `+=`/`-=`, enough to
  flip `ix1 = int(delta)` when `delta` lands on an integer — a discrete difference, not
  rounding. `np.cumsum` over the same `±step` sequence in the same order is bit-exact.
- **A window wider than a pixel's own key range is free.** The extra slots stay zero
  and add exactly `0.0`, which is what lets a single uniform `K` replace the C's
  per-pixel `rng`. Rows `k0[p] + m` past the last one can then only receive zero
  slots, so the band slices simply truncate — no padding needed.
- **Every `T[m, j, p]` cell holds at most one zeta entry.** `j` fixes the source
  column and `(p, m)` then fix the source row and subpixel, so the squared weights
  the closing uncertainty pass needs are `sum_m T**2` rather than a second
  five-million-entry scatter.
- **Boolean indexing beats `flatnonzero` + `take` 4:1** when compacting the geometry
  candidates (80 vs 352 µs per chunk array): one sequential scan instead of an index
  array plus four gathers.
- **numpy counting-sorts 16-bit integers.** Grouping the pixels costs 1.3 ms when
  `k0` is cast to `int16` and 12 ms at `int32` or wider, so `_group` narrows when the
  values fit.
- **Uniform `(d, m)` column bincounts beat the flat per-`d` form 2:1**, and sorted
  runs plus `reduceat` beat both by another 3.8x — a `bincount` whose indices repeat
  consecutively serialises on the accumulator, which is why the fill had to be
  grouped rather than merely re-indexed. The closing uncertainty pass is grouped the
  same way, which is why it no longer uses `bincount` either.
- **`a[idx]` beats `np.take(a, idx, out=b)` by 3-4x** (numpy 2.4): 2.0 vs 8.0 ms for
  the 5M-entry pixel remap, 1.8 vs 5.5 ms for the eight per-iteration `sL` gathers.
  The `out=` path does not hit numpy's fast take loop. This was the single largest
  win, and the one that pushed the per-iteration cost below the C's.
- **`T[idx] = zw` beats `np.bincount(idx, weights=zw)` by ~1.6x** and is exact here:
  a cell holds at most one entry, so there is nothing to accumulate. Given `(p, m, j)`
  the source column, row and subpixel are all pinned, and the A/B pair of a subpixel
  can only share a target column when `frac == 0`, where B is dropped anyway.

Masked pixels are multiplied out (`W *= maskf`) rather than skipped, so a heavily
masked frame costs the same as a clean one. The sP window is kept **unmasked** because
the model needs every pixel; the mask is applied to a copy for the fill.

## Speed

CRIRES-like swath 2048x176, osample=6 (npix=360k, ny=1063, 5.02M zeta entries, K=8,
Kx=3, 13.9 entries per pixel), 4 iterations, split by regressing runtime on the
reported iteration count (`maxiter=m` does *not* run `m` iterations, so fitting on
`maxiter` saturates once the run converges):

| | per iteration | once per call | total |
|---|---|---|---|
| `slitdec` (C, CFFI) | 23.0 ms | **37.1 ms** | 130 ms (1.0x) |
| `numba_slitdec` | 25.0 ms (1.09x) | 71.2 ms (1.9x) | 173 ms (**1.34x**) |
| `numpy_slitdec` (first version) | 25.8 ms (1.12x) | 118.1 ms (3.2x) | 223 ms (1.72x) |
| `numpy_slitdec` (now) | **19.6 ms (0.85x)** | 103.3 ms (2.8x) | 182 ms (**1.40x**) |

**Per iteration the numpy backend is now faster than the C**, 0.85x, so the whole
remaining gap is setup — 103 ms against the C's 37, and 57% of the numpy total. Of
that ~103 ms: ~44 ms building the zeta candidate lists, ~16 ms scattering them into
`T`, ~13 ms in the surviving `np.minimum.at` scan and its relative keys, ~3 ms in the
second relative-key gather, ~4 ms grouping, ~3 ms `Tsum`/`Tsq`, the rest index
arithmetic and gathers.

Across swath shapes (`iter` = iterations to convergence, so short runs weight setup
more heavily):

| Swath | osample | iter | C | numba | numpy (before) | numpy (now) |
|---|---|---|---|---|---|---|
| 128x10 | 6 | 3 | 0.4 ms | — | 3.73x | 3.32x |
| 400x10 | 6 | 3 | 1.0 ms | — | 2.71x | 2.21x |
| 400x20 | 6 | 3 | 1.9 ms | — | 2.53x | 2.08x |
| 400x40 | 6 | 3 | 3.7 ms | 1.41x | 2.39x | **1.97x** |
| 400x100 | 6 | 5 | 11.7 ms | 1.33x | 2.07x | **1.66x** |
| 1000x25 | 10 | 3 | 8.4 ms | 1.44x | 2.30x | 1.99x |
| 2048x40 | 6 | 3 | 20.6 ms | 1.33x | 2.25x | 1.86x |
| 2048x176 | 6 | 4 | 131.3 ms | 1.34x | 1.70x | **1.36x** |

Small swaths remain the worst case: with ~50 numpy calls per iteration, per-call
overhead stops being negligible, and setup is a larger share of a short call. At
128x10 — smaller than anything PyReduce actually extracts — it is 3.3x, but the whole
call is 1.3 ms. PyReduce's default `swath_width` is 400, so expect ~2x in practice.

**Measured and rejected**, so nobody spends the afternoon again:

- **int32 for the integer geometry planes** (`iy`, `ix1/ix2`, `xx`, `yy`, `keep`):
  1.03x on `_geometry`, i.e. nothing. The float64 planes carry most of the traffic,
  and an int32 index array is widened to `intp` internally for the `ycen_offset`
  gather anyway.
- **Vectorising `_fill_system` over the band index** (one `(width, npix)` multiply and
  one 2-D `reduceat` per `m`, cutting 44 numpy calls to 16) and **tiling it over
  pixels** so `W` stays in cache: both within noise at every size, 5.3 ms either way
  at 2048x176. The fill is at the bandwidth wall, not the call-overhead wall.
- **Sort-based scatter-min** (`argsort` + `minimum.reduceat`): 158 ms against
  `np.minimum.at`'s 10 ms. Collapsing runs of equal consecutive `pix` first
  (5.0M entries to 0.98M) only gets 10.2 to 9.0 ms.
- **Deriving `k0_x` analytically** the way `k0_iy` now is: the bound needs a sliding
  max over ±`delta_x` columns and inflates `Kx` from 3 to 4-5, which costs more per
  iteration than the 10 ms scatter-min it saves.
- Still true from before: `(npix, K, Kx)` layout for `T` is *slower* to fill (17.9 vs
  15.4 ms) and needs a 20 ms transpose to feed the contraction; geometry chunk size is
  irrelevant (100k to 8M candidates per chunk within 10%).

`_geometry` is ~35 passes over the dense `(nc, nrows, osample+1)` planes at ~23 GB/s
and is the wall; short of restructuring the algorithm it stays where it is.

## Correctness

Worst relative deviation vs the C oracle: **2.3e-14** (numba: 1.3e-14). Identical
masks, identical iteration counts, identical `delta_x` and status codes. Verified on
straight and curved geometry, `lambda_sP > 0`, `kappa = 0`, preset slit function, the
`nx > ncols` bail-out, tilt up to 1.3, negative tilt, and nonzero `slitdeltas`, over 31
shape/option combinations including a sloped `ycen` that makes `ycen_offset` vary
across the swath — the case the derived `k0_iy` has to get right.

The geometry was checked separately and harder: the COO tensor is **bit-exact**
against the numba transliteration of `zeta_tensors` — same entry set, zero weight
deviation — which is what pins down the `dy` accumulation and the collapsed branches.

Full unit suite (742 tests) passes with `PYREDUCE_EXTRACTION=numpy`.

Two places where the numpy version could in principle diverge from the C, neither
observed: `np.sum` is pairwise where the C accumulates sequentially, so `dev` differs
at ~1e-16 relative and a residual sitting that close to `kappa*dev` would flip a mask
pixel; the zeta weights reaching one pixel are summed into `T` before they are scaled
by `sP`/`sL` rather than after, which moves rounding by the same order; and `solveh_banded` is Cholesky where `bandsol` is unpivoted Gaussian
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

What survives setup is `T` at `K * Kx` doubles per detector pixel — 24 on CRIRES-like
geometry at osample=6, i.e. 69 MB for 2048x176, against ~120 MB for the C's zeta
tensor. Peak is higher: the zeta candidate lists (`pix`, `src_x`, `src_iy` as int64
plus `w`, ~32 bytes per entry at 13.9 entries per pixel, ~160 MB here) are live while
`T` is being filled, and are dropped immediately after. The geometry is built in
chunks of ~2M candidates so intermediates stay small. Relevant because extraction is
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
