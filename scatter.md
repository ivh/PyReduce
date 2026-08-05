# Background scatter: what was wrong, what changed

Notes for the HARPSPOL work in #38/#39. Short version: the scatter step had a real
defect that was not HARPSPOL-specific, it is fixed on master, and the HARPSPOL
measurements in #39 were taken with two problems stacked on top of each other. They
are worth repeating before concluding the step is unusable.

## The defect

`BackgroundScatter` fits a 2D polynomial to the inter-order pixels of its own input
frames. Every instrument's `id_scatter` points at a flat-type frame (HARPSPOL:
`LAMP,LAMP,TUN`; UVES: `LAMP,ORDERDEF`; XSHOOTER: `LAMP,FLAT`; and so on) — so the
coefficients are in the flux units of a *lamp* frame, and `combine_frames` **sums**
its inputs, so they carry `N_frames x exposure` of lamp light.

`NormalizeFlatField` subtracts that model from the same master flat it was measured
on, which is self-consistent and was always correct.

`ScienceExtraction` subtracted the same coefficients from a science frame **without
any rescaling**. Nothing between `np.savez(..., scatter=coeff)` and `polyval2d` in
`calc_scatter_correction` consulted an exposure time or a file count. The subtracted
level therefore had no defined relationship to the frame it was subtracted from.

Two things make this worse than it sounds:

- The sign and size are not predictable from exposure time. UVES fits on a *7.2 s*
  `LAMP,ORDERDEF` frame and still over-subtracts by 10x against a 60 s science frame,
  because the order-definition lamp is bright. Lamp brightness dominates, not exposure.
- The damage is partly hidden. Optimal extraction absorbs some of a large constant
  offset into the slit function, so products come out looking plausible while being
  wrong by an order-dependent factor. On XSHOOTER the same frame gave anywhere from
  "30% flux loss" to "93% of points negative" depending on whether the bad-pixel mask
  was applied.

## The fix (on master)

Scattered light scales with the illumination of a frame, so it is now **measured on
the frame it corrects**. No rescaling factor, no exposure bookkeeping.

- `ScatterModel` (in `estimate_background_scatter.py`) carries the coefficients, the
  fit parameters, and a note of which frame it was measured on, plus `.refit(img, traces)`.
- `ScienceExtraction` re-estimates on each calibrated science frame, masking **all**
  traces rather than only the selected/`trace_range` subset.
- `NormalizeFlatField` re-estimates on the master flat it is normalizing (same answer
  as before — it was already self-consistent).
- `extract()` / `extract_normalize()` refit if handed a `ScatterModel`, as a safety net
  for direct callers.

`.scatter.npz` is unchanged on disk and old files still load. A bare coefficient array
is still accepted and used unchanged, with a warning that its scale is unknown.
No reduction setting controls any of this.

## Parameters cannot substitute for it

Worth stating explicitly, because it was the obvious first question. For four
instruments I swept 18 fits — `scatter_degree` in {2, 4, 6} x `extraction_height` in
{0.2 ... 1.0} of order spacing — fitting once on the calibration frames and once on the
science frame, and compared each model against a fit-independent reference: the median
of mid-gap pixels, further than 0.4 x order spacing from every trace.

Best result reachable anywhere in the grid:

| | fit on calibration frames | fit on the science frame |
|---|---|---|
| XSHOOTER nir | 24.1x | 0.97x |
| HARPS red | 30.8x | 1.01x |
| UVES middle | 5.0x | 1.01x |
| LICK_APF | 82.4x | 1.00x |

No masking width or polynomial degree gets the old method below ~5x, and on XSHOOTER
never below 24x — masking does not change flux scale. Fitting on the right frame
reaches ~1.0 everywhere.

## Parameters *do* set the residual accuracy — and several were wrong

Once the fit is on the right frame, what is left is a masking problem: if
`extraction_height` is narrower than the order footprint, order wings leak into the fit
and bias it high. HARPS shipped 20 px against **77 px** order spacing (0.26 of the
spacing) and consequently over-estimated by 6.1x even after the method fix — briefly
*worse* than before. Changed on master:

| instrument | was | now | model/truth after |
|---|---|---|---|
| HARPS | `extraction_height: 20` (0.26 of spacing) | `0.9` | 6.14x -> 1.33x |
| LICK_APF | `extraction_height: 18`, degree 4 | `0.6`, `scatter_degree: 2` | 0.48x -> 1.20x |
| UVES | `extraction_height: 0.9` | `1.0` | 1.23x -> 1.04x |
| XSHOOTER | `0.9` | unchanged | 0.92x |

LICK_APF needed the degree drop as well: at 22 px order spacing there are too few
mid-gap pixels for a degree-4 surface, which overfitted to 0.48x.

## End-to-end, before vs after

One science frame per instrument, extracted three ways. "Before" is the old method at
the then-shipped parameters; "after" is the new method at the new parameters. Reference
is the mid-gap floor above.

| | floor | model before | after | before/floor | after/floor |
|---|---|---|---|---|---|
| XSHOOTER nir | 214.1 | 7308.8 | 195.9 | 34.1x | **0.92x** |
| HARPS red | 6.00 | 25.0 | 8.0 | 4.2x | **1.33x** |
| UVES middle | 2.60 | 27.5 | 2.7 | 10.6x | **1.04x** |
| LICK_APF | 1.00 | 31.5 | 1.2 | 31.5x | **1.20x** |

Median extracted flux, and the fraction of negative points:

| | no correction | before | after | negative before | after |
|---|---|---|---|---|---|
| XSHOOTER nir | 13354 | **-617820** | 2756 | 93.3% | 1.6% |
| HARPS red | 17505 | 16941 | 16753 | 0.0% | 0.0% |
| UVES middle | 4298 | 3220 | 4138 | 7.1% | 0.9% |
| LICK_APF | 507 | **-1773** | 472 | 100.0% | 2.9% |

On LICK_APF, before the fix, *every point in all 83 orders* came out negative.

## What this means for HARPSPOL

The `scatter` block and the `scatter` entry in `get_expected_values` were removed in
#38, which was the right call on the evidence available at the time and is not being
reverted here. But the #39 measurements were taken with:

1. the model fitted on `LAMP,LAMP,TUN` and applied to science frames unscaled, and
2. `extraction_height: 20` — HARPSPOL is dual-beam, so its order spacing is roughly
   half of HARPS's already-mismatched 77 px, and 20 px is very likely still in the
   wing-contaminated regime.

Both are now addressed in the code and in the HARPS settings respectively. So the
conclusion "scatter removes most of the stellar flux at every `extraction_height`
tested" was measured on a code path that no longer exists, and at heights chosen
against the wrong reference.

Worth re-running on the 2015-06 month before deciding, and if it is re-enabled, setting
`extraction_height` from the measured A+B trace separation rather than by hand. The
useful diagnostic is the one used above: fit the model, evaluate it on mid-gap pixels
(>0.4 x order spacing from any trace), and check the ratio to the frame's own median
there is near 1. That catches both failure modes in one number and needs no extraction.

## Still open

- The polynomial is not constrained positive. On XSHOOTER the old flat-derived surface
  ranged -17098 to +23601 across the detector, i.e. it *added* flux in places. The
  per-frame fits are better behaved (the XSHOOTER refit spans +56 to +296) but nothing
  enforces it.
- `simple` extraction ignores `scatter` entirely. Pre-existing, unrelated to this.
- The optimum `extraction_height` will shift somewhat with target brightness; the values
  above come from one frame per instrument.
