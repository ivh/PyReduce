# LATER

Deferred side-quests from the 2026-10-23 slitdec subpixel-boundary fix.
Numbers are from a sigma=2 px Gaussian, flat trace, Poisson noise, 10 seeds.

- ~~**slitdec `sig` runs ~15% low.**~~ That was the residual estimate; the default
  is now `extraction_uncertainty: "propagated"` (2026-11-25), calibrated to 10%.
- **os=1 flux bias.** Constant +0.57% at every flux level for a trace at a
  half pixel; `sig` then grows with flux (9x sqrt(F) at 1e6). Likely the
  one-subpixel slit function not representing a half-pixel-offset profile,
  but unconfirmed. os=4 shows +2e-4 at 1e6, os=10 none.
- **Jump when ycen crosses an integer.** `ycen_offset` steps and the slit grid
  re-anchors by one row against the swath edge: 3e-4 (SlitDec fixedslope) to
  18% (Hsim) between ycen = n - 1e-11 and n + 1e-11. Present before and after
  the d1 fix.
- **Rejection threshold is `kappa * RMS(raw residuals)`**, not per-pixel
  sigma-normalized, so at high SNR the bright core is clipped first. On a
  noiseless frame `dev` is pure round-off; a floor on `dev` was discussed and
  deferred.

Deferred from the 2026-11-23 SOXS instrument setup (examples/soxs_example.py,
examples/soxs_compare_soxspipe.py, public commissioning data):

- ~~**slitdec `sig` too large on SOXS NIR A-B frames.**~~ The residual estimate
  (1.6-1.9x the photon limit); "propagated" gives 0.9-1.0x (2026-11-25).
- **Per-pixel noise 1.15-1.35x the photon limit on SOXS NIR nod** (all weights),
  1.0-1.2x once binned to >= 0.4 nm, with ~10-15% excess power near Nyquist:
  neighbouring spectral pixels anticorrelated under the tilted slit with
  `smooth_spectrum = 0`. Try a small `smooth_spectrum` together with an LSF
  check; the C's covariance output (`nlag`) would quantify it.
- ~~**`pix_unc` is rebuilt from the image being extracted**, so an A-B difference
  misses the subtracted frame's read noise and background.~~ Optional `var`
  input / `VAR` extension (2026-11-26); SOXS NIR nod 1.27x -> 1.00x of the
  pair-to-pair scatter. Still open: without `var`, the scattered-light
  background subtracted in the science step is missing from the variances too.
- **`extraction_weights: "flux"` fails on A-B frames** (6-13x the photon
  limit on SOXS NIR): 1/slit function blows up in the near-empty wings.
- **SOXS wavelength zero point.** Arc-only solution is off by -15 to -28 km/s
  (VIS, ~1 px) and ~-5 km/s (NIR) on sky lines; soxspipe tunes per order to sky
  lines. A sky-line shift step (or per-order offset from the science frame) would fix it.
- **SOXS NIR order 10 wavecal** keeps only 6 of 12 lines (degree 4, 5000 m/s);
  +18 km/s on sky lines.
