# LATER

Deferred side-quests from the 2026-10-23 slitdec subpixel-boundary fix.
Numbers are from a sigma=2 px Gaussian, flat trace, Poisson noise, 10 seeds.

- **slitdec `sig` runs ~15% low.** Reported uncertainty is 0.81-0.91 x sqrt(F)
  while the actual scatter is 1.04-1.10 x sqrt(F), at 1e2-1e6 counts, os=4 and 10.
- **os=1 flux bias.** Constant +0.57% at every flux level for a trace at a
  half pixel; `sig` then grows with flux (9x sqrt(F) at 1e6). Likely the
  one-subpixel slit function not representing a half-pixel-offset profile,
  but unconfirmed. os=4 shows +2e-4 at 1e6, os=10 none.
- **Jump when ycen crosses an integer.** `ycen_offset` steps and the slit grid
  re-anchors by one row against the swath edge: 3e-4 (CharSlit fixedslope) to
  18% (Hsim) between ycen = n - 1e-11 and n + 1e-11. Present before and after
  the d1 fix.
- **Rejection threshold is `kappa * RMS(raw residuals)`**, not per-pixel
  sigma-normalized, so at high SNR the bright core is clipped first. On a
  noiseless frame `dev` is pure round-off; a floor on `dev` was discussed and
  deferred.

Deferred from the 2026-11-23 SOXS instrument setup (examples/soxs_example.py,
examples/soxs_compare_soxspipe.py, public commissioning data):

- **slitdec `sig` too large on SOXS NIR A-B frames.** Scatter between the two
  independent nod pairs is 0.6-0.8 x the reported uncertainty in every order.
  The VIS 5" A vs B test gives 1.1-1.6 x, but that one also carries flat-field
  structure from different detector positions. Check gain/RON handling of the
  NDIT-averaged NIR frames vs the slitdec noise model.
- **SOXS wavelength zero point.** Arc-only solution is off by -15 to -28 km/s
  (VIS, ~1 px) and ~-5 km/s (NIR) on sky lines; soxspipe tunes per order to sky
  lines. A sky-line shift step (or per-order offset from the science frame) would fix it.
- **SOXS NIR order 10 wavecal** keeps only 6 of 12 lines (degree 4, 5000 m/s);
  +18 km/s on sky lines.
