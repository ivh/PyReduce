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
