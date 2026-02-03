High-level review notes for `tracespec`

Scope
- Review focused on refactor surface changes and potential behavioral risks. Tests are green and some end-to-end runs have been manually verified; the points below are mainly “watch items” or questions worth validating over time.

Key risks / watch items (ordered by severity)
1) Trace ordering vs wavelength mapping
   - Wavecal finalization attaches per-group wavelength polynomials to Trace objects by the order of the grouped trace list. If the trace ordering differs from the wavecal order list (or changes in future), the wrong wavelength polynomial could be attached without errors.
   - Code: `pyreduce/reduce.py` (wavecal save), `pyreduce/trace.py` (`create_trace_objects` ordering)

2) Slit curvature convention alignment
   - `Trace.slit` is stored as (deg_y+1, deg_x+1) and evaluated via `slit_at_x`. Extraction builds a curvature cube by evaluating those coefficients per x. This assumes the curvature fitter outputs the same coefficient orientation. If that orientation flips, extraction will be wrong but not crash.
   - Code: `pyreduce/slit_curve.py`, `pyreduce/trace_model.py`, `pyreduce/extract.py`

3) Wavelength evaluation depends on Trace.m being set correctly
   - Many steps now compute wavelength arrays from `Trace.wlen(x)` rather than a separate `wavecal` output. For 2D polynomials this uses `Trace.m`. If `m` is missing or still sequential (no `obase`), the derived wavelength image can be wrong while still looking plausible.
   - Code: `pyreduce/reduce.py`, `pyreduce/trace_model.py`

4) Fiber selection semantics changed
   - `select_traces_for_step()` now filters by `Trace.group` / `Trace.fiber_idx`, replacing prior grouped array semantics. For `fibers.use = groups`, it now returns all traces with group != 0, and does not enforce per-order stacking. This is a behavioral change relative to merged group traces and could alter multi-fiber order alignment or extraction heights.
   - Code: `pyreduce/trace.py`

5) Public API break: `pyreduce.echelle` removed
   - In-repo usage updated, tests updated, but external code importing `pyreduce.echelle` will break. Consider a small shim module that forwards to `pyreduce.spectra` with a deprecation warning.
   - Code: `pyreduce/spectra.py`, `pyreduce/echelle.py` (deleted)

Other notable changes (lower risk)
- `ScienceExtraction.load()` no longer accepts `files` arg and instead uses `self.files`. This may break custom scripts calling the load method directly.
- `WavelengthCalibrationComb.execute()` now returns coefficients, not the full wavelength array. Internal call sites updated; external callers may need adjustment.
- Several masking/normalization changes (masked array handling in wavecal; NaN masking for spectra) alter behavior but appear reasonable.

Questions / assumptions to validate
- Does `CurvatureModule.execute()` return `fitted_coeffs` in the (y-degree, x-degree) orientation expected by `Trace.slit_at_x()`?
- For multi-fiber per-order instruments, is `linelist.obase` always available to set correct `Trace.m`? If not, 2D polynomial evaluation will be wrong.
- Is the new `groups` selection behavior (exclude group 0) intended for all multi-fiber instruments, or do some rely on previously merged group orderings?

Suggested follow-up checks (optional)
1) Add a regression test that validates ordering consistency between traces and wavecal polynomials (e.g., monotonic order center vs. m).
2) Add a compatibility shim for `pyreduce.echelle` to avoid external breakage.
3) Add a small unit test to confirm curvature coefficient orientation is consistent across fit → store → extract.

Ordering dependency note (added)
- Even if each step preserves input order, several handoffs use positional matching across steps (curvature, wavecal, trace_range slicing, wavelengths_from_traces). This is safe only if the trace list order is stable across all steps. To reduce coupling, consider attaching outputs by explicit identity (m/group/fiber_idx) rather than index where feasible.
