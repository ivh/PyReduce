# Janus Fork Integration

Summary of changes from https://github.com/janusbrink/PyReduce integrated into this branch.

## Fork Status

- **Fork point:** `ebcd35c` (PR #33 "Return final linelist with flags")
- **Master ahead by:** ~300 commits
- **Instruments affected:** METIS_IFU, METIS_LSS, MICADO

## Changes Applied

Config and settings updates for ELT instruments based on Janus's testing with simulated data.

### METIS_IFU

**config.yaml:**
- `instrument_mode`: added `HIERARCH` prefix
- `id_bias`: `DARK` → `IFU_WCU_OFF_RAW`
- `id_flat`, `id_orders`, `id_scatter`: `RSRF` → `IFU_RSRF_PINH_RAW`
- `id_curvature`, `id_wave`: `WAVE` → `SKY`

**settings.json:**
- `trace.degree`: 4 → 3
- `trace.min_cluster`: 100 → 1000
- `trace.border_width`: 6 → 5
- `norm_flat.extraction_height`: 0.28 → 7 (pixels)
- `norm_flat.threshold`: 1000 → 100
- `curvature.curve_degree`: 2 → 1
- `curvature.curve_height`: 1.54 → 14
- `curvature.extraction_height`: 1 → 7
- Various peak detection parameters tuned

### METIS_LSS

**config.yaml:**
- `id_instrument`: `METIS_LSS` → `METIS` (match FITS keyword)

### MICADO

**config.yaml:**
- `channels`: `NIR` → `SPEC`
- `instrument_mode`: `ESO SEQ ARM` → `ESO DPR TECH`
- `id_flat`: `SFLATSLIT` → `SFLAT`
- `id_orders`: `SFLAT_PINH` → `PINH`

**__init__.py:**
- `get_extension()`: 5 → 4

**settings.json:**
- `trace.degree`: 5 → 3
- `norm_flat`: heavily tuned (extraction_height: 0.28 → 600, smooth_slitfunction: 1000 → 1, added maxiter)
- `curvature`: switched from 2D to 1D, various parameter changes

## Changes NOT Applied

Janus made code changes to handle out-of-bounds array indexing. These are **not needed** in the current codebase.

### What Janus changed

1. **`util.py` - `make_index()`**: Added `imdim` parameter to return a mask and clip out-of-bounds indices
2. **`extract.py`**: Used the mask to handle edge pixels
3. **`make_shear.py`** (now `slit_curve.py`): Same boundary handling
4. **`reduce.py`**: Added `ordr` to NormalizeFlatField output

### Why not needed

Janus's fork passed `ignore_column_range=True` to `fix_parameters()`, which bypasses the safety mechanism in `fix_column_range()`. This required the mask-based boundary handling.

The current codebase:
- Never uses `ignore_column_range=True`
- `fix_column_range()` at `extract.py:848` already ensures `(y_bot >= 0) & (y_top < nrow)`
- Similar bounds checks exist in `slit_curve.py:234` and `estimate_background_scatter.py:86`

The boundary protection is already comprehensive through `fix_column_range()`.

## How to Update Your Fork

Your fork is ~300 commits behind and the codebase has restructured significantly (JSON→YAML configs, renamed files). A clean reset is recommended:

```bash
git remote add tom https://github.com/ivh/PyReduce.git
git fetch tom janus
git checkout master
git reset --hard tom/janus
git push --force
```

This preserves your config changes while getting the current codebase.
