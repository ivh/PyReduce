# ANDES Instrument Configuration Design

## Background

### ANDES Instrument Overview

ANDES is a future ELT high-resolution spectrograph with multiple spectrographs, each containing multiple arms:

| Spectrograph | Arms | Detector Type |
|--------------|------|---------------|
| 1 | U, B, V | Optical CCD |
| 2 | R, IZ | Red optical CCD |
| 3 | Y, J, H | Near-IR (H2RG or similar) |
| 4 | K | Thermal IR |

Key characteristics:
- All arms fed from the same fibers on sky
- Each arm has its own independent detector
- FITS files organized **per spectrograph** (one file with arms as extensions)
- Detectors are similar within each spectrograph

### Terminology

| ANDES term | PyReduce term |
|------------|---------------|
| Spectrograph | Instrument |
| Arm | Channel |

### Current State

- `ANDES_YJH/` - Working instrument for near-IR spectrograph (Y, J, H arms)
- Old `ANDES/` directory removed (was empty)

---

## PyReduce Architecture Context

### How Channels Work

Channels in PyReduce handle multi-extension FITS files. The `getter` class in `instruments/common.py` automatically indexes list values by channel position:

```yaml
# Example: UVES with 3 channels
channels: [BLUE, MIDDLE, RED]
extension: [0, 2, 1]        # BLUE→ext0, MIDDLE→ext2, RED→ext1
orientation: [2, 1, 1]      # BLUE rotated differently
```

When processing channel "RED" (index 2), PyReduce automatically uses `extension[2]=1`, `orientation[2]=1`.

### Per-Channel Settings

PyReduce supports per-channel settings files. When loading configuration for a channel, it looks for `settings_{channel}.json` first, falling back to `settings.json`.

Example from MOSAIC:
```
MOSAIC/
├── settings.json           # base: __inherits__: "defaults"
├── settings_nir.json       # __inherits__: "MOSAIC", NIR-specific tuning
├── settings_VIS1.json      # __inherits__: "MOSAIC", VIS1-specific tuning
└── ...
```

This allows different arms to have different reduction parameters while sharing a single instrument config.

### Channel-Based File Filtering

Files can be filtered by channel using `kw_channel` and `id_channel`:

```yaml
# From ANDES_YJH config
kw_channel: BAND
id_channel: ["^Y$", "^J$", "^H$"]
```

When reducing channel `J`, only files with `BAND=J` header are selected.

---

## Design Options

### Option A: Single ANDES Instrument (9 channels)

```yaml
# andes/config.yaml
channels: [U, B, V, R, IZ, Y, J, H, K]
extension: [1, 2, 3, 1, 2, 1, 2, 3, 1]
kw_channel: BAND
id_channel: ["^U$", "^B$", "^V$", "^R$", "^IZ$", "^Y$", "^J$", "^H$", "^K$"]
```

With per-channel settings:
```
ANDES/
├── config.yaml
├── settings.json           # base settings
├── settings_U.json         # optical CCD tuning
├── settings_B.json
├── settings_V.json
├── settings_R.json
├── settings_IZ.json
├── settings_Y.json         # near-IR tuning
├── settings_J.json
├── settings_H.json
└── settings_K.json         # thermal IR tuning
```

| Pros | Cons |
|------|------|
| Single instrument, unified management | Larger config directory |
| Can reduce any/all arms in one command | Requires consistent BAND header across spectrographs |
| Conceptually cleaner (ANDES is one instrument) | |
| Per-arm tuning via settings files | |

**Requirements**: All ANDES FITS files must have a `BAND` header (or equivalent) identifying the arm.

### Option B: Per-Spectrograph Instruments (4 instruments)

```
ANDES_UBV/   channels=[U,B,V]   extension=[1,2,3]
ANDES_RIZ/   channels=[R,IZ]    extension=[1,2]
ANDES_YJH/   channels=[Y,J,H]   extension=[1,2,3]
ANDES_K/     channels=[K]       extension=1
```

| Pros | Cons |
|------|------|
| Smaller, focused config files | 4 instruments to manage |
| Natural file separation (each spectrograph = separate FITS) | Cannot reduce "all ANDES" in one command |
| Independent development per spectrograph | |
| Already working for ANDES_YJH | |

---

## Recommendation

**Option A (single instrument)** is now viable and cleaner, provided FITS headers are consistent across spectrographs.

**Option B (per-spectrograph)** remains a practical alternative if:
- Different spectrographs have incompatible header conventions
- Independent development of each spectrograph config is preferred
- ANDES_YJH is already working and migration isn't worth the effort

### For ANDES_YJH Now

The current setup works. If Y/J/H need different tuning, add per-channel settings:
```
ANDES_YJH/
├── settings.json       # shared base
├── settings_Y.json     # __inherits__: "ANDES_YJH"
├── settings_J.json
└── settings_H.json
```

### Future Migration to Single Instrument

When other spectrographs (UBV, RIZ, K) are needed:
1. Verify all use consistent `BAND` header convention
2. If yes: create single `ANDES/` instrument with all 9 channels
3. If no: create additional per-spectrograph instruments (ANDES_UBV, etc.)

---

## File Filtering Details

For a single ANDES instrument to work, file classification must correctly select files by arm:

1. **Channel identification**: `kw_channel: BAND` with `id_channel` patterns
2. **File type identification**: `kw_flat`, `kw_wave`, etc. must work consistently across spectrographs

If spectrographs use different header conventions for file types, separate instruments are cleaner.

---

## Summary

| Approach | Per-arm tuning | File filtering | Complexity |
|----------|----------------|----------------|------------|
| Single ANDES | Via settings_*.json | Via kw_channel/id_channel | One instrument |
| Per-spectrograph | Via settings_*.json | Natural (separate FITS) | Multiple instruments |

Both approaches now support per-arm tuning via per-channel settings files. The choice depends on FITS header consistency and organizational preference.
