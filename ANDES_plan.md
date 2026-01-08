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

### Current State

- `andes.yaml` - Multi-channel IR-focused config
- `settings_ANDES.json` - Inherits from CRIRES_PLUS (IR-tuned)
- `aj.yaml` / `settings_AJ.json` - Simulation instrument (draft for testing)

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

### What MUST Be Shared Across Channels

1. **File classification keywords** (`kw_bias`, `kw_flat`, `kw_wave`, `kw_spec`)
2. **File classification patterns** (`id_bias`, `id_flat`, etc.)
3. **Reduction settings** - `settings_INSTRUMENT.json` applies to ALL channels
4. **Header keyword mappings** (date, target, exposure_time, etc.)

### What CAN Vary Per Channel

1. FITS extension index
2. Image orientation
3. Detector-specific header values (gain, readnoise) via `{id[n]}` templates
4. Output filenames (mask, wavecal)

### Critical Limitation: No Per-Channel Settings

**YAML instrument configs** support per-channel values via list indexing.

**JSON settings do NOT** - all channels share identical reduction parameters:
- Same `trace.degree`, `trace.noise`, `trace.min_cluster`
- Same `science.extraction_height`, `science.oversampling`
- Same `wavecal.degree`, `wavecal.threshold`

This means optical CCDs and IR arrays would share the same tuning if in one instrument.

---

## Design Options

### Option A: Single ANDES Instrument (9 channels)

```yaml
# andes.yaml
channels: [U, B, V, R, IZ, Y, J, H, K]
extension: [1, 2, 3, 1, 2, 1, 2, 3, 1]
```

| Pros | Cons |
|------|------|
| Single abstraction level | Cannot tune settings per arm/spectrograph |
| Simple mental model | Must handle 4 FITS file types in one instrument |
| One config file | Optical/IR forced to share parameters |

**Implementation**: Would follow CRIRES+ pattern (see below).

### Option A Implementation: CRIRES+ Pattern

CRIRES+ shows how to handle multiple file types with a single instrument:

```python
class ANDES(Instrument):
    def __init__(self):
        super().__init__()
        # Add spectrograph filter - files tagged by which spectrograph they're from
        self.filters["spectrograph"] = Filter(self.info["kw_spectrograph"])
        self.shared += ["spectrograph"]

    def get_expected_values(self, target, night, channel):
        expectations = super().get_expected_values(target, night)
        # Map channel to spectrograph: J → YJH, U → UBV, etc.
        spectrograph = self.channel_to_spectrograph(channel)
        for key in expectations.keys():
            expectations[key]["spectrograph"] = spectrograph
        return expectations

    def get_extension(self, header, channel):
        # Map channel to extension: U→1, B→2, V→3, Y→1, J→2, H→3, etc.
        return self.channel_extensions[channel]

    def discover_channels(self, input_dir):
        # Scan files, find which spectrographs present, return available channels
        ...
```

**How file filtering works:**
1. Files have header keyword identifying spectrograph (UBV, RIZ, YJH, K)
2. When `channel=J` requested, ANDES maps J → YJH spectrograph
3. File filter selects only YJH spectrograph files
4. Extension 2 read from those files

**Without specifying channel:**
- `discover_channels()` scans files to find which spectrographs exist
- Returns available channels (e.g., if only YJH files exist → `[Y, J, H]`)
- Pipeline loops over each discovered channel

This makes Option A fully viable following an existing pattern.

### Option B: Separate Instruments (9 instruments, 1 channel each)

```
ANDES_U   channels=[U]   extension=1   → inherits pyreduce
ANDES_B   channels=[B]   extension=2   → inherits ANDES_U
ANDES_V   channels=[V]   extension=3   → inherits ANDES_U
ANDES_R   channels=[R]   extension=1   → inherits pyreduce
ANDES_IZ  channels=[IZ]  extension=2   → inherits ANDES_R
ANDES_Y   channels=[Y]   extension=1   → inherits CRIRES_PLUS
ANDES_J   channels=[J]   extension=2   → inherits ANDES_Y
ANDES_H   channels=[H]   extension=3   → inherits ANDES_Y
ANDES_K   channels=[K]   extension=1   → inherits CRIRES_PLUS
```

| Pros | Cons |
|------|------|
| Single abstraction (instruments only) | 9 YAML + 9 settings files |
| Maximum per-arm flexibility | File sorting overlap (U,B,V find same file) |
| Settings inheritance keeps configs DRY | Cannot reduce "all of UBV" in one command |
| Matches AJ pattern | More instruments to manage |

### Option C: Per-Spectrograph Instruments (4 instruments)

```
ANDES_UBV  channels=[U,B,V]   extension=[1,2,3]   → inherits pyreduce
ANDES_RIZ  channels=[R,IZ]    extension=[1,2]     → inherits ANDES_UBV
ANDES_YJH  channels=[Y,J,H]   extension=[1,2,3]   → inherits CRIRES_PLUS
ANDES_K    channels=[K]       extension=1         → inherits ANDES_YJH
```

| Pros | Cons |
|------|------|
| Matches FITS file structure (1:1) | Two abstraction levels |
| Cleaner file sorting | Per-spectrograph tuning only (not per-arm) |
| Per-spectrograph settings tuning | |
| Follows XSHOOTER/HARPS/UVES pattern | |
| Can reduce entire spectrograph at once | |

---

## Recommendation

**Option C** is recommended because:

1. **Matches physical reality**: One instrument = one FITS file type
2. **Sufficient tuning granularity**: Detectors are similar within spectrograph
3. **Proven pattern**: XSHOOTER (UVB/VIS/NIR), HARPS (BLUE/RED), UVES work this way
4. **Clean file handling**: No overlap in file discovery

If per-arm tuning becomes necessary, Option B remains viable with settings inheritance minimizing duplication.

---

## File Structure (Option C)

```
pyreduce/instruments/
  andes_ubv.yaml    # channels=[U,B,V], extension=[1,2,3]
  andes_riz.yaml    # channels=[R,IZ], extension=[1,2]
  andes_yjh.yaml    # channels=[Y,J,H], extension=[1,2,3]
  andes_k.yaml      # channels=[K], extension=1

pyreduce/settings/
  settings_ANDES_UBV.json  # __inherits__: pyreduce (optical CCD tuning)
  settings_ANDES_RIZ.json  # __inherits__: ANDES_UBV
  settings_ANDES_YJH.json  # __inherits__: CRIRES_PLUS (near-IR tuning)
  settings_ANDES_K.json    # __inherits__: ANDES_YJH
```

### Migration

- Rename `andes.yaml` → `andes_yjh.yaml` (or delete and create fresh)
- Rename `settings_ANDES.json` → `settings_ANDES_YJH.json`
- Keep `AJ` as simulation/test instrument

---

## Future Enhancement: Per-Channel Settings

Adding per-channel settings would enable Option A (single instrument with full flexibility).

### Current Flow

```
configuration.py:load_config()
  → loads settings_INSTRUMENT.json
  → merges with defaults via _resolve_inheritance()
  → returns config dict (same for all channels)

reduce.py:Step.__init__()
  → receives config dict
  → uses values directly: self.degree = config["degree"]
```

### Proposed Enhancement

Add channel-aware value selection in Step classes, similar to how `getter` works for YAML configs.

#### Option 1: Settings-level indexing

Modify `_resolve_inheritance()` or add post-processing to index list values:

```python
def _apply_channel_index(config, channel, channels):
    """Index list values in settings by channel position."""
    if channel is None or channels is None:
        return config
    try:
        idx = channels.index(channel.upper())
    except (ValueError, AttributeError):
        return config

    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _apply_channel_index(value, channel, channels)
        elif isinstance(value, list) and len(value) == len(channels):
            result[key] = value[idx]
        else:
            result[key] = value
    return result
```

Call this when creating Step instances in `pipeline.py`.

#### Option 2: Step-level helper

Add a method to Step base class:

```python
class Step:
    def get_config(self, key, default=None):
        """Get config value, indexing by channel if value is a list."""
        value = self.config.get(key, default)
        if isinstance(value, list) and hasattr(self, 'channel_index'):
            if len(value) > self.channel_index:
                return value[self.channel_index]
        return value
```

#### Settings Format

```json
{
  "__instrument__": "ANDES",
  "trace": {
    "degree": 4,
    "noise": [10, 10, 10, 20, 20, 50, 50, 50, 100],
    "min_cluster": [500, 500, 500, 1000, 1000, 2000, 2000, 2000, 5000]
  },
  "science": {
    "extraction_height": 10,
    "oversampling": [10, 10, 10, 10, 10, 5, 5, 5, 3]
  }
}
```

Values that are scalars apply to all channels. Values that are lists with length matching `channels` are indexed.

#### Implementation Scope

1. **Minimal**: ~20-30 lines in `configuration.py` or `reduce.py`
2. **Changes needed**:
   - Pass `channel` and `channels` list to config processing
   - Add indexing logic (one of the options above)
   - Update Step classes to use indexed values
3. **Backward compatible**: Scalar values work as before; only lists trigger indexing
4. **Risk**: Low - additive change, doesn't break existing instruments

#### Semantic List Parameters (Potential Conflict)

Current settings use lists for semantic pairs, not per-channel values:

| Parameter | Meaning | Where Used |
|-----------|---------|------------|
| `closing_shape` | [x, y] kernel | trace morphology |
| `opening_shape` | [x, y] kernel | trace morphology |
| `degree` | [order, column] | 2D fits (wavecal, curvature, freq_comb) |
| `extraction_height` | [below, above] | rarely as list, usually scalar |

**Option 1: Refactor semantic lists away**

```json
// Before
"closing_shape": [5, 5]
"degree": [6, 6]

// After
"closing_shape_x": 5,
"closing_shape_y": 5,
"degree_order": 6,
"degree_column": 6
```

Pros: Clean separation, no ambiguity
Cons: Breaking change, migration needed

**Option 2: Length-based disambiguation**

Per-channel lists must match `len(channels)`. Semantic lists are always 2 elements.
- ANDES has 9 channels, so `[10, 20, 30, 40, 50, 60, 70, 80, 90]` → per-channel
- `[6, 6]` → semantic (2 elements ≠ 9 channels)

Pros: Backward compatible
Cons: Fragile if instrument has exactly 2 channels

**Option 3: Explicit marker**

```json
"noise": {"__per_channel__": [10, 20, 50, 100]}
// or
"noise": {"U": 10, "B": 10, "V": 10, "Y": 50, "J": 50, "H": 50, "K": 100}
```

Pros: Explicit, no ambiguity
Cons: More verbose

**Recommendation**: Option 1 (refactor) for cleanest long-term solution. The semantic lists are:
- `closing_shape`, `opening_shape`: Rarely changed, easy to split
- `degree`: Common but well-understood, split to `degree_order`/`degree_column`
- `extraction_height`: Already scalar in most configs, deprecate list form

---

## Summary

| Approach | Effort | Flexibility | Complexity |
|----------|--------|-------------|------------|
| Option A + per-channel settings | Medium | Per-channel | Single instrument, single abstraction |
| Option B (9 instruments) | Low | Per-arm | 9 instruments, single abstraction |
| Option C (4 instruments) | Low | Per-spectrograph | 4 instruments, two abstractions |

**Revised Recommendation:**

**Option A** is now the most attractive if we implement per-channel settings:
1. Single instrument, single abstraction level
2. Maximum flexibility (per-channel tuning)
3. Follows proven CRIRES+ pattern for file handling
4. Clean long-term solution

**Implementation path:**
1. Refactor semantic list parameters (`degree` → `degree_order`/`degree_column`, etc.)
2. Add per-channel settings indexing (~30 lines)
3. Create ANDES instrument with spectrograph filter (like CRIRES+ band filter)

If per-channel settings enhancement is deferred, use **Option C** as interim solution.
