# Per-Channel Fiber Group Selection

**Status**: Proposal (not yet implemented)

## Problem

Some instruments have multiple channels that need different fiber group selections for the same reduction step. Currently `fibers.use` maps step names directly to group selections, applying the same selection to all channels.

## Proposed Solution

Extend `fibers.use` to optionally accept per-channel mappings:

```yaml
fibers:
  groups:
    sci_nir: { range: [1, 50], merge: average }
    sci_vis: { range: [1, 30], merge: average }
    cal_nir: { range: [51, 55], merge: average }
    cal_vis: { range: [31, 35], merge: average }
  use:
    # Per-channel selection (new syntax)
    science:
      NIR: [sci_nir]
      VIS1: [sci_vis]
      default: [sci_nir]  # fallback for unlisted channels
    # Same for all channels (existing syntax, unchanged)
    wavecal: [cal_nir, cal_vis]
```

Both syntaxes coexist - a list/string means "same for all channels", a dict means "per-channel selection".

## Implementation

### 1. trace.py - `select_traces_for_step()`

Add `channel` parameter and modify selection lookup:

```python
def select_traces_for_step(
    raw_traces,
    raw_cr,
    group_traces,
    group_cr,
    fibers_config,
    step_name,
    channel=None,  # new parameter
):
    ...
    if fibers_config.use is not None and step_name in fibers_config.use:
        selection = fibers_config.use[step_name]
        # Handle per-channel dict
        if isinstance(selection, dict):
            selection = selection.get(channel, selection.get("default", "groups"))
```

### 2. reduce.py - Update call sites

Two locations need to pass `self.channel`:

- Line ~366 (in `Step.get_traces_for_step`)
- Line ~978 (in another step class)

### 3. instruments/models.py - Update Pydantic model

Modify `FibersUseConfig` to accept either:
- `str` or `list[str]` (current)
- `dict[str, str | list[str]]` (new per-channel syntax)

## Scope

~10-15 lines of code across 3 files. Backward compatible - existing configs work unchanged.

---

## Future Extension: Per-Group Extraction Parameters

### Problem

Different fiber groups may need different extraction parameters. For example, calibration fibers might be physically narrower than science fibers and need a smaller `extraction_height`.

### Design Consideration

This raises a question about the config.yaml / settings.json separation:
- **config.yaml** - WHAT the instrument is (physical properties, fiber groupings)
- **settings.json** - HOW to reduce (algorithm parameters like extraction_height)

Putting extraction_height in config.yaml with the group definition would blur this line.

### Recommended Approach: `group_overrides` in settings.json

Keep the separation clean by adding per-group parameter overrides in settings:

```json
{
  "science": {
    "extraction_height": 0.5,
    "oversampling": 10,
    "group_overrides": {
      "cal": { "extraction_height": 0.3 },
      "sky": { "extraction_height": 0.4, "oversampling": 5 }
    }
  }
}
```

### Benefits

- config.yaml remains purely descriptive (physical layout)
- Users can tune per-group parameters without modifying instrument config
- Per-channel settings files (`settings_{channel}.json`) can have different group_overrides
- Any extraction parameter can be overridden, not just extraction_height
