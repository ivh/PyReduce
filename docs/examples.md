# Examples

PyReduce includes example scripts for each supported instrument in the `examples/` directory.

## Running an Example

The UVES example is a good starting point:

```bash
# Download sample data
uv run reduce download UVES

# Run the example
uv run python examples/uves_example.py
```

Or use the CLI directly:

```bash
uv run reduce run UVES -t "HD[- ]?132205" --steps bias,flat,trace,science
```

## Example Structure

Each example script follows the same pattern:

```python
from pyreduce.pipeline import Pipeline
from pyreduce import datasets

# Define parameters
instrument = "UVES"
target = "HD132205"
night = "2010-04-01"
channel = "middle"
steps = ("bias", "flat", "trace", "science")

# Download/locate data
base_dir = datasets.UVES()

# Run pipeline
Pipeline.from_instrument(
    instrument,
    target,
    night=night,
    channel=channel,
    steps=steps,
    base_dir=base_dir,
    plot=1,
).run()
```

## Modifying Steps

Edit the `steps` tuple to control which reduction steps run:

```python
steps = (
    "bias",
    "flat",
    "trace",
    # "curvature",    # Skip curvature
    # "scatter",      # Skip scatter
    "norm_flat",
    "wavecal",
    "science",
    # "continuum",    # Skip continuum
    "finalize",
)
```

Steps not in the list but required as dependencies will be loaded from
previous runs if the output files exist.

## Available Examples

### ESO Instruments

| Example | Description |
|---------|-------------|
| `uves_example.py` | ESO UVES |
| `harps_example.py` | ESO HARPS |
| `harpn_example.py` | HARPS-N (TNG) |
| `xshooter_example.py` | ESO XSHOOTER |
| `crires_plus_example.py` | ESO CRIRES+ |

### Space Telescopes

| Example | Description |
|---------|-------------|
| `jwst_niriss_example.py` | JWST NIRISS |
| `jwst_miri_example.py` | JWST MIRI |

### Other Observatories

| Example | Description |
|---------|-------------|
| `nirspec_example.py` | Keck NIRSPEC |
| `lick_apf_example.py` | Lick APF |
| `mcdonald_example.py` | McDonald Observatory |
| `neid_example.py` | NEID |

### ELT Instruments (Simulated)

| Example | Description |
|---------|-------------|
| `metis_lss_example.py` | ELT METIS Long-Slit |
| `metis_ifu_example.py` | ELT METIS IFU |
| `micado_example.py` | ELT MICADO |
| `mosaic_nir.py` | ELT MOSAIC NIR channel |
| `mosaic_vis.py` | ELT MOSAIC VIS channels |
| `andes_yjh_example.py` | ELT ANDES YJH channels |

### Advanced Usage

| Example | Description |
|---------|-------------|
| `mosaic_preset-slitfunc.py` | Using pre-computed slit function for single-pass extraction |

### Templates

| Example | Description |
|---------|-------------|
| `custom_instrument_example.py` | Template for adding new instruments |
| `toes_example.py` | Custom instrument example |
