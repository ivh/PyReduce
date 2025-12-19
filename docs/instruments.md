# Instruments

PyReduce supports many instruments out of the box. Custom instruments can be added
by creating YAML configuration files.

## Supported Instruments

### ESO Instruments
- **HARPS** - High Accuracy Radial velocity Planet Searcher
- **HARPS-N** - HARPS-North at TNG
- **UVES** - UV-Visual Echelle Spectrograph
- **XSHOOTER** - Wide-band spectrograph (UVB/VIS/NIR)
- **CRIRES+** - Cryogenic IR Echelle Spectrograph

### Space Telescopes
- **JWST NIRISS** - Near Infrared Imager and Slitless Spectrograph
- **JWST MIRI** - Mid-Infrared Instrument

### Other Observatories
- **Keck NIRSPEC** - Near-IR spectrograph
- **Lick APF** - Automated Planet Finder
- **McDonald** - McDonald Observatory spectrograph
- **NEID** - NN-EXPLORE Exoplanet Investigations with Doppler spectroscopy

### ELT Instruments (Simulated)
- **METIS** - Mid-infrared ELT Imager and Spectrograph
- **MICADO** - Multi-AO Imaging Camera for Deep Observations

## Adding a Custom Instrument

### Method 1: YAML Configuration

Create a YAML file defining your instrument:

```yaml
# myinstrument.yaml
instrument: MyInstrument
telescope: MyTelescope
arms: [default]

# Detector
naxis: [2048, 2048]
orientation: 0
extension: 0
gain: 1.0
readnoise: 5.0

# Header keywords
date: DATE-OBS
target: OBJECT
exposure_time: EXPTIME

# File classification
kw_bias: IMAGETYP
id_bias: bias
kw_flat: IMAGETYP
id_flat: flat
kw_spec: IMAGETYP
id_spec: object
```

Load it directly:

```python
from pyreduce.instruments import load_instrument

inst = load_instrument("/path/to/myinstrument.yaml")
```

### Method 2: Python Class

For instruments needing custom logic, create a Python class:

```python
from pyreduce.instruments.common import Instrument

class MyInstrument(Instrument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom initialization

    def add_header_info(self, header):
        # Compute derived header values
        header["MJD_MID"] = header["MJD_OBS"] + header["EXPTIME"] / 86400 / 2
        return header

    def get_wavecal_filename(self, header, arm, **kwargs):
        # Return path to wavelength calibration file
        return f"wavecal_{arm}.npz"
```

Place the YAML in `pyreduce/instruments/` and the Python file alongside it.

## Instrument Configuration Fields

### Required Fields

| Field | Description |
|-------|-------------|
| `instrument` | Instrument name |
| `telescope` | Telescope name |
| `arms` | List of instrument arms/modes |
| `naxis` | Detector dimensions [x, y] |

### Detector Properties

| Field | Description |
|-------|-------------|
| `extension` | FITS extension (0 or name) |
| `orientation` | Rotation/flip code (0-7) |
| `gain` | Detector gain (value or header keyword) |
| `readnoise` | Read noise (value or header keyword) |
| `dark` | Dark current (value or header keyword) |
| `prescan_x` | Prescan region in x |
| `overscan_x` | Overscan region in x |

### Header Mappings

These map instrument-specific header keywords to internal names:

| Field | Description |
|-------|-------------|
| `date` | Observation date |
| `target` | Target name |
| `exposure_time` | Exposure time |
| `ra`, `dec` | Coordinates |
| `jd` | Julian date |
| `instrument_mode` | Instrument mode |

### File Classification

| Field | Description |
|-------|-------------|
| `kw_bias`, `id_bias` | Bias file keyword and pattern |
| `kw_flat`, `id_flat` | Flat file keyword and pattern |
| `kw_wave`, `id_wave` | Wavelength calibration keyword and pattern |
| `kw_spec`, `id_spec` | Science file keyword and pattern |

See `pyreduce/instruments/models.py` for the complete schema.

## Detector Orientation

The `orientation` field controls how the raw image is rotated/flipped before processing.
Values 0-7 correspond to different transformations:

| Value | Transformation |
|-------|----------------|
| 0 | No change |
| 1 | Rotate 90 CCW |
| 2 | Rotate 180 |
| 3 | Rotate 90 CW |
| 4 | Flip horizontal |
| 5 | Flip horizontal + rotate 90 CCW |
| 6 | Flip vertical |
| 7 | Flip horizontal + rotate 90 CW |

The goal is to orient the image so that:
- Dispersion direction is horizontal (wavelength increases left to right)
- Cross-dispersion is vertical (orders are stacked vertically)

## Reduction Settings

Each instrument can have its own settings file at `pyreduce/settings/settings_INSTRUMENT.json`.
This overrides the defaults for that instrument. See [Configuration](configuration_file.md) for details.
