# Wavelength Calibration

Wavelength calibration in PyReduce happens in multiple steps.

## Initial Linelist

To start the wavelength calibration we need an initial guess. PyReduce provides
initial guess files in the wavecal directory for supported instruments and modes.
These files are numpy `.npz` archives containing a recarray with the key `cs_lines`:

| Column | Type | Description |
|--------|------|-------------|
| `wlc` | float64 | Wavelength (before fit) |
| `wll` | float64 | Wavelength (after fit) |
| `posc` | float64 | Pixel position (before fit) |
| `posm` | float64 | Pixel position (after fit) |
| `xfirst` | int16 | First pixel of the line |
| `xlast` | int16 | Last pixel of the line |
| `width` | float64 | Width of the line in pixels |
| `height` | float64 | Relative strength of the line |
| `order` | int16 | Echelle order the line is found in |
| `flag` | bool | Whether to use the line |

If such a file is not available, or you want to create a new one, it is possible
to do so based on a rough initial guess of the wavelength ranges of each order
and a reference atlas of known spectral lines for the calibration lamp.

### Creating a New Linelist

1. Create the master wavelength calibration spectrum by running the `wavecal_master` step
2. Use the `wavecal_creator.py` script in tools to create the linelist, providing
   rough wavelength guesses (uncertainties up to 20 Angstrom are allowed)

Alternatively, run the `wavecal_init` step in PyReduce if the instrument provides
the correct initial wavelength guess via the `get_wavelength_range` function.

The initial line identification detects peaks in the observed spectrum, matches them
to atlas lines using offset voting and iterative polynomial fitting with outlier
rejection. The created linelist should be saved to the location provided by
`get_wavecal_filename` so `wavecal_init` only needs to run once.

## Spectral Order Numbers

PyReduce uses physical spectral order numbers (`m`) in wavelength calibration. Understanding
how these are assigned is important because the 2D wavelength polynomial depends on them.

### What is the Order Number?

In echelle spectrographs, light is dispersed by a grating where different wavelengths
satisfy the grating equation at different diffraction orders. The order number `m` is
this physical diffraction order. Higher orders correspond to shorter wavelengths.

### How Order Numbers are Assigned

Order numbers are assigned to traces in one of three ways:

1. **From `order_centers.yaml`** (preferred for new instruments):

   Instruments can provide a file listing known order centers and their order numbers.
   During trace detection, detected traces are matched to these centers by y-position
   and assigned the corresponding order number immediately.

   ```yaml
   # Example: pyreduce/instruments/ANDES_RIZ/order_centers_r2.yaml
   orders:
     85: 2048.5   # Order 85 centered at y=2048.5
     86: 1892.3
     87: 1741.2
     ...
   ```

2. **From the initial linelist** (wavecal_init step):

   If no order_centers file exists, traces start with `m = None`. The initial linelist
   file (e.g., `wavecal_middle.npz`) contains an `obase` value - the order number of
   the first trace. During `wavecal` step, each trace is assigned `m = obase + index`.

3. **Sequential fallback**:

   For legacy files or modes like MOSAIC where orders cannot be identified, `m` may
   remain None or be assigned sequentially from 0.

### Why Order Numbers Matter

The 2D wavelength polynomial fits wavelength as a function of both pixel position (x)
and order number (m):

```
wavelength = P(x, m) = sum_{i,j} c_{i,j} * x^i * m^j
```

Using physical order numbers (not sequential indices) is critical because the grating
equation creates predictable relationships between adjacent orders. A fit using physical
order numbers can interpolate and extrapolate more accurately.

When you call `Trace.wlen(x)`, it evaluates the 2D polynomial at the trace's order number:

```python
# Inside Trace.wlen():
wavelength = np.polynomial.polynomial.polyval2d(x, self.m, self.wave)
```

## Gas Lamp Calibration

For absolute wavelength reference, most spectrometers use gas lamps (e.g., ThAr).
These lamps have well-known spectral lines distributed across the detector range.

The calibration process:

1. Match observation to linelist using cross correlation in both order and pixel directions
2. Match observed peaks to closest partners in the linelist
3. Discard peaks further than a cutoff (typically ~100 m/s)
4. Fit a 2D polynomial between pixel position and wavelength
5. Use the polynomial to identify additional peaks
6. Iterate to refine the solution

PyReduce uses a 2D polynomial where the order number is the second coordinate.
This works because wavelength derivatives between orders are similar.

## Frequency Comb / Fabry-Perot Interferometer

Frequency combs and Fabry-Perot interferometers provide superior wavelength calibration
by generating dense, evenly-spaced peaks across all wavelengths.

The key property is constant frequency spacing:

```
f(n) = f0 + n * fr
```

where:
- `f0` is the anchor frequency
- `fr` is the frequency step between peaks
- `n` is the peak number

### Frequency Comb Process

1. Identify peaks in each order
2. Estimate wavelengths using the gas lamp solution
3. Number peaks consistently across orders (single f0 and fr for all peaks)
4. Use the grating equation (`n * w = const`) to correct peak numbering
5. Fit f0 and fr using all peaks
6. Derive wavelengths for each peak
7. Fit final polynomial for pixel-to-wavelength mapping

The dense peak coverage provides a much better solution than gas lamps alone.

## Reference Spectra

### ThAr (Thorium-Argon)

- Palmer, B.A. and Engleman, R., Jr., 1983, *Atlas of the Thorium Spectrum*, Los Alamos National Laboratory
- Norlen, G., 1973, *Physica Scripta*, 8, 249

### UNe (Uranium-Neon)

- Redman S.L. et al., *A High-Resolution Atlas of Uranium-Neon in the H Band*

## Step Dependencies

The wavelength calibration steps have dependencies:

```
wavecal_master  ->  wavecal_init  ->  wavecal
       |                |               |
   Extract         Initial line     Refine
   calibration     identification   solution
   spectrum
```

For instruments with frequency combs:

```
freq_comb_master  ->  freq_comb
       |                  |
   Extract           Apply frequency
   comb spectrum     comb calibration
```

The frequency comb steps use the gas lamp solution as a starting point.
