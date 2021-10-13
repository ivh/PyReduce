Wavelength Calibration
======================

Wavelength calibration in PyReduce happens in multiple steps.

Initial Linelist
----------------
To start the wavelength calibration we need an initial guess. PyReduce provides
a number of initial guess files in the wavecal directory for the supported
instruments and modes. These files are numpy ".npz" archives that contain a
recarray with the key "cs_lines", which has the following datatype:

    - (("wlc", "WLC"), ">f8"), # Wavelength (before fit)
    - (("wll", "WLL"), ">f8"), # Wavelength (after fit)
    - (("posc", "POSC"), ">f8"), # Pixel Position (before fit)
    - (("posm", "POSM"), ">f8"), # Pixel Position (after fit)
    - (("xfirst", "XFIRST"), ">i2"), # first pixel of the line
    - (("xlast", "XLAST"), ">i2"), # last pixel of the line
    - (("approx", "APPROX"), "O"), # ???
    - (("width", "WIDTH"), ">f8"), # width of the line in pixels
    - (("height", "HEIGHT"), ">f8"), # relative strength of the line
    - (("order", "ORDER"), ">i2"), # echelle order the line is found in
    - ("flag", "?"), # flag that tells us if we should use the line or not

If such a file is not available, or you want to create a new one, it is possible
to do so just based on a rough initial guess of the wavelength ranges of each order¨
as well as using a reference atlas of known spectral lines, for the gas lamp used in
the wavelength calibration.

First create the master wavelength calibration spectrum by running the "wavecal_master" step.
Then you can use the "wavecal_creator.py" script in tools to create the linelist, by providing
rough guesses for the wavelength (by default uncertainties of up to 20 Angstrom are allowed).
Alternatively you can run the "wavecal_init" step in PyReduce, assuming that the
current instrument provides the correct initial wavelength guess when calling
the "get_wavelength_range" function.

Either way, this will start a search for the best fit between the observation
and the known atlas using MCMC. This approach is unfortunately quite slow but
should be stable. It is therefore recommended that the created linelist is placed
in the location provided by the "get_wavecal_filename" function of the instrument,
and the "wavecal_init" step is only run once.

The MCMC determines the best fit based on a combination of the cross correlation
and the least squares match between the two spectra.

References for the included spectra:

ThAr
Palmer, B.A. and Engleman, R., Jr., 1983, Atlas of the Thorium Spectrum, Los Alamos National Laboratory, Los Alamos, New Mexico
Norlén, G., 1973, Physica Scripta, 8, 249.

UNe
Redman S.L. et al. A High-Resolution Atlas of Uranium-Neon in the H Band

"Traditional" Gas Lamp
----------------------
For an absolute wavelength reference most spectrometers rely on a gas lamp, e.g.
based on ThAr. Such lamps have a limited number of well known spectral lines,
well spaced over the entire range of the detector. Different gas lamps may be used
to cover different settings of the detector. The initial linelist described above
will tell us at which pixels and in which orders to expect which lines on the detector.
This allows us to assign a mapping between the pixel position and the wavelength.
In PyReduce we use a polynomial to interpolate between the known spectral lines.
First we match the observation to the linelist using cross correlation in both
order and pixel directions. Afterwards we match the observed peaks to their closest
partners in the linelist to get the wavelength. Here we discard peaks that are further
than a set cutoff away (usually around 100 m/s). Based on these we can fit a polynomial
between the peaks. In PyReduce we recommend using a 2D polynomial, where the order
number is used as the second coordinate. This works since the wavelength derivatives
between orders is similar. Using this polynomial we can derive the wavelengths of
all pixels in the spectrum. We also use this to identify additional peaks that exist
both in the observation and the linelist, but weren't matched before.
After repeating this method a few times, we arrive at a stable solution.

Frequency Comb / Fabry Perot Interferometer
--------------------------------------------
Recent developments in the detector design incorporate Frequency Combs or
Fabry Perot Interferometers to achieve better wavelength calibrations. On problem
with the gas lamp, is that there are only a limited number of spectral lines, and
they are not evenly spaced. Using a FC/FPI however solves these problems, by providing
dense, evenly spaced peaks over all wavelengths. An additional feature of these
is that the Frequency between peaks is constant. PyReduce can use such calibrations
just using the assumption that the steps are constant in frequency space. This is
done in the "freq_comb" step.

f(n) = f0 + n * fr,
where f0 is the anchor frequency, and fr is the frequency step between two peaks,
and n is the number of the peak.

First we identify the peaks in each order as usual. The wavelength of each peak
is estimated by using the gas lamp solution calculated above. The individual peaks
of the FC/FPI are great for relative differences but lack an absolute calibration.
Special care is taken to number the peaks correctly, even between orders, so that
we only have one anchor frequency and one frequency step, for all peaks.
One correction that we particularly want to point out is that the grating equation
provides us with the assumption that n * w = const for neighbouring peaks, where
w is the wavelength of the peak. This can be used to correct a misidentified order,
and correct their numbering, usually by 1 or 2 peak numbers.

The wavelength solution is then given by determining the best fit f0 and fr using
all peaks in all orders. Those can then be used to determine the wavelength of each peak.
Once we have a wavelength for each peak we can fit a polynomial just as we did
for the gas lamp, to get the wavelength for each pixel. Of course now there
are a lot more peaks, so the solution is much better.
