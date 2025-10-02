![Python application](https://github.com/nadsabha/PyReduce_ELT/workflows/Python%20application/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pyreduce-astro/badge/?version=latest)](https://pyreduce-astro.readthedocs.io/en/latest/?badge=latest)
[![Updates](https://pyup.io/repos/github/nadsabha/PyReduce_ELT/shield.svg)](https://pyup.io/repos/github/nadsabha/PyReduce_ELT/)

# PyREDUCE_ELT

PyReduce_ELT is an adapted branch of the PyReduce package optimised to handle ScopeSim-simulated ELT data for MICADO and METIS (optimized for LSS_M mode only for now) instruments. Only the relevant information pertaining to operating PyReduce on ELT MICADO and METIS data are addressed in this README file, while all details of the original PyReduce package can be found in this [README.md](https://github.com/AWehrhahn/PyReduce/blob/master/README.md).


Installation
------------

The most up-to-date version can be installed using python 3 command:
```
pip install git+https://github.com/nadsabha/PyReduce_ELT
```

PyReduce uses CFFI to link to the C code, on non-linux platforms you might have to install libffi.
See also https://cffi.readthedocs.io/en/latest/installation.html#platform-specific-instructions for details.

Output Format
-------------
PyReduce will create ``.ech`` files when run. Despite the name those are just regular ``.fits`` files and can be opened with any programm that can read ``.fits``. The data is contained in a table extension. The header contains all the keywords of the input science file, plus some extra PyReduce specific keyword, all of which start with ``e_``.

Other PyReduce outputs are ``.npz`` files. These are numpy recarrays and can be opened as in the example copied below:
```
file_py_wave='metis_lss_m.thar.npz'
data_npz_py_wave = np.load(file_py_wave,  allow_pickle=True)
#load wave_cal npz file
lst = data_npz_py_wave.files
print(lst)
for item in lst:
    print(item)
    print(data_npz_py_wave[item])
# or
wave=data_npz_py_wave['wave']
coef=data_npz_py_wave['coef']
```

How To
------
PyReduce can be run using the provided example files, e.g.:
for MICADO ``examples/micado_example.py``,
for METIS ``examples/metis_example.py``.

In this example script, we first define the instrument and instrument mode (``LSS_M`` for METIS). Then the path to where the data are located is defined, as well as the output directory. Lastly, all the specific settings of the reduction (e.g. polynomial degrees of various fits) are defined in the json configuration files [settings_MICADO.json](https://github.com/nadsabha/PyReduce_ELT/blob/master/pyreduce/settings/settings_MICADO.json), [settings_METIS.json](https://github.com/nadsabha/PyReduce_ELT/blob/master/pyreduce/settings/settings_METIS.json), or alternatively directly within the script by adding, e.g. ``config["curvature"]["extraction_width"] = 0.77``, ``config["wavecal"]["dimensionality"] = "1D"``, etc.

Explanation of the settings paratemters can be found in [settings_schema.json](https://github.com/nadsabha/PyReduce_ELT/blob/master/pyreduce/settings/settings_schema.json).

The steps of the reduction desired to be performed are then specified. Steps that are not specified, but are still required, will be loaded from previous runs if possible, or executed otherwise.
All of this is then passed to pyreduce.reduce.main to start the reduction.

In this example, PyReduce will plot all intermediary results, and also plot the progres during some of the steps. Close them to continue calculations. Once you are statisified with the results you can disable them in [settings_MICADO.json](https://github.com/nadsabha/PyReduce_ELT/blob/master/pyreduce/settings/settings_MICADO.json) or [settings_METIS.json](https://github.com/nadsabha/PyReduce_ELT/blob/master/pyreduce/settings/settings_METIS.json) (with ``plot : false`` in each step) to speed up the computation.

Relevant for MICADO:

Please note that in the micado example file it is specified to return only the order trace corresponding to the center of the order on MICADO (HK band) files, i.e. fit number 4 (or 3 as per Python convention counted from bottom to up) of the traces on the pinhole frame.

Relevant for METIS:

Please note that reduce.py main script is modified to return only the order trace corresponding to the center of the slit trace on METIS LSS files, i.e. fit number 17 (or 16 as per Python convention counted from bottom to up) of the traces on the pinhole frame.

Input Data
------

MICADO data:

Input simulated MICADO 'raw' data  can be downloaded directly from this [link](https://www.dropbox.com/sh/e3lnvtkmyjveajk/AABPHxeUdDO5AnkWCAjbM0e1a?dl=0) and placed in the input file path defined in [micado_example.py](https://github.com/nadsabha/PyReduce_ELT/blob/master/examples/micado_example.py). The files include:

• FF_detector_SPEC_3000x20_Spec_HK.fits: spectroscopic flatfield

• PINHOLE_detector_SPEC_3000x20_Spec_HK.fits: pinhole frame with the flatfiled lamp

• LL_detector_SPEC_3000x20_Spec_HK.fits: linelamp spectrum full slit

• detector_final_delta.fits: linelamp spectrum that acts as a "science" target to make pyreduce run

METIS data:

Input simulated METIS 'raw' data  can be downloaded directly from this [link](https://www.dropbox.com/sh/h1dz80vsw4lwoel/AAAqJD_FGDGC-t12wgnPXVR8a?dl=0) and placed in the input file path defined in [metis_example.py](https://github.com/nadsabha/PyReduce_ELT/blob/master/examples/metis_example.py). The files include:

• lss_m_thermal.fits: spectroscopic flat field

• lss_m_pinholes.fits: pinhole frame with the flat field, used for order/trace detection and fit

• lss_m_sky.fits: sky emission line spectrum covering the full slit, used for curvature determination and the wavelength calibration

• lss_m_star.fits: sky emission line spectrum covering the full slit with a star spectrum "science" frame

Reference Papers
------
The original REDUCE paper: [doi:10.1051/0004-6361:20020175](https://doi.org/10.1051/0004-6361:20020175)

A paper describing the changes and updates of PyReduce can be found here: [https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..32P/abstract](https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..32P/abstract)


Diff to reduce.py that was removed when merging
------

diff --git a/pyreduce/reduce.py b/pyreduce/reduce.py
index ba2c639..f7c2772 100755
--- a/pyreduce/reduce.py
+++ b/pyreduce/reduce.py
@@ -764,6 +764,26 @@ class OrderTracing(CalibrationStep):
             plot_title=self.plot_title,
         )

+        # NBS: BEGINNING of fix for MICADO and METIS pinholes
+        print('#NBS:here!!!')
+        print(orders.shape) #NBS
+
+        # if len(orders) == 7:
+        #     orders=orders[3] #NBS:  MICADO fix if only 1 order on the detector, use [3::7] if 2 orders present
+        #     orders = orders.reshape((1, 5))#NBS: to reshape it
+        #     print(orders.shape) #NBS
+
+        # if len(orders) == 14:
+        #     orders=orders[3::7] #NBS:  MICADO fix if 2 orders on the detector
+        #     print(orders.shape) #NBS
+
+        if len(orders) == 33:
+            orders=orders[16] #NBS: METIS fix For MICADO if only 1 order on the detector, use [3::7] if 2 orders present
+            orders = orders.reshape((1, 5))#NBS: to reshape it
+            print(orders.shape) #NBS
+
+        # NBS: END of fix for MICADO and METIS pinholes
+
         self.save(orders, column_range)

         return orders, column_range
