import os.path
import numpy as np
import matplotlib.pyplot as plt
import pyreduce
from pyreduce import datasets, instruments, util
from pyreduce.wavelength_calibration import (
    WavelengthCalibration as WavelengthCalibrationModule
)


def func_wavecal(
    degree, thar, instrument, mode, threshold, iterations, wavecal_mode, shift_window
):
    reference = instruments.instrument_info.get_wavecal_filename(None, instrument, mode)
    reference = np.load(reference, allow_pickle=True)
    linelist = reference["cs_lines"]

    module = WavelengthCalibrationModule(
        plot=False,
        manual=False,
        degree=degree,
        threshold=threshold,
        iterations=iterations,
        mode=wavecal_mode,
        shift_window=shift_window,
    )
    wave, coef = module.execute(thar, linelist)

    return module.aic

def func_freq_comb(degree, comb, wave, threshold, wavecal_mode):
    module = WavelengthCalibrationModule(
            plot=False,
            degree=degree,
            threshold=threshold,
            mode=wavecal_mode,
            lfc_peak_width=3,
            nstep=8
        )
    wave = module.frequency_comb(comb, wave)
    return module.aic

# define parameters
instrument = "HARPS"
target = "HD109200"
night = "2015-04-09"
mode = "red"

# some basic settings
# Expected Folder Structure: base_dir/datasets/HD132205/*.fits.gz
# Feel free to change this to your own preference, values in curly brackets will be replaced with the actual values {}

# load dataset (and save the location)
base_dir = "/DATA/PyReduce/"
input_dir = "FrequencyComb/raw"
output_dir = f"FrequencyComb/reduced_{mode}"

f = os.path.join(base_dir, output_dir, "harps_red.thar.npz")
data = np.load(f, allow_pickle=True)
thar = data["thar"]
wave = data["wave"]

f = os.path.join(base_dir, output_dir, "harps_red.comb.npz")
data = np.load(f, allow_pickle=True)
comb = data["comb"]

ndim = 2
kwargs = {
    "thar": thar,
    "instrument": instrument,
    "mode": mode,
    "threshold": 100,
    "iterations": 3,
    "wavecal_mode": f"{ndim}D",
    "shift_window": 0.01,
}
kwargs_comb = {"wave": wave, "comb": comb, "threshold":100, "wavecal_mode": f"{ndim}D"}

shape = tuple([20] * ndim)
grid = np.zeros((*shape, ndim), int)
for i in np.ndindex(shape):
    grid[i] = i

# matrix = util.gridsearch(func_wavecal, grid, kwargs=kwargs)
matrix = util.gridsearch(func_freq_comb, grid, kwargs=kwargs_comb)

np.save(f"matrix_comb_{ndim}D.npy", matrix)

if ndim == 1:
    plt.plot(grid[:, 0], matrix)
elif ndim == 2:
    plt.imshow(matrix)
    # plt.xticks(labels=grid[0])
    # plt.yticks(labels=grid[:, 0])
plt.show()
