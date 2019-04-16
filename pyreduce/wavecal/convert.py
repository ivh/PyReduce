"""
Simple script that converts IDL savefiles into numpy savefiles
"""

import os

import numpy as np
from scipy.io import readsav


files = [f for f in os.listdir() if f.endswith(".sav")]

if len(files) == 0:
    print("Nothing to convert")

for f in files:
    data = readsav(f)
    new_name = f[:-4]
    np.savez(new_name, **data)
    print("Converted {f} to {new_name}.npz")
