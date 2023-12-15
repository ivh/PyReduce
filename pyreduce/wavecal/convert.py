# -*- coding: utf-8 -*-
"""
Simple script that converts IDL savefiles into numpy savefiles
"""

import os

import numpy as np
from numpy.lib.recfunctions import append_fields
from scipy.io import readsav


def remove_field_name(a, name):
    names = list(a.dtype.names)
    if name in names:
        names.remove(name)
    b = a[names]
    return b


files = [f for f in os.listdir() if f.endswith(".sav")]

if len(files) == 0:
    print("Nothing to convert")

for f in files:
    data = readsav(f)

    cs_lines = data["cs_lines"]
    mask = ~cs_lines.flag.astype(bool)

    cs_lines = remove_field_name(cs_lines, "FLAG")
    cs_lines = append_fields(cs_lines, "flag", mask, usemask=False, asrecarray=True)

    data["cs_lines"] = cs_lines

    new_name = f[:-4]
    np.savez(new_name, **data)
    print(f"Converted {f} to {new_name}.npz")
