# -*- coding: utf-8 -*-
import json
import os
import subprocess
import sys
import tempfile
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import signal
from scipy.optimize import NonlinearConstraint, minimize

from pyreduce.echelle import Echelle
from pyreduce.util import polyfit2d
from pyreduce.wavelength_calibration import WavelengthCalibration


def obs_to_ech(obs, fname):
    ech = Echelle(data={"spec": obs})
    ech.save(fname)


def make_lab_spec(wpoints):
    wmin, wmax = wpoints.min(), wpoints.max()
    n = 10_000
    ref = np.zeros(n)
    wave = np.linspace(wmin, wmax, num=n, endpoint=True)
    idx = np.digitize(wpoints, wave)

    for i in range(len(wpoints)):
        mid = idx[i]
        xfirst, xlast = max(mid - 25, 0), min(mid + 25, n)
        ref[xfirst:xlast] += signal.gaussian(xlast - xfirst, 5)

    ref = np.clip(ref, 0, 1)

    header = fits.Header({"CRVAL1": wmin, "CDELT1": wave[1] - wave[0]})
    hdu = fits.PrimaryHDU(data=ref, header=header)
    hdu.writeto("labspec.fits", overwrite=True)

    pass


def get_typecode(dtype):
    """Get the IDL typecode for a given dtype"""
    if dtype.name[:5] == "bytes":
        return "1"
    if dtype.name == "int16":
        return "2"
    if dtype.name == "int32":
        return "3"
    if dtype.name == "float32":
        return "4"
    if dtype.name == "float64":
        return "5"
    if dtype.name[:3] == "str":
        return dtype.name[3:]
    raise ValueError("Don't recognise the datatype")


temps_to_clean = []


def save_as_binary(arr):
    global temps_to_clean

    with tempfile.NamedTemporaryFile("w+", suffix=".dat", delete=False) as temp:
        if arr.dtype.name[:3] == "str" or arr.dtype.name == "object":
            arr = arr.astype(bytes)
            shape = (arr.dtype.itemsize, len(arr))
        else:
            shape = arr.shape[::-1]

        # Most arrays should be in the native endianness anyway
        # But if not we swap it to the native representation
        endian = arr.dtype.str[0]
        if endian == "<":
            endian = "little"
        elif endian == ">":
            endian = "big"
        elif endian == "|":
            endian = sys.byteorder

        if endian != sys.byteorder:
            arr = arr.newbyteorder().byteswap()
            endian = "native"

        arr.tofile(temp)
        value = [temp.name, str(list(shape)), get_typecode(arr.dtype), endian]
    temps_to_clean += [temp]
    return value


def clean_temps():
    global temps_to_clean
    for temp in temps_to_clean:
        try:
            os.remove(temp)
        except:
            pass

    temps_to_clean = []


def write_as_idl(cs_lines):
    idl_fields = {
        "wlc": cs_lines["wll"].tolist(),
        "wll": cs_lines["wll"].tolist(),
        "posc": cs_lines["posc"].tolist(),
        "posm": cs_lines["posm"].tolist(),
        "xfirst": cs_lines["xfirst"].tolist(),
        "xlast": cs_lines["xlast"].tolist(),
        "approx": np.full(len(cs_lines), "G").tolist(),
        "width": cs_lines["width"].tolist(),
        "flag": (~cs_lines["flag"]).astype(int).tolist(),
        "height": cs_lines["height"].tolist(),
        "order": cs_lines["order"].tolist(),
    }

    sep = ""
    text = ""

    for key, value in idl_fields.items():
        if isinstance(value, dict):
            text += f"{sep}{key!s}:{{$\n"
            sep = ""
            for key2, value2 in value.items():
                text += f"{sep}{key2!s}:{value2!r}$\n"
                sep = ","
            sep = ","
            text += "}$\n"
        else:
            text += f"{sep}{key!s}:{value!r}$\n"
            sep = ","

    return text


def save_as_idl(obj, objname, fname):
    """
    Save the SME structure to disk as an idl save file

    This writes a IDL script to a temporary file, which is then run
    with idl as a seperate process. Therefore this reqires a working
    idl installation.

    There are two steps to this. First all the fields from the sme,
    structure need to be transformed into simple idl readable structures.
    All large arrays are stored in seperate binary files, for performance.
    The script then reads those files back into idl.
    """
    with tempfile.NamedTemporaryFile("w+", suffix=".pro") as temp:
        tempname = temp.name
        temp.write("print, 'Hello'\n")
        temp.write("sme = {")
        # TODO: Save data as idl compatible data
        temp.write(write_as_idl(obj))
        temp.write("} \n")
        # This is the code that will be run in idl
        temp.write("print, 'there'\n")
        temp.write(
            """tags = tag_names(sme)
print, tags
new_sme = {}

for i = 0, n_elements(tags)-1 do begin
    arr = sme.(i)
    s = size(arr)
    if (s[0] eq 1) and (s[1] eq 4) then begin
        void = execute('shape = ' + arr[1])
        type = fix(arr[2])
        endian = string(arr[3])
        arr = read_binary(arr[0], data_dims=shape, data_type=type, endian=endian)
        if type eq 1 then begin
            ;string
            arr = string(arr)
        endif
    endif
    if (s[s[0]+1] eq 8) then begin
        ;struct
        tags2 = tag_names(sme.(i))
        new2 = {}
        tmp = sme.(i)

        for j = 0, n_elements(tags2)-1 do begin
            arr2 = tmp.(j)
            s = size(arr2)
            if (s[0] eq 1) and (s[1] eq 4) then begin
                void = execute('shape = ' + arr2[1])
                type = fix(arr2[2])
                endian = string(arr2[3])
                arr2 = read_binary(arr2[0], data_dims=shape, data_type=type, endian=endian)
                if type eq 1 then begin
                    ;string
                    arr2 = string(arr2)
                endif
            endif
            new2 = create_struct(temporary(new2), tags2[j], arr2)
        endfor
        arr = new2
    endif
    new_sme = create_struct(temporary(new_sme), tags[i], arr)
endfor
"""
        )

        temp.write(
            f"cat = REPLICATE({{CSLINE, WLC:0.0, WLL:0.0, POSC:0.0, POSM: 0.0, XFIRST: 0, XLAST: 0, APPROX: 'G', WIDTH: 1.0, FLAG: 1, height: 1.0, order:0}}, {len(obj)})\n"
        )
        temp.write(
            """for i = 0, n_elements(cat) -1 do begin
    cat[i] = {CSLINE, WLC:sme.WLC[i], WLL:sme.WLL[i], POSC:sme.posc[i], POSM: sme.posm[i], XFIRST: sme.xfirst[i], XLAST: sme.xlast[i], APPROX: sme.approx[i], WIDTH: sme.width[i], FLAG: sme.flag[i], height: sme.height[i], order:sme.order[i]}
end
obase = long(14)
oincr = 1
"""
        )
        solution_2d = np.zeros(27)
        temp.write(f"solution_2d = {solution_2d.tolist()}\n")

        temp.write(f"{objname} = cat\n")
        temp.write(f'save, {objname}, obase, oincr, filename="{fname}"\n')
        temp.write("end\n")
        temp.flush()

        # with open(os.devnull, 'w') as devnull:
        subprocess.run(["idl", "-e", ".r %s" % tempname])
        # input("Wait for me...")
        clean_temps()


class OrderPlot:
    def __init__(self, obs, lines, order, degree=5):
        self.nord, self.ncol = obs.shape
        self.order = order

        self.obs = obs
        self.data = obs[order]

        self.peaks, _ = signal.find_peaks(self.data, width=3)

        self.lines = lines[lines["order"] == order]

        self.degree = degree
        self.fit()

        self.fig, self.ax = plt.subplots()

        self.connect()

        self.plot()
        self.op = None
        self.firstclick = True

    def plot(self):
        self.ax.clear()
        ref = np.zeros(self.ncol)

        xfirst = self.lines["xfirst"]
        xlast = self.lines["xlast"]

        # xfirst = np.digitize(xfirst, self.wave) - 3
        # xlast = np.digitize(xlast, self.wave) + 3

        xfirst = np.clip(xfirst, 0, len(ref)).astype(int)
        xlast = np.clip(xlast, 0, len(ref)).astype(int)

        for i, line in enumerate(self.lines):
            ref[xfirst[i] : xlast[i]] += line["height"] * signal.gaussian(
                xlast[i] - xfirst[i], line["width"]
            )

        # ref /= np.max(ref)
        ref += 1
        data = self.data + 1

        plt.title(f"Order {self.order}")
        plt.plot(data, label="data")
        plt.plot(self.peaks, data[self.peaks], "d", label="peaks")
        plt.plot(ref, label="reference")
        plt.yscale("log")
        plt.legend()

        # x0, x1 = np.where(self.data != 0)[0][[0, -1]]
        # plt.xlim(self.wave[x0], self.wave[x1])

        self.fig.canvas.draw()

    def connect(self):
        """connect the click event with the appropiate function"""
        self.cidclick = self.ax.figure.canvas.mpl_connect(
            "button_press_event", self.on_click
        )

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cidclick)

    def fit(self):
        mask = self.lines["flag"]

        pos = self.lines["posm"][mask]
        wave = self.lines["wll"][mask]
        deg = min(self.degree, len(pos))

        x = np.arange(self.ncol)
        fun = lambda p: np.where(np.gradient(np.polyval(p, x)) >= 0, 1, 0)

        # constraints = NonlinearConstraint(fun, 1, 1, jac="3-point", keep_feasible=True)

        # func = lambda p: np.sum((np.polyval(p, wave) - pos)**2)
        # x0 = np.polyfit(pos, wave, deg)
        # res = minimize(func, x0=x0, jac="3-point", method="trust-constr", constraints=constraints, options={"initial_constr_penalty": 100})

        self.coeff = np.polyfit(pos, wave, deg)
        self.wave = self.eval(x)
        mask = fun(self.coeff) == 0
        self.wave[mask] = np.interp(x[mask], x[~mask], self.wave[~mask])

    def eval(self, x):
        return np.polyval(self.coeff, x)

    def on_click(self, event):
        if event.xdata is None:
            return
        print(event.xdata)

        # Right Click
        if event.button == 3:
            # x = np.digitize(event.xdata, self.wave)
            x = event.xdata
            idx = np.argmin(np.abs(x - self.lines["posm"]))

            # find closest peak
            (line,) = plt.plot(self.lines["posm"][idx], self.lines["height"][idx], "rx")
            self.ax.figure.canvas.draw()
            self.fig.canvas.flush_events()
            self.lines["flag"][idx] = False
            self.lines["height"][idx] = 0.1

            self.fit()
            self.plot()

        # Left Click
        elif event.button == 1:
            if self.firstclick:
                x = event.xdata
                # find closest peak
                idx = np.argmin(np.abs(x - self.peaks))
                (line,) = plt.plot(
                    [
                        self.peaks[idx],
                    ],
                    [
                        self.data[self.peaks[idx]] + 1,
                    ],
                    "rx",
                )
                self.ax.figure.canvas.draw()
                self.fig.canvas.flush_events()

                self.position = self.peaks[idx]
                self.height = self.data[self.peaks[idx]]

                self.firstclick = False
            else:
                # x = np.digitize(event.xdata, self.wave)
                x = event.xdata
                idx = np.argmin(np.abs(x - self.lines["posm"]))

                # find closest peak
                (line,) = plt.plot(
                    self.lines["posm"][idx], self.lines["height"][idx] + 1, "rx"
                )
                self.ax.figure.canvas.draw()
                self.fig.canvas.flush_events()

                self.lines["xfirst"][idx] -= self.lines["posm"][idx] - self.position
                self.lines["xlast"][idx] -= self.lines["posm"][idx] - self.position
                self.lines["posm"][idx] = self.position
                self.lines["height"][idx] = self.height
                self.lines["flag"][idx] = True

                self.fit()
                self.plot()

                self.firstclick = True


names = ("posc", "order", "wll", "height", "width", "xfirst", "xlast", "posm", "flag")
dtype = [
    ("posc", "<f8"),
    ("order", "<f8"),
    ("wll", "<f8"),
    ("height", "<f8"),
    ("width", "<f8"),
    ("xfirst", "<f8"),
    ("xlast", "<f8"),
    ("posm", "<f8"),
    ("flag", "?"),
]

ifile = join(dirname(__file__), "xshooter_nir.json")
with open(ifile) as f:
    instr = json.load(f)

afile = join(dirname(__file__), "xshooter_nir.txt")
df = pd.read_table(afile, sep=",", header=None, usecols=(0,), names=["wave"])
df["wave"] *= 10

make_lab_spec(df["wave"])

ofile = join(dirname(__file__), "xshooter_nir.thar.npz")
obs = np.load(ofile)["thar"]
obs[obs < 0] = 0
for i in range(len(obs)):
    obs[i][obs[i] != 0] -= np.nanmedian(obs[i][obs[i] != 0])
obs[obs < 0] = 0

# obs[obs > np.nanpercentile(obs, 95)] = np.nanpercentile(obs, 95)
# obs /= np.max(obs)

obs_to_ech(obs, "xshooter.thar.ech")


# Whether to flip the order of orders
flip_orders = True

# posc, order, wll, height, width, xfirst, xlast, posm, flag = (
#     [],
#     [],
#     [],
#     [],
#     [],
#     [],
#     [],
#     [],
#     [],
# )

# solution_2d = np.zeros(27)
# solution_2d[0] = 2.5
# solution_2d[1] = obs.shape[1]
# solution_2d[2] = obs.shape[0]
# solution_2d[3] = 11
# solution_2d[7] = 4
# solution_2d[8] = 1
# solution_2d[9] = 2

# x = np.zeros((len(obs), 2))
# y = np.zeros((len(obs), 2))
# z = np.zeros((len(obs), 2))

# for i in range(len(obs)):
#     x[i] = 0, obs.shape[1]
#     if not flip_orders:
#         y[i] = i + instr["order-range"][0], i + instr["order-range"][0]
#         z[i] = instr["orders"][str(i + instr["order-range"][0])]
#     else:
#         y[i] = instr["order-range"][1] - 3 - i, instr["order-range"][1] - 3 - i
#         z[i] = instr["orders"][str(instr["order-range"][1] - i)]

# coeff = polyfit2d(x, y, z, [1, 2], plot=True)
# solution_2d[10] = coeff[0, 0]
# solution_2d[11] = coeff[1, 0]
# solution_2d[12] = coeff[0, 1]
# solution_2d[13] = coeff[0, 2]


# for ord in range(*instr["order-range"]):
#     wmin, wmax = instr["orders"][str(ord)]
#     lines = df[(df["wave"] > wmin) & (df["wave"] < wmax)]
#     n = len(lines)

#     if not flip_orders:
#         i = float(ord) - instr["order-range"][0]
#     else:
#         i = instr["order-range"][1] - float(ord)

#     if i < len(obs) and i >= 0:
#         x0, x1 = np.where(obs[int(i)] != 0)[0][[0, -1]]
#         posc += [(lines["wave"].values - wmin) / (wmax - wmin) * (x1 - x0) + x0]
#     else:
#         posc += [(lines["wave"].values - wmin) / (wmax - wmin) * instr["npix"]]
#     order += [np.full(n, i)]
#     wll += [lines["wave"].values]
#     height += [np.full(n, 1.0)]
#     width += [np.full(n, 5)]
#     xfirst += [posc[-1] - 2.5]
#     xlast += [posc[-1] + 2.5]
#     posm += [np.copy(posc[-1])]
#     flag += [np.full(n, True)]

# posc = np.concatenate(posc)
# order = np.concatenate(order)
# wll = np.concatenate(wll)
# height = np.concatenate(height)
# width = np.concatenate(width)
# xfirst = np.concatenate(xfirst)
# xlast = np.concatenate(xlast)
# posm = np.concatenate(posm)
# flag = np.concatenate(flag)

# cs_lines = np.rec.fromarrays(
#     (posc, order, wll, height, width, xfirst, xlast, posm, flag), dtype=dtype
# )

# np.savez("bla.npz", cs_lines=cs_lines)
# save_as_idl(cs_lines, "cs_lines", "xshooter_2D.sav")

cs_lines = np.load("cs_lines.npz")["cs_lines"]

wc = WavelengthCalibration(manual=False, plot=2)
wc.nord, wc.ncol = obs.shape
obs, cs_lines = wc.normalize(obs, cs_lines)
cs_lines = wc.align(obs, cs_lines)

for ord in range(len(obs)):
    plot = OrderPlot(obs, cs_lines, ord)
    plot.connect()
    plt.show()
    cs_lines[cs_lines["order"] == ord] = plot.lines

save_as_idl(cs_lines, "cs_lines", "xshooter_2D.sav")

np.savez("cs_lines.npz", cs_lines=cs_lines)

pass
# cs_lines = np.rec.fromarrays(, dtype=dtype)
