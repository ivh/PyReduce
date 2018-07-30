import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

from skimage import transform as tf
from scipy.signal import gaussian
import extract

# import clib.build_extract
# clib.build_extract.build()


def test_extract_spectrum():
    nx, ny = 1000, 50
    swath_width = 213

    img = np.zeros((ny, nx))
    # spec = np.linspace(5, 7, num=nx)
    spec = 5 + np.sin(np.linspace(0, 20 * np.pi, nx))
    slitf = gaussian(ny, ny / 8)
    img[:, :] = spec[None, :] * slitf[:, None] + np.random.randn(ny, nx) * 0.01
    # img[5:15, :] =  (5 + np.sin(np.linspace(0, 4*np.pi, num=nx))[None, :]) * gaussian(10, 5)[:, None]

    # shear
    shear = 0
    afine_tf = tf.AffineTransform(shear=-shear)
    img = tf.warp(img, inverse_map=afine_tf)

    head = {"e_readn": 0, "e_gain": 1, "e_drk": 0}
    orders = np.array([[(ny - 1) / 2]])
    column_range = [[100, nx - 100]]

    spec, sunc = extract.extract(
        img, head, orders, shear=shear, column_range=column_range, plot=False
    )

    spec_vert, sunc = extract.extract(
        img, head, orders, column_range=column_range, plot=False
    )

    # assert np.all(np.abs(np.diff(spec / img[9, :])) < 1e-14)
    plt.ioff()
    plt.close()

    plt.plot(spec[0])
    plt.plot(spec_vert[0])
    plt.show()

    plt.plot(img[ny // 2, :] * np.nanmax(spec_vert) / np.max(img[ny // 2]))
    plt.plot(np.nan_to_num(spec_vert[0]))
    plt.show()
    pass

def compare_idl_python():
    fname_idl = "./Test/UVES/HD132205/reduced/UVES.2010-04-02T09_28_05.650.fits.sp.ech"
    fname_py = "./Test/UVES/HD132205/reduced/2010-04-02/Reduced_middle/UVES.2010-04-02T09_28_05.650.ech"

    idl = fits.open(fname_idl)[1].data[0]
    py = fits.open(fname_py)[1].data[0]

    for i in range(20):
        plt.plot(idl["spec"][i] - py["spec"][i+1] , label="IDL - Python")
        #plt.plot(py["spec"][i+1], label="Python")
        plt.legend(loc="best")
        plt.show()


if __name__ == "__main__":
    #test_extract_spectrum()
    compare_idl_python()
