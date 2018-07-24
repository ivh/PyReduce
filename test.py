import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian
import extract


def test_extract_spectrum():
    nx, ny = 1000, 50
    swath_width = 202

    img = np.zeros((ny, nx))
    spec = np.linspace(5, 7, num=nx)
    slitf = gaussian(ny, ny/8)
    img[:, :] = spec[None, :]  * slitf[:, None]
    # img[5:15, :] =  (5 + np.sin(np.linspace(0, 4*np.pi, num=nx))[None, :]) * gaussian(10, 5)[:, None]
    ycen = np.full(nx, ny/2)
    ylow, yhigh = ny//4, ny//4
    xlow, xhigh = 0, nx

    shear = np.full(nx, 0.)

    head = {"e_readn":0, "e_gain":1, "e_drk":0}
    orders = np.array([[(ny-1)/2]])

    spec, sunc = extract.extract(img, head, orders, shear=shear, swath_width=swath_width, plot=True)


    # assert np.all(np.abs(np.diff(spec / img[9, :])) < 1e-14)
    plt.ioff()
    plt.close()

    #plt.plot(slitf)
    #plt.show()

    plt.plot(np.nan_to_num(spec[0]))
    plt.plot(img[ny//2, :] * np.nanmax(spec) / np.max(img[ny//2]))
    plt.show()
    pass


if __name__ == "__main__":
    test_extract_spectrum()
