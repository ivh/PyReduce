import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian
import extract


def test_extract_spectrum():
    nx, ny = 50, 20
    img = np.zeros((ny, nx))
    img[5:15, :] = np.linspace(5, 7, num=nx)[None, :]  * gaussian(10, 5)[:, None]
    # img[5:15, :] =  (5 + np.sin(np.linspace(0, 4*np.pi, num=nx))[None, :]) * gaussian(10, 5)[:, None]
    ycen = np.full(nx, 10)
    ylow, yhigh = 7, 7
    xlow, xhigh = 0, nx
    swath_width = 15

    spec, slitf, model, sunc = extract.extract_spectrum(
        img, ycen, ylow, yhigh, xlow, xhigh, swath_width=swath_width, plot=True
    )

    # assert np.all(np.abs(np.diff(spec / img[9, :])) < 1e-14)

    plt.plot(slitf)
    plt.show()

    plt.plot(spec)
    plt.plot(img[9, :] * 10 + 5)
    plt.show()
    pass


if __name__ == "__main__":
    test_extract_spectrum()
