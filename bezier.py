"""
Handle Bezier interpolation just like bezier_init and bezier_interp in IDL(SME)
"""

import numpy as np
import scipy.interpolate

def interpolate(x_old, y_old, x_new):
    knots, coef, order = scipy.interpolate.splrep(x_old, y_old)
    y_new = scipy.interpolate.BSpline(knots, coef, order)(x_new)
    return y_new

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, endpoint=False)
    y = np.sin(x)
    xa = np.linspace(0, 9, 100)
    ya = interpolate(x, y, xa)
    plt.plot(x, y, label="old")
    plt.plot(xa, ya, label="new")
    plt.legend(loc="best")
    plt.show()
