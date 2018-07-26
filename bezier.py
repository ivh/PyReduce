"""
Handle Bezier interpolation just like bezier_init and bezier_interp in IDL(SME)
"""

import numpy as np

# TODO implement using scipy BSpline
# from scipy.interpolate import BSpline


def __init__(X, Y):
    """
     Computes automatic control points for cubic Bezier splines

     If we define for points x_a and x_b along a ray:
       u = (x - x_a)/(x_b - x_a)
     then any function can be fit with a Bezier spline as
       f(u) = f(x_a)*(1 - u)^3 + 3*C0*u*(1-u)^2 + 3*C1*u^2*(1-u) + f(x_b)*u^3
     where C0 and C1 are the local control parameters.

     Control parameter1 for interval [x_a, x_b] are computed as:
       C0 = f(x_a) + delta/3*D'_a
     and
       C1 = f(x_b) - delta/3*D'_b

       If D(b-1/2)*D(b+1/2) > 0 then
         D'_b  = D(b-1/2)*D(b+1/2) / (alpha*D(b+1/2) + (1-alpha)*D(b-1/2))
       Else
         D'_b  = 0

       D(b-1/2) = [f(x_b) - f(x_a)] / delta
       D(b+1/2) = [f(x_c) - f(x_b)] / delta'
       alpha    = [1 + delta'/(delta + delta')]/3
       delta    = x_b - x_a
       delta'   = x_c - x_b

     For the first and the last step we assume D(b-1/2)=D(b+1/2) and, therefore,
     D'_b = D(b+1/2) for the first point and
     D'_b = D(b-1/2) for the last point

    The actual interpolation is split in two parts. This INIT subroutine
    computes the array D'_b
    """

    N = len(X)
    if np.any(np.diff(X) < 0):
        raise Exception(
            "Arrays X and Y in the call to BEZIER_INIT should be sorted so that X is increasing"
        )

    i = np.where(X[1:] == X[:-1])
    if i[0].shape[0] > 0:
        raise Exception(
            "Array X in the call to BEZIER_INIT should not have identical values"
        )

    Y2 = np.copy(X)
    H2 = X[1] - X[0]
    DER2 = (Y[1] - Y[0]) / H2
    Y2[0] = DER2
    for I in range(1, N - 1):
        H1 = H2
        DER1 = DER2
        H2 = X[I + 1] - X[I]
        DER2 = (Y[I + 1] - Y[I]) / H2
        ALPHA = (1 + H2 / (H1 + H2)) / 3
        if DER1 * DER2 > 0:
            Y2[I] = DER1 * DER2 / (ALPHA * DER2 + (1 - ALPHA) * DER1)
        else:
            Y2[I] = 0.
    Y2[N - 1] = DER2
    return Y2


def interpolate(XA, YA, X):
    """
    Performs cubic Bezier spline interpolation
    IMPORTANT: the XA array must be monotonic!!!

    Parameters:
    ----------
    XA : array
        old x values
    YA : array
        old y values
    X : array
        new x values
    """
    Y2A = __init__(XA, YA)

    N = len(XA)

    ii = np.where((X >= np.min(XA)) & (X <= np.max(XA)))
    if ii[0].shape[0] == 0:
        raise Exception("points outside range")

    # TODO find better way to do this
    KLO = [np.argmax(XA > xi) - 1 for xi in X]
    KLO = np.clip(KLO, 0, (N - 2))

    KHI = KLO + 1
    H = XA[KHI] - XA[KLO]
    Y1 = YA[KLO]
    Y2 = YA[KHI]

    A = (XA[KHI] - X) / H
    B = (X - XA[KLO]) / H
    C0 = Y1 + H / 3 * Y2A[KLO]
    C1 = Y2 - H / 3 * Y2A[KHI]
    Y = A * A * A * Y1 + 3 * A * A * B * C0 + 3 * A * B * B * C1 + B * B * B * Y2
    return Y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, endpoint=False)
    y = np.sin(x)
    xa = np.linspace(0, 9, 20)
    ya = interpolate(x, y, xa)
    plt.plot(x, y, label="old")
    plt.plot(xa, ya, label="new")
    plt.legend(loc="best")
    plt.show()
