import numpy as np
import logging
from scipy.linalg import solve_banded
import sparse

from util import make_index

def slitfunc(
    img, ycen, lambda_sp=0, lambda_sf=0.1, osample=1, threshold=1e-5, max_iterations=20
):
    """ Python implementation of the C slitfunction decomposition algorithm of Nikolai, mainly for testing """
    im = np.ma.getdata(img)
    mask = np.ma.getmaskarray(img)
    mask = (~mask).astype(int)

    nrows, ncols = img.shape
    ny = osample * (nrows + 1) + 1
    nd = 2 * osample + 1
    step = 1 / osample

    sp = np.sum(im, axis=0) #/ (img.size - np.ma.count_masked(img))
    #sf = np.zeros(ny)
    #model = np.zeros_like(im)
    #unc = np.zeros(ncols)

    E = np.zeros(ncols)
    # sp_old = np.zeros(ncols)
    Aij = np.zeros((nd, ny))
    Adiag = np.zeros((3, ncols))
    bj = np.zeros(ny)

    omega = np.zeros((ny, nrows, ncols))

    # Populate omega

    iy = np.arange(ny)
    iy2 = np.floor((1 - ycen) * osample).astype(int) - 1
    iy1 = iy2 - osample

    d1 = np.where(iy2 == 0, step, np.where(iy1 == 0, 0, ycen % step))
    d2 = step - d1

    iy1 = iy1[:, None] + osample * np.arange(1, nrows+1, 1)[None, :]
    iy2 = iy2[:, None] + osample * np.arange(1, nrows+1, 1)[None, :]

    for x in range(ncols):
        omega[iy[:, None] == iy1[None, x, :], x] = d1[x]
        omega[
            (iy[:, None] > iy1[None, x, :]) & (iy[:, None] < iy2[None, x, :]), x
        ] = step
        omega[iy[:, None] == iy2[None, x, :], x] = d2[x]

    # This is constant and can therefore be calculated beforehand (it also takes quite some time so its worthwhile)
    omega_tmp = np.zeros((ny, nd, ncols, nrows))
    for iy in range(ny):
        mx = iy - osample if iy > osample else 0
        mn = iy + osample + 1 if iy + osample < ny else ny
        omega_tmp[iy, mx - iy + osample : mn - iy + osample] = (
            omega[iy] * omega[mx:mn]
        ).swapaxes(1, 2)

    # Loop through sf, sp reconstruction until convergence is reached
    for iteration in range(max_iterations):
        ##  Compute slit function sf

        # TODO: this takes almost 90% of the time
        # Fill in band-diagonal SLE array and the RHS
        for iy in range(ny):
            mx = iy - osample if iy > osample else 0
            mn = iy + osample + 1 if iy + osample < ny else ny
            tmp = omega_tmp[iy, mx - iy + osample : mn - iy + osample]
            tmp = np.sum(tmp * mask.T, axis=2)  # This summation is the worst offender
            Aij[mx - iy + osample : mn - iy + osample, iy] = np.sum(
                tmp * sp * sp, axis=1
            )

            bj[iy] = np.sum(omega[iy] * mask * im * sp)
        diag_tot = np.sum(Aij[osample, :])

        ## Scale regularization parameters */

        lamb = lambda_sf * diag_tot / ny * osample

        ## Add regularization parts for the SLE matrix */
        # Franklin regularization? (A + lambda * D) x = b

        # Main diagonal  */
        Aij[osample, 0] += lamb
        Aij[osample, 1:-1] += lamb * 2
        Aij[osample, -1] += lamb
        # Upper diagonal */
        Aij[osample + 1, :-1] -= lamb
        # Lower diagonal */
        Aij[osample - 1, 1:] -= lamb

        # Solve the system of equations */
        bj = solve_banded((osample, osample), Aij, bj)

        # Normalize the slit function */
        sf = bj / np.sum(bj) * osample

        #  Compute spectrum sp */
        tmp = np.sum(omega * sf[:, None, None], axis=0)
        Adiag[1, :] = np.sum(tmp * tmp * mask, axis=0)
        E = np.sum(tmp * im * mask, axis=0)

        sp_old = sp
        if lambda_sp > 0.:
            lamb = lambda_sp * np.sum(sp) / ncols

            Adiag[0, 1:] -= lamb

            Adiag[1, 0] += lamb
            Adiag[1, 1:-1] += 2 * lamb
            Adiag[1, -1] += lamb
            
            Adiag[2, :-1] -= lamb

            sp = solve_banded((1, 1), Adiag, E)
        else:
            sp = E / Adiag[1]

        # Compute the model */
        # tmp = np.sum(omega * sf[:, None, None], axis=0)
        model = tmp * sp[None, :]

        # Compare model and data */
        tmp = (model - im) * mask
        dev = np.std(tmp)

        # Adjust the mask marking outlyers */
        idx_bad = np.abs(model - im) > 6 * dev
        mask[idx_bad] = 0
        mask[~idx_bad] = 1

        # Compute the change in the spectrum */
        sp_max = np.max(sp)
        sp_change = np.max(np.abs(sp - sp_old))

        # logging.debug("Iteration: %i", iteration)

        ## Check the convergence */
        if sp_change < threshold * sp_max:
            break

    logging.debug(
        "iterations = %i, sp_max = %f, sp_change = %f", iteration, sp_max, sp_change
    )
    tmp = (model - im) * mask
    unc = np.sum(tmp * tmp, axis=0)
    unc = np.sqrt(unc * nrows)

    mask = ~mask.astype(bool)
    return sp, sf, model, unc, mask

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.io import readsav
    from scipy.signal import gaussian

    spec = 10 + 2 * np.sin(np.linspace(0, 40*np.pi, 100)) #np.linspace(5, 20, num=40)
    slitf = gaussian(40, 2)[:, None] + 1
    img = spec[None, :] * slitf
    ycen = np.linspace(-2, 3, 50)

    for i in range(img.shape[1]):
        img[:, i] = np.roll(img[:, i], -i//8)
    img = img[10:-10, :50]

    sp, sl, model, unc = slitfunc(img, ycen)

    plt.subplot(211)
    plt.imshow(img)
    plt.title("Observation")

    plt.subplot(212)
    plt.imshow(model, vmin=8)
    plt.title("Model")
    plt.show()



    plt.subplot(211)
    plt.plot(sp)
    plt.title("Spectrum")

    plt.subplot(212)
    plt.plot(sl)
    plt.title("Slitfunction")
    plt.show()
