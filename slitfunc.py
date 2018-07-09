import numpy as np
from scipy.linalg import solve_banded

def slitfunc(img, ycen, lambda_sp=0, lambda_sf=0.1, osample=1):
    """ Python implementation of the C slitfunction decomposition algorithm of Nikolai, mainly for testing """
    im = img.data
    mask = (~img.mask).astype(int)

    nrows, ncols = img.shape
    ny = osample * (nrows + 1) + 1
    nd = 2 * osample + 1
    step = 1 / osample

    sp = np.sum(img, axis=0).data / (img.size - np.ma.count_masked(img))
    sf = np.zeros(ny)
    model = np.zeros((ncols, nrows))
    unc = np.zeros(ncols)

    E = np.zeros(ncols)
    sp_old = np.zeros(ncols)
    Aij = np.zeros((ny, ny))
    Adiag = np.zeros((3, ncols))
    bj = np.zeros(ny)
    omega = np.zeros((ny, nrows, ncols))

    # Populate omega

    iy = np.arange(ny)
    for x in range(ncols):
        iy2 = (1 - ycen[x]) * osample
        iy1 = iy2 - osample
        if iy2 == 0:
            d1 = step
        elif iy1 == 0:
            d1 = 0.e0
        else:
            d1 = ycen[x] % step
        d2 = step - d1

        iy1 = iy1 + osample * np.arange(nrows)
        iy2 = iy2 + osample * np.arange(nrows)

        omega[iy[:, None] == iy1[None, :], x] = d1
        omega[(iy[:, None] > iy1[None, :]) & (iy[:, None] < iy2[None, :]), x] = step
        omega[iy[:, None] == iy2[None, :], x] = d2

    ## Loop through sf , sp reconstruction until convergence is reached */
    iteration = 0
    sp_change, sp_max = 1, 0
    while iteration < 20 and sp_change > 1.e-5 * sp_max:
        iteration += 1
        ##  Compute slit function sf */

        ## Fill in band-diagonal SLE array and the RHS */
        #TODO more efficient way of doing this!
        for iy in range(ny):
            for jy in range(max(iy - osample, 0), min(iy + osample, ny - 1) + 1):
                Aij[iy, jy - iy + osample] = np.sum(
                    np.sum(omega[iy] * omega[jy] * mask, axis=0) * sp * sp
                )

            bj[iy] = np.sum(np.sum(omega[iy] * mask * im, axis=0) * sp)

        diag_tot = np.sum(Aij[:, osample])

        ## Scale regularization parameters */

        lamb = lambda_sf * diag_tot / ny

        ## Add regularization parts for the SLE matrix */
        # Franklin regularization? (A + lambda * D) x = b

        # Main diagonal  */
        Aij[0, osample] += lamb
        Aij[1:-1, osample] += lamb * 2
        Aij[-1, osample] += lamb

        # Upper diagonal */
        Aij[0, osample + 1] -= lamb
        Aij[1:-1, osample + 1] -= lamb

        # Lower diagonal */
        Aij[1:-1, osample - 1] -= lamb
        Aij[-1, osample - 1] -= lamb

        # Solve the system of equations */
        tmp = (nd - 1) // 2
        bj = solve_banded((tmp, tmp), Aij[:, :nd].T, bj)

        # Normalize the slit function */
        sf = np.copy(bj) / np.sum(bj) * osample

        #  Compute spectrum sp */
        sum = np.sum(omega * sf[:, None, None], axis=0)
        Adiag[1, :] = np.sum(sum * sum * mask, axis=0)
        E = np.sum(sum * img, axis=0)

        if lambda_sp > 0.e0:
            norm = 0.e0
            sp_old = np.copy(sp)
            norm = np.sum(sp) / ncols

            lamb = lambda_sp * norm
            Adiag[0, 0] = 0.e0
            Adiag[1, 0] += lamb
            Adiag[2, 0] -= lamb
            Adiag[0, 1:-1] -= lamb
            Adiag[1, 1:-1] += 2.e0 * lamb
            Adiag[2, 1:-1] -= lamb
            Adiag[0, -1] -= lamb
            Adiag[2, -1] += lamb
            Adiag[3, -1] = 0.e0

            E = solve_banded((1, 1), Adiag, E)
            sp = np.copy(E)
        else:
            sp_old = np.copy(sp)
            sp = E / Adiag[1]

        # Compute the model */
        model = np.sum(omega * sf[:, None, None], axis=0) * sp[None, :]

        # Compare model and data */

        #sum = np.sum(mask * (model - im) * (model - im))
        #isum = np.sum(mask)
        #dev = np.sqrt(sum / isum)
        dev = np.std(model - img)

        # Adjust the mask marking outlyers */

        mask[np.abs(model - img) > 6 * dev] = 0
        mask[np.abs(model - img) <= 6 * dev] = 1

        # Compute the change in the spectrum */
        sp_max = np.max(sp)
        sp_change = np.max(np.abs(sp - sp_old))
        ## Check the convergence */

    unc = np.sum((model - img) * (model - img), axis=0)
    unc = np.sqrt(unc * nrows)

    model.shape = img.shape
    img.mask = ~mask.astype(bool)
    return sp, sf, model, unc
