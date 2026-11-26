/* slitdec with the spectrum optionally oversampled along x as well as the slit
   function along y ("slitdec_ox").

   N. Piskunov's slitdec_ox (github.com/nikolai-piskunov/slitdec_ox), derived from
   CharSlit slitdec.c c878d00 and taken back as slitdec.c in November 2026. With
   osample_x = 1 and no mu_x it reproduces that slitdec() up to the tilt reference
   point (see zeta_tensors_ox); on untilted swaths bit for bit.
   What osample_x > 1 and the new arguments add:
   - The spectrum is solved on a fine grid of osample_x bins per detector column
     (sP_fine, ncols * osample_x values, a piecewise-constant density in the same
     units as sP). A sub-pixel of the slit function at column x, shifted by the
     slit curvature, contributes to a detector pixel with the overlap of each
     shifted fine bin with that pixel (box integration). osample_x = 1 is the
     historic two-column linear weighting.
   - The penalties are sums of k-th differences in physical units:
         sum_k mu_x[k-1] * F * integral (d^k sP / dx^k)^2 dx,   x in detector pixels,
     with F the median data curvature per pixel of the sP system, and the same in y
     for the slit function (mu_y; none given: slitdec's lambda_sL). The filter is
     then 1 / (1 + sum_k mu_k nu^2k) whatever osample, osample_x, flux and weights.
     mu1 and mu2 together cannot ring if mu1 >= 2 sqrt(mu2). For x, mu3 = 1.64e-4
     alone whitens the extraction noise for any tilt (continuum model; see
     ox_geometry.py, penalty_design.py); in y, k = 1 is best (sl_penalty_study.py).
     No x penalty with osample_x > 1 leaves sub-pixel structure unconstrained where
     the tilt does not sample it.
   - sP (out, ncols) is the detector-pixel spectrum: the mean of the fine bins of
     each column, i.e. what a vertical slit would record. sP_fine (out, may be
     NULL) receives the fine solution.
   - info[5] (if the caller allocates 7 values) reports the number of zeta
     entries dropped for lack of room (should be 0), info[6] the sP band width.
   Everything else (weights, outlier rejection, preset slit function,
   uncertainty estimate, convergence test) follows slitdec(). */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include "slitdec.h"

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

#ifndef REGULARIZE_DIAGONAL
#define REGULARIZE_DIAGONAL 1
#endif

/* Index helpers (same layouts as slitdec.c). The zeta list length per pixel
   is a runtime value, mzz, which must be in scope wherever zeta_index is used. */
#define zeta_index(x, y, z) ((size_t)(z) + (size_t)(x) * mzz + (size_t)(y) * mzz * ncols)
#define mzeta_index(x, y) ((x) + (y) * ncols)
#define curve_index(x, y) ((x) * 6 + (y))
#define a_index(x, y) ((size_t)(x) * nd + (y))
#define r_index(i) (i)
#define laij_index(x, y) ((size_t)(x) * (4 * osample + 1) + (y))
#define paij_index(x, y) ((size_t)(x) * nx + (y))
#define im_index(x, y) ((y) * ncols + (x))

static int bandsol(double *a, double *r, int n, int nd)
{
    /*
    bandsol solves a sparse system of linear equations with band-diagonal matrix.
    Band is assumed to be symmetric relative to the main diaginal.

    ..math:

        A * x = r

    Parameters
    ----------
    a : double array of shape [n,nd]
        The left-hand-side of the equation system
        The main diagonal should be in a(*,nd/2),
        the first lower subdiagonal should be in a(1:n-1,nd/2-1),
        the first upper subdiagonal is in a(0:n-2,nd/2+1) etc.
        For example:
                / 0 0 X X X \
                | 0 X X X X |
                | X X X X X |
                | X X X X X |
            A = | X X X X X |
                | X X X X X |
                | X X X X X |
                | X X X X 0 |
                \ X X X 0 0 /
    r : double array of shape [n]
        the right-hand-side of the equation system
    n : int
        The number of equations
    nd : int
        The width of the band (3 for tri-diagonal system). Must be an odd number.

    Returns
    -------
    code : int
        0 on success, -1 on incorrect size of "a" and -4 on degenerate matrix.
    */
    double aa;
    int i, j, k;

#if DEBUG
    _n = n;
    _nd = nd;
#endif

    /* Forward sweep */
    for (i = 0; i < n - 1; i++)
    {
        aa = a[a_index(i, nd / 2)];
#if DEBUG
        if (aa == 0)
        {
            printf("1, index: %i, %i\n", i, nd / 2);
            aa = 1;
        }
#endif
        r[r_index(i)] /= aa;
        for (j = 0; j < nd; j++)
            a[a_index(i, j)] /= aa;
        for (j = 1; j < min(nd / 2 + 1, n - i); j++)
        {
            aa = a[a_index(i + j, nd / 2 - j)];
            r[r_index(i + j)] -= r[r_index(i)] * aa;
            for (k = 0; k < nd - j; k++)
                a[a_index(i + j, k)] -= a[a_index(i, k + j)] * aa;
        }
    }

    /* Backward sweep */
    aa = a[a_index(n - 1, nd / 2)];
#if DEBUG
    if (aa == 0)
    {
        printf("3, index: %i, %i\n", 0, nd / 2);
        aa = 1;
    }
#endif
    r[r_index(n - 1)] /= aa;
    for (i = n - 1; i > 0; i--)
    {
        for (j = 1; j <= min(nd / 2, i); j++)
            r[r_index(i - j)] -= r[r_index(i)] * a[a_index(i - j, nd / 2 + j)];
        r[r_index(i - 1)] /= a[a_index(i - 1, nd / 2)];
    }

    aa = a[a_index(0, nd / 2)];
#if DEBUG
    if (aa == 0)
    {
        printf("4, index: %i, %i\n", 0, nd / 2);
        aa = 1;
    }
#endif
    r[r_index(0)] /= aa;
    return 0;
}

// Fast median/percentile via quickselect.
// Algorithm from Numerical recipes in C of 1992
// see http://ndevilla.free.fr/median/median/
#define ELEM_SWAP(a, b)          \
    {                            \
        register double t = (a); \
        (a) = (b);               \
        (b) = t;                 \
    }

static double quick_select_median(double arr[], unsigned int n)
{
    int low, high;
    int median;
    int middle, ll, hh;

    low = 0;
    high = n - 1;
    median = (low + high) / 2;
    for (;;)
    {
        if (high <= low) /* One element only */
            return arr[median];
        if (high == low + 1)
        { /* Two elements only */
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]);
            return arr[median];
        }
        /* Find median of low, middle and high items; swap into position low */
        middle = (low + high) / 2;
        if (arr[middle] > arr[high])
            ELEM_SWAP(arr[middle], arr[high]);
        if (arr[low] > arr[high])
            ELEM_SWAP(arr[low], arr[high]);
        if (arr[middle] > arr[low])
            ELEM_SWAP(arr[middle], arr[low]);
        /* Swap low item (now in position middle) into position (low+1) */
        ELEM_SWAP(arr[middle], arr[low + 1]);
        /* Nibble from each end towards middle, swapping items when stuck */
        ll = low + 1;
        hh = high;
        for (;;)
        {
            do
                ll++;
            while (arr[low] > arr[ll]);
            do
                hh--;
            while (arr[hh] > arr[low]);
            if (hh < ll)
                break;
            ELEM_SWAP(arr[ll], arr[hh]);
        }
        /* Swap middle item (in position low) back into correct position */
        ELEM_SWAP(arr[low], arr[hh]);
        /* Re-set active partition */
        if (hh <= median)
            low = ll;
        if (hh >= median)
            high = hh - 1;
    }
}

/* Quickselect for arbitrary percentile (0-100) */
static double quick_select_percentile(double arr[], unsigned int n, double percentile)
{
    int low, high;
    int target;
    int middle, ll, hh;

    low = 0;
    high = n - 1;
    target = (int)((percentile / 100.0) * (n - 1));
    for (;;)
    {
        if (high <= low)
            return arr[target];
        if (high == low + 1)
        {
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]);
            return arr[target];
        }
        middle = (low + high) / 2;
        if (arr[middle] > arr[high])
            ELEM_SWAP(arr[middle], arr[high]);
        if (arr[low] > arr[high])
            ELEM_SWAP(arr[low], arr[high]);
        if (arr[middle] > arr[low])
            ELEM_SWAP(arr[middle], arr[low]);
        ELEM_SWAP(arr[middle], arr[low + 1]);
        ll = low + 1;
        hh = high;
        for (;;)
        {
            do
                ll++;
            while (arr[low] > arr[ll]);
            do
                hh--;
            while (arr[hh] > arr[low]);
            if (hh < ll)
                break;
            ELEM_SWAP(arr[ll], arr[hh]);
        }
        ELEM_SWAP(arr[low], arr[hh]);
        if (hh <= target)
            low = ll;
        if (hh >= target)
            high = hh - 1;
    }
}
/* d1 must come from the same floor(ycen * osample) that picks iy1/k0. With
   fmod, an exact subpixel boundary (or one fmod rounds differently, e.g.
   ycen=0.5 at osample=10) paired the index of one side with the weight of
   the other, shifting that column's slit sampling by a whole subpixel. */
static double subpixel_d1(double ycen, int osample)
{
    const double step = 1.e0 / osample;
    double d1 = ycen - floor(ycen * osample) * step;
    if (d1 < 0)
        d1 = 0;
    if (d1 > step)
        d1 = step;
    return d1;
}
/* ---------- Fast path for exactly flat geometry ----------
   When every horizontal shift is exactly zero (all curvature coefficients
   and all slitdeltas are 0.0), pixel (x,y) receives exactly the subpixels
   iy = k0col[x] + y*osample .. + osample of its own column, with weights
   (d1, step, ..., step, step-d1) where d1 = d1col[x]. The zeta tensor is
   then fully determined by ycen, so the SLE fills, the model and the
   uncertainty pass are computed directly, without building or streaming
   zeta (by far the largest array). All sums are algebraically identical
   to the general path; accumulation order differs only at the rounding
   level (same class of reordering as the Round 1/2 optimizations). */

static void flat_setup(int ncols, int osample, const double *ycen,
                       int *k0col, double *d1col)
{
    for (int x = 0; x < ncols; x++)
    {
        /* same expressions as in zeta_tensors: k0col[x] is the first
           subpixel of row y=0, i.e. iy1 after the first `iy1 += osample` */
        k0col[x] = osample - floor(ycen[x] * osample);
        d1col[x] = subpixel_d1(ycen[x], osample);
    }
}

/* sL system: one pixel contributes the outer product of
   sP[x] * (d1, s, ..., s, s-d1) at band rows k0..k0+osample. Per detector
   row y and per k0-class c the weight products are polynomials in d1 of
   degree <= 2, so all contributions of one row collapse into 5 moments
   per class: M{0,1,2} = sum mask*sP^2*d1^{0,1,2}, R{0,1} = sum mask*im*sP*d1^{0,1}. */
/* Fills rows y0..y1-1; band-row indices are offset by lo so the target can
   be a partial slice (lo = 0 with the full matrices for the sequential
   call). */
static void flat_fill_sL(int ncols, int y0, int y1, int osample,
                         const double *im, const unsigned char *mask, const double *wpix,
                         const double *sP, const int *k0col, const double *d1col,
                         double *scratch, int lo, double *l_Aij, double *l_bj)
{
    const double s = 1.e0 / osample;
    const int bw = 4 * osample + 1;
    const int nc = osample + 1; /* classes c = 1..osample, indexed directly */
    /* Four interleaved accumulator sets (combined per row) break the
       serial dependency chains through the per-class sums; this only
       reorders the additions at the rounding level. */
    double *M0 = scratch, *M1 = M0 + 4 * nc, *M2 = M1 + 4 * nc,
           *R0 = M2 + 4 * nc, *R1 = R0 + 4 * nc;

    for (int y = y0; y < y1; y++)
    {
        for (int c = 0; c < 4 * nc; c++)
            M0[c] = M1[c] = M2[c] = R0[c] = R1[c] = 0.e0;
        const double *imrow = im + (size_t)y * ncols;
        const unsigned char *mrow = mask + (size_t)y * ncols;
        for (int x = 0; x < ncols; x++)
        {
            if (!mrow[x])
                continue;
            const int c = k0col[x] + nc * (x & 3);
            const double d = d1col[x];
            const double wv = wpix ? wpix[(size_t)y * ncols + x] : 1.e0;
            const double a = wv * sP[x] * sP[x];
            const double b = wv * imrow[x] * sP[x];
            M0[c] += a;
            M1[c] += a * d;
            M2[c] += a * d * d;
            R0[c] += b;
            R1[c] += b * d;
        }
        for (int c = 1; c <= osample; c++)
        {
            M0[c] = ((M0[c] + M0[c + nc]) + M0[c + 2 * nc]) + M0[c + 3 * nc];
            M1[c] = ((M1[c] + M1[c + nc]) + M1[c + 2 * nc]) + M1[c + 3 * nc];
            M2[c] = ((M2[c] + M2[c + nc]) + M2[c + 2 * nc]) + M2[c + 3 * nc];
            R0[c] = ((R0[c] + R0[c + nc]) + R0[c + 2 * nc]) + R0[c + 3 * nc];
            R1[c] = ((R1[c] + R1[c + nc]) + R1[c + 2 * nc]) + R1[c + 3 * nc];
        }
        for (int c = 1; c <= osample; c++)
        {
            if (M0[c] == 0.e0 && R0[c] == 0.e0)
                continue;
            const int k0 = y * osample + c - lo;
            const double m0ss = M0[c] * s * s;
            const double m1s = M1[c] * s;
            const double r0s = R0[c] * s;
            /* band row k0 (first subpixel, weight d1): pairs with all */
            double *arow = &l_Aij[(size_t)k0 * bw + 2 * osample];
            arow[0] += M2[c];                    /* d1*d1        */
            for (int d = 1; d < osample; d++)
                arow[d] += m1s;                  /* d1*s         */
            arow[osample] += m1s - M2[c];        /* d1*(s-d1)    */
            l_bj[k0] += R1[c];
            /* band rows k0+i (interior, weight s) */
            for (int i = 1; i < osample; i++)
            {
                arow = &l_Aij[(size_t)(k0 + i) * bw + 2 * osample];
                for (int d = 0; d < osample - i; d++)
                    arow[d] += m0ss;             /* s*s          */
                arow[osample - i] += m0ss - m1s; /* s*(s-d1)     */
                l_bj[k0 + i] += r0s;
            }
            /* band row k0+osample (last subpixel, weight s-d1) */
            l_Aij[(size_t)(k0 + osample) * bw + 2 * osample]
                += m0ss - 2 * m1s + M2[c];       /* (s-d1)^2     */
            l_bj[k0 + osample] += r0s - R1[c];
        }
    }
}

/* Per-row, per-class pieces of the merged slit-function value
   v(x,y) = sum_i w_i(x) * sL[k0+i] = A(y,c) + d1col[x] * B(y,c) */
static void flat_row_AB(int y, int osample, const double *sL,
                        double *A, double *B)
{
    const double s = 1.e0 / osample;
    for (int c = 1; c <= osample; c++)
    {
        const int k0 = y * osample + c;
        double mid = 0.e0;
        for (int i = 1; i < osample; i++)
            mid += sL[k0 + i];
        A[c] = s * (mid + sL[k0 + osample]);
        B[c] = sL[k0] - sL[k0 + osample];
    }
}

/* sP system: with all shifts zero every pixel maps to its own column, so
   the matrix is purely diagonal (band offset bx, which is 0 unless
   lambda_sP > 0 forces a minimum band of 1). */
static void flat_fill_sP(int ncols, int nrows, int osample, int bx, int nx,
                         const double *im, const unsigned char *mask, const double *wpix,
                         const double *sL, const int *k0col, const double *d1col,
                         double *scratch, double *p_Aij, double *p_bj)
{
    double *A = scratch, *B = A + osample + 1;
    for (int y = 0; y < nrows; y++)
    {
        flat_row_AB(y, osample, sL, A, B);
        const double *imrow = im + (size_t)y * ncols;
        const unsigned char *mrow = mask + (size_t)y * ncols;
        for (int x = 0; x < ncols; x++)
        {
            if (!mrow[x])
                continue;
            const double v = A[k0col[x]] + d1col[x] * B[k0col[x]];
            const double wv = wpix ? wpix[(size_t)y * ncols + x] : 1.e0;
            p_Aij[(size_t)x * nx + bx] += wv * v * v;
            p_bj[x] += wv * imrow[x] * v;
        }
    }
}

static void flat_model(int ncols, int y0, int y1, int osample,
                       const double *sP, const double *sL,
                       const int *k0col, const double *d1col,
                       double *scratch, double *model)
{
    double *A = scratch, *B = A + osample + 1;
    for (int y = y0; y < y1; y++)
    {
        flat_row_AB(y, osample, sL, A, B);
        double *mdrow = model + (size_t)y * ncols;
        for (int x = 0; x < ncols; x++)
            mdrow[x] = sP[x] * (A[k0col[x]] + d1col[x] * B[k0col[x]]);
    }
}

/* Uncertainty pass: per pixel the summed weight and summed squared weight
   of its (fixed) subpixel list, precomputed per column into swcol/sw2col. */
static void flat_unc(int ncols, int nrows,
                     const double *im, const double *model,
                     const unsigned char *mask,
                     const double *swcol, const double *sw2col,
                     double *unc, double *p_bj, double *norm_sq)
{
    for (int y = 0; y < nrows; y++)
    {
        const double *imrow = im + (size_t)y * ncols;
        const double *mdrow = model + (size_t)y * ncols;
        const unsigned char *mrow = mask + (size_t)y * ncols;
        for (int x = 0; x < ncols; x++)
        {
            if (!mrow[x])
                continue;
            const double tmp = imrow[x] - mdrow[x];
            unc[x] += tmp * tmp * swcol[x];
            p_bj[x] += swcol[x];
            norm_sq[x] += sw2col[x];
        }
    }
}


/* Scale positive weights to mean 1 */
static void normalise_weights(double *w, long n)
{
    double wsum = 0.e0;
    long nw = 0;
    for (long i = 0; i < n; i++)
        if (w[i] > 0.e0)
        {
            wsum += w[i];
            nw++;
        }
    if (nw > 0 && wsum > 0.e0)
        for (long i = 0; i < n; i++)
            w[i] *= nw / wsum;
}

static inline void zeta_add(zeta_ref_ox *zeta, int *m_zeta, zeta_rng_ox *z_rng,
                            int ncols, int nrows, int mzz, int *ndrop,
                            int x, int iy, int xx, int yy, double w)
{
    if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && w > 0)
    {
        if (zeta == NULL) /* counting pass: only the list lengths */
        {
            m_zeta[mzeta_index(xx, yy)]++;
            return;
        }
        const int m = m_zeta[mzeta_index(xx, yy)];
        if (m >= mzz)
        {
            (*ndrop)++;
            return;
        }
        zeta_rng_ox *zr = &z_rng[mzeta_index(xx, yy)];
        zeta[zeta_index(xx, yy, m)].x = x;
        zeta[zeta_index(xx, yy, m)].iy = iy;
        zeta[zeta_index(xx, yy, m)].w = w;
        m_zeta[mzeta_index(xx, yy)]++;
        if (iy < zr->min_iy) zr->min_iy = iy;
        if (iy > zr->max_iy) zr->max_iy = iy;
        if (x < zr->min_x) zr->min_x = x;
        if (x > zr->max_x) zr->max_x = x;
    }
}

/* Same subpixel bookkeeping along y as zeta_tensors() in slitdec.c. Along x,
   the spectrum at source column x is split into osample_x bins of width
   h = 1/osample_x; bin p covers [p*h, (p+1)*h) relative to the left edge of
   column x. Shifted by delta (the curvature at this subpixel), it lands on
   [p*h + delta, (p+1)*h + delta) and contributes w * overlap to every
   detector column it overlaps. For osample_x = 1 this is exactly the
   (1 - frac, frac) split of slitdec.c. */
static int zeta_tensors_ox(
    int ncols, int nrows, int ny, double *ycen, int *ycen_offset,
    int y_lower_lim, int osample, int osample_x,
    double *slitcurve, double *slitdeltas,
    zeta_ref_ox *zeta, int *m_zeta, zeta_rng_ox *z_rng, int mzz, int *ndrop)
{
    int x, xx, y, yy, iy, iy1, iy2, p, r;
    double step, delta, w;
    const double h = 1.e0 / osample_x;

    step = 1.e0 / osample;
    *ndrop = 0;

    for (x = 0; x < ncols; x++)
        for (y = 0; y < nrows; y++)
        {
            zeta_rng_ox *zr = &z_rng[mzeta_index(x, y)];
            m_zeta[mzeta_index(x, y)] = 0;
            zr->min_iy = INT_MAX;
            zr->max_iy = INT_MIN;
            zr->min_x = INT_MAX;
            zr->max_x = INT_MIN;
        }

    int *iy1c = malloc(ncols * sizeof(int));
    int *iy2c = malloc(ncols * sizeof(int));
    double *d1c = malloc(ncols * sizeof(double));
    double *d2c = malloc(ncols * sizeof(double));
    double *dyc = malloc(ncols * sizeof(double));

    for (x = 0; x < ncols; x++)
    {
        iy2c[x] = osample - floor(ycen[x] * osample);
        iy1c[x] = iy2c[x] - osample;
        d1c[x] = subpixel_d1(ycen[x], osample);
        d2c[x] = step - d1c[x];
        dyc[x] = ycen[x] - floor((y_lower_lim + ycen[x]) / step) * step - step;
    }

    for (y = 0; y < nrows; y++)
    {
        for (x = 0; x < ncols; x++)
        {
            const double d1 = d1c[x], d2 = d2c[x];
            double dy = dyc[x];
            iy1 = iy1c[x] += osample;
            iy2 = iy2c[x] += osample;
            dy -= step;
            for (iy = iy1; iy <= iy2; iy++)
            {
                if (iy == iy1)
                    w = d1;
                else if (iy == iy2)
                    w = d2;
                else
                    w = step;
                dy += step;
                /* Subpixel centre. The historic slitdec.c (and cr2res) used the lower
                   edge, which shifts every line by c1 / (2 * osample) px. */
                double t = dy - ycen[x] + 0.5 * step;
                delta = t * (slitcurve[curve_index(x, 1)] +
                        t * (slitcurve[curve_index(x, 2)] +
                        t * (slitcurve[curve_index(x, 3)] +
                        t * (slitcurve[curve_index(x, 4)] +
                        t *  slitcurve[curve_index(x, 5)]))))
                        + slitdeltas[iy];
                if (osample_x == 1)
                {
                    /* historic two-column split, kept verbatim for exact regression */
                    const int ix1 = delta;
                    const int ix2 = ix1 + ((delta > 0) ? 1 : (delta < 0) ? -1 : 0);
                    if (ix1 < ix2)
                    {
                        if (x + ix1 >= 0 && x + ix2 < ncols)
                        {
                            xx = x + ix1;
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            zeta_add(zeta, m_zeta, z_rng, ncols, nrows, mzz, ndrop, x, iy, xx, yy,
                                     w - fabs(delta - ix1) * w);
                            xx = x + ix2;
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            zeta_add(zeta, m_zeta, z_rng, ncols, nrows, mzz, ndrop, x, iy, xx, yy,
                                     fabs(delta - ix1) * w);
                        }
                    }
                    else if (ix1 > ix2)
                    {
                        if (x + ix2 >= 0 && x + ix1 < ncols)
                        {
                            xx = x + ix2;
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            zeta_add(zeta, m_zeta, z_rng, ncols, nrows, mzz, ndrop, x, iy, xx, yy,
                                     fabs(delta - ix1) * w);
                            xx = x + ix1;
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            zeta_add(zeta, m_zeta, z_rng, ncols, nrows, mzz, ndrop, x, iy, xx, yy,
                                     w - fabs(delta - ix1) * w);
                        }
                    }
                    else
                    {
                        xx = x + ix1;
                        yy = y + ycen_offset[x] - ycen_offset[xx];
                        zeta_add(zeta, m_zeta, z_rng, ncols, nrows, mzz, ndrop, x, iy, xx, yy, w);
                    }
                    continue;
                }
                for (p = 0; p < osample_x; p++)
                {
                    const double a = p * h + delta;
                    const double b = a + h;
                    const int r1 = (int)floor(a);
                    const int r2 = (int)ceil(b) - 1;
                    /* As slitdec.c: a bin that would partly fall off the swath
                       contributes nowhere (keeps every used bin's weight whole) */
                    if (r2 > r1 && (x + r1 < 0 || x + r2 >= ncols))
                        continue;
                    for (r = r1; r <= r2; r++)
                    {
                        const double ov = fmin(b, r + 1.e0) - fmax(a, (double)r);
                        if (ov <= 1.e-12)
                            continue;
                        xx = x + r;
                        if (xx < 0 || xx >= ncols)
                            continue;
                        yy = y + ycen_offset[x] - ycen_offset[xx];
                        zeta_add(zeta, m_zeta, z_rng, ncols, nrows, mzz, ndrop,
                                 x * osample_x + p, iy, xx, yy, w * ov);
                    }
                }
            }
            dyc[x] = dy;
        }
    }

    free(iy1c);
    free(iy2c);
    free(d1c);
    free(d2c);
    free(dyc);
    return 0;
}

/* Add mu * D_k^T D_k to a symmetric band matrix (row-major, nx = 2*bx+1,
   diagonal at column bx) of n unknowns. Requires bx >= k. */
static void add_difference_penalty(double *A, int n, int nx, int bx, int k, double mu)
{
    double c[16];
    int i, a, b;
    if (k < 1 || mu <= 0.e0 || n <= k)
        return;
    /* binomial coefficients with alternating sign: D_k row = c[0..k] */
    c[0] = 1.e0;
    for (i = 1; i <= k; i++)
        c[i] = -c[i - 1] * (k - i + 1) / i;
    for (i = 0; i + k < n; i++)
        for (a = 0; a <= k; a++)
            for (b = 0; b <= k; b++)
                A[(size_t)(i + a) * nx + bx + (b - a)] += mu * c[a] * c[b];
}

static int cmp_double(const void *p, const void *q)
{
    const double a = *(const double *)p, b = *(const double *)q;
    return (a > b) - (a < b);
}

static int cmp_pair(const void *p, const void *q)
{
    const double *a = p, *b = q;
    if (a[0] != b[0])
        return (a[0] > b[0]) - (a[0] < b[0]);
    return (a[1] > b[1]) - (a[1] < b[1]);
}

/* Fit pix_unc^2 = a * model + b over the usable pixels, a >= 0, b >= b_min (b = b_min if
   fix_b). The model is the background-subtracted object, so the variance is affine in it:
   b carries read noise and the shot noise of any subtracted background (nodding A-B),
   a = 1/(gain * frames combined). The line goes through the medians of model and
   pix_unc^2 in bins of equal pixel count along the model, as unmasked hot pixels with huge
   pix_unc dominate a plain least-squares fit. Ties sort by pix_unc^2 so that the bins are
   reproducible in the numpy and numba ports. */
static void variance_fit(const double *model, const double *pix_unc, const unsigned char *mask,
                         long npix, double b_min, int fix_b, double *a, double *b)
{
    const long nbin_max = 32, nbin_min_pix = 32;
    double sm = 0.e0, sv = 0.e0, smm = 0.e0, smv = 0.e0;
    long nk = 0, nbin;
    double *pv = malloc((size_t)(2 * npix) * sizeof(double));

    *a = 0.e0;
    *b = b_min;
    for (long i = 0; i < npix; i++)
    {
        const double u = pix_unc[i];
        if (!mask[i] || !isfinite(u) || !(u > 0.e0) || !isfinite(model[i]))
            continue;
        pv[2 * nk] = model[i];
        pv[2 * nk + 1] = u * u;
        nk++;
    }
    nbin = nk / nbin_min_pix;
    if (nbin > nbin_max)
        nbin = nbin_max;
    if (nbin < 2)
    {
        free(pv);
        return;
    }
    qsort(pv, (size_t)nk, 2 * sizeof(double), cmp_pair);

    double *tmp = malloc((size_t)(nk / nbin + 1) * sizeof(double));
    for (long k = 0; k < nbin; k++)
    {
        const long i0 = k * nk / nbin, i1 = (k + 1) * nk / nbin, nb = i1 - i0;
        const double bm = 0.5 * (pv[2 * (i0 + (nb - 1) / 2)] + pv[2 * (i0 + nb / 2)]);
        for (long i = i0; i < i1; i++)
            tmp[i - i0] = pv[2 * i + 1];
        qsort(tmp, (size_t)nb, sizeof(double), cmp_double);
        const double bv = 0.5 * (tmp[(nb - 1) / 2] + tmp[nb / 2]);
        sm += bm;
        sv += bv;
        smm += bm * bm;
        smv += bm * bv;
    }
    free(tmp);
    free(pv);

    if (!fix_b)
    {
        const double det = nbin * smm - sm * sm;
        if (det > 0.e0)
        {
            *a = (nbin * smv - sm * sv) / det;
            *b = (sv - *a * sm) / nbin;
        }
        if (*a < 0.e0)
        {
            *a = 0.e0;
            *b = sv / nbin;
        }
        if (*b >= b_min)
            return;
        *b = b_min;
    }
    if (smm > 0.e0)
        *a = fmax((smv - *b * sm) / smm, 0.e0);
}

/* Median of the positive entries v[0], v[stride], ... v[(n-1)*stride] (0 if none) */
static double median_positive(const double *v, int n, int stride)
{
    double *dg = malloc(n * sizeof(double)), med = 0.e0;
    int i, nd = 0;
    for (i = 0; i < n; i++)
        if (v[(size_t)i * stride] > 0.e0)
            dg[nd++] = v[(size_t)i * stride];
    if (nd > 0)
    {
        qsort(dg, nd, sizeof(double), cmp_double);
        med = dg[nd / 2];
    }
    free(dg);
    return med;
}

/* Damping ratio mu1 / (2 sqrt(mu2)) of a penalty mu[k-1], k = 1..kmax */
static double penalty_damping(const double *mu, int kmax)
{
    const double m1 = kmax >= 1 ? mu[0] : 0.e0, m2 = kmax >= 2 ? mu[1] : 0.e0;
    if (m2 > 0.e0)
        return m1 / (2.e0 * sqrt(m2));
    return m1 > 0.e0 ? INFINITY : 0.e0;
}

/* Nominal resolution pi / nu_half [px], 1 + sum mu_k nu_half^2k = 2 (0: no penalty) */
static double penalty_resolution(const double *mu, int kmax)
{
    double lo = 0.e0, hi = 1.e0, f;
    int i, k;
    if (kmax < 1)
        return 0.e0;
    for (i = 0; i < 200; i++)
    {
        for (f = 0.e0, k = 1; k <= kmax; k++)
            f += mu[k - 1] * pow(hi, 2 * k);
        if (f >= 1.e0)
            break;
        hi *= 2.e0;
    }
    for (i = 0; i < 100; i++)
    {
        const double mid = 0.5 * (lo + hi);
        for (f = 0.e0, k = 1; k <= kmax; k++)
            f += mu[k - 1] * pow(mid, 2 * k);
        if (f < 1.e0)
            lo = mid;
        else
            hi = mid;
    }
    return M_PI / (0.5 * (lo + hi));
}

/* Data part of the sP normal equations on the fine grid: p_Aij = sum over usable pixels of
   wfac * a a^T (full symmetric band), p_bj = sum of wfac * im * a, with a the pixel's
   coefficients on the fine bins for the current sL. wfac NULL means 1. */
static void sp_fill(int ncols, int nrows, int nf, int nx, int bx, int osample, int mzz,
                    int fast_flat, const double *im, const unsigned char *mask,
                    const double *wfac, const double *sL, const zeta_ref_ox *zeta,
                    const int *m_zeta, const zeta_rng_ox *z_rng, const int *k0col,
                    const double *d1col, double *flat_scratch, double *zw,
                    double *p_Aij, double *p_bj)
{
    int xx, yy, j, m, n;

    for (j = 0; j < nf; j++)
        p_bj[j] = 0;
    for (size_t i = 0; i < (size_t)nf * nx; i++)
        p_Aij[i] = 0;

    if (fast_flat)
        flat_fill_sP(ncols, nrows, osample, bx, nx, im, mask, wfac, sL,
                     k0col, d1col, flat_scratch, p_Aij, p_bj);
    else
        for (yy = 0; yy < nrows; yy++)
        {
            for (xx = 0; xx < ncols; xx++)
            {
                const int mz = m_zeta[mzeta_index(xx, yy)];
                if (mz <= 0 || !mask[im_index(xx, yy)])
                    continue;
                const zeta_ref_ox *zrow = &zeta[zeta_index(xx, yy, 0)];
                const double imv = im[im_index(xx, yy)];
                const double wv = wfac ? wfac[im_index(xx, yy)] : 1.e0;
                const zeta_rng_ox *zr = &z_rng[mzeta_index(xx, yy)];
                const int k0 = zr->min_x;
                const int rng = zr->max_x - k0;
                for (n = 0; n <= rng; n++)
                    zw[n] = 0.e0;
                for (m = 0; m < mz; m++)
                    zw[zrow[m].x - k0] += sL[zrow[m].iy] * zrow[m].w;
                for (m = 0; m <= rng; m++)
                {
                    const double um = wv * zw[m];
                    if (um == 0.e0)
                        continue;
                    const double *restrict uv = zw + m;
                    double *restrict arow = &p_Aij[paij_index(k0 + m, bx)];
                    const int dmax = rng - m;
                    for (n = 0; n <= dmax; n++)
                        arow[n] += um * uv[n];
                    p_bj[k0 + m] += imv * um;
                }
            }
        }

    for (m = 1; m <= bx; m++)
        for (j = 0; j < nf - m; j++)
            p_Aij[paij_index(j + m, bx - m)] = p_Aij[paij_index(j, bx + m)];
}

/* Penalties and diagonal safeguards of the sP system, added to the data part */
static void sp_penalties(double *p_Aij, int nf, int nx, int bx, int ox, int kx,
                         const double *mux, double lambda_sP)
{
    int j, m;

    if (lambda_sP > 0.e0)
    {
        const double lambda = lambda_sP;
        p_Aij[paij_index(0, bx)] += lambda;
        p_Aij[paij_index(0, bx + 1)] -= lambda;
        for (j = 1; j < nf - 1; j++)
        {
            p_Aij[paij_index(j, bx - 1)] -= lambda;
            p_Aij[paij_index(j, bx)] += lambda * 2.e0;
            p_Aij[paij_index(j, bx + 1)] -= lambda;
        }
        p_Aij[paij_index(nf - 1, bx - 1)] -= lambda;
        p_Aij[paij_index(nf - 1, bx)] += lambda;
    }

    /* Penalty mu_x * F * int (d^k sP/dx^k)^2 dx. With fine bins of width
       h = 1/ox the data curvature of one bin is F h^2, so F = median
       diagonal * ox^2, and the discrete form of the integral is
       h^(1-2k) * sum (D_k sPf)^2: coefficient mu_x * diag_med * ox^(2k+1). */
    if (kx > 0 || ox > 1)
    {
        double diag_med = median_positive(p_Aij + bx, nf, nx);
        for (m = 1; m <= kx; m++)
            add_difference_penalty(p_Aij, nf, nx, bx, m,
                                   mux[m - 1] * diag_med * pow((double)ox, 2 * m + 1));
        if (ox > 1) /* tiny ridge: keeps unsampled sub-pixel modes finite */
            for (j = 0; j < nf; j++)
                p_Aij[paij_index(j, bx)] += 1.e-9 * diag_med;
    }

#if REGULARIZE_DIAGONAL
    {
        double max_diag = 0.0;
        for (j = 0; j < nf; j++)
            if (p_Aij[paij_index(j, bx)] > max_diag)
                max_diag = p_Aij[paij_index(j, bx)];
        if (max_diag > 0.0)
        {
            double min_diag = max_diag * 1.0e-10;
            for (j = 0; j < nf; j++)
                if (p_Aij[paij_index(j, bx)] < min_diag)
                    p_Aij[paij_index(j, bx)] = min_diag;
        }
    }
#endif
}

/* Cholesky factor of a symmetric positive definite band matrix (full band, row-major,
   nx = 2*bx+1). L[i*(bx+1)+k] = L(i, i-k). Returns -1 if a pivot is not positive. */
static int band_cholesky(const double *A, int n, int nx, int bx, double *L)
{
    const int nl = bx + 1;
    for (int i = 0; i < n; i++)
        for (int k = min(bx, i); k >= 0; k--)
        {
            const int j = i - k;
            double s = A[(size_t)i * nx + bx - k];
            for (int q = k + 1; q <= min(bx, i); q++)
                s -= L[(size_t)i * nl + q] * L[(size_t)j * nl + q - k];
            if (k == 0)
            {
                if (!(s > 0.e0))
                    return -1;
                L[(size_t)i * nl] = sqrt(s);
            }
            else
                L[(size_t)i * nl + k] = s / L[(size_t)j * nl];
        }
    return 0;
}

/* Noise covariance of the spectrum, C = N^-1 M N^-1, contracted onto output bins.
   N: the sP system matrix of the last solve (data + penalties), M = A^T W V W A, the
   same fill with weight^2 * pixel variance. An output bin is the mean b of ox/out_x
   fine bins, so with y = N^-1 b: var = y^T M y, and the covariance with another bin
   is y^T M y'. One banded solve per output bin; y decays away from the bin, and both
   substitutions stop once it has fallen below 1e-10 of its maximum.
   unc[x]: sigma of the detector-pixel spectrum. cov (may be NULL): for output bin j,
   cov[j*(nlag+1)+l] = Cov(bin j, bin j+l).
   Returns -1 if N is not positive definite. */
static int propagate_unc(int ncols, int ox, int out_x, int nlag, int nf,
                         int nx, int bx, const double *N, const double *M,
                         double *unc, double *cov)
{
    const int nl = bx + 1, nb = ox / out_x, R = nlag + 1;
    const double eps = 1.e-10;
    int x, q, i, k, l, r, c, rc = 0;
    double *L = malloc((size_t)nf * nl * sizeof(double));
    double *z = malloc(nf * sizeof(double));
    double *ys = malloc((size_t)R * nf * sizeof(double));
    double *mys = malloc((size_t)R * nf * sizeof(double));
    double *yd = calloc(nf, sizeof(double)), *myd = calloc(nf, sizeof(double));
    int *rng = malloc((size_t)R * 4 * sizeof(int)); /* lo, hi, mlo, mhi; lo > hi: unused */

    if (L == NULL || z == NULL || ys == NULL || mys == NULL || yd == NULL || myd == NULL ||
        rng == NULL || band_cholesky(N, nf, nx, bx, L) != 0)
        rc = -1;
    for (k = 0; k < R && rng != NULL; k++)
    {
        rng[4 * k] = 1;
        rng[4 * k + 1] = 0;
    }

    for (x = 0; x < ncols && rc == 0; x++)
    {
        int dlo = nf, dhi = -1, dmlo = nf, dmhi = -1;
        for (q = 0; q < out_x; q++)
        {
            const int j = x * out_x + q, slot = j % R;
            double *y = ys + (size_t)slot * nf, *my = mys + (size_t)slot * nf;
            int *rg = rng + 4 * slot;
            if (cov != NULL)
                for (l = 0; l <= nlag; l++)
                    cov[(size_t)j * R + l] = 0.e0;
            const int i0 = j * nb, i1 = i0 + nb - 1;
            int lo = 0, hi = nf - 1;
            double zmax = 0.e0, ymax = 0.e0, s;

            /* L z = b */
            for (i = i0; i < nf; i++)
            {
                s = (i <= i1) ? 1.e0 / nb : 0.e0;
                for (k = 1; k <= min(bx, i - i0); k++)
                    s -= L[(size_t)i * nl + k] * z[i - k];
                z[i] = s / L[(size_t)i * nl];
                if (fabs(z[i]) > zmax)
                    zmax = fabs(z[i]);
                if (i - i1 >= bx)
                {
                    double t = 0.e0;
                    for (k = 0; k <= bx && i - k >= i0; k++)
                        if (fabs(z[i - k]) > t)
                            t = fabs(z[i - k]);
                    if (t < eps * zmax)
                    {
                        hi = i;
                        break;
                    }
                }
            }
            /* L^T y = z */
            for (i = hi; i >= 0; i--)
            {
                s = (i >= i0) ? z[i] : 0.e0;
                for (k = 1; k <= min(bx, hi - i); k++)
                    s -= L[(size_t)(i + k) * nl + k] * y[i + k];
                y[i] = s / L[(size_t)i * nl];
                if (fabs(y[i]) > ymax)
                    ymax = fabs(y[i]);
                if (i0 - i >= bx)
                {
                    double t = 0.e0;
                    for (k = 0; k <= bx; k++)
                        if (fabs(y[i + k]) > t)
                            t = fabs(y[i + k]);
                    if (t < eps * ymax)
                    {
                        lo = i;
                        break;
                    }
                }
            }
            const int mlo = max(0, lo - bx), mhi = min(nf - 1, hi + bx);
            for (r = mlo; r <= mhi; r++)
            {
                s = 0.e0;
                for (c = max(lo, r - bx); c <= min(hi, r + bx); c++)
                    s += M[(size_t)r * nx + bx + c - r] * y[c];
                my[r] = s;
            }
            rg[0] = lo; rg[1] = hi; rg[2] = mlo; rg[3] = mhi;

            if (cov != NULL)
                for (l = 0; l <= nlag && l <= j; l++)
                {
                    const int *ro = rng + 4 * ((j - l) % R);
                    const double *myo = mys + (size_t)((j - l) % R) * nf;
                    if (ro[0] > ro[1])
                        continue;
                    s = 0.e0;
                    for (r = max(lo, ro[2]); r <= min(hi, ro[3]); r++)
                        s += y[r] * myo[r];
                    cov[(size_t)(j - l) * R + l] = s;
                }
            for (r = lo; r <= hi; r++)
                yd[r] += y[r] / out_x;
            for (r = mlo; r <= mhi; r++)
                myd[r] += my[r] / out_x;
            dlo = min(dlo, lo); dhi = max(dhi, hi);
            dmlo = min(dmlo, mlo); dmhi = max(dmhi, mhi);
        }
        double v = 0.e0;
        for (r = dlo; r <= dhi; r++)
            v += yd[r] * myd[r];
        unc[x] = sqrt(v);
        for (r = dlo; r <= dhi; r++)
            yd[r] = 0.e0;
        for (r = dmlo; r <= dmhi; r++)
            myd[r] = 0.e0;
    }

    free(L); free(z); free(ys); free(mys); free(yd); free(myd); free(rng);
    return rc;
}

/* Largest horizontal shift of the slit image in the swath, in detector pixels,
   rounded up: curvature polynomial and slitdeltas, each on its own. */
static int max_shift(int ncols, int nrows, const double *slitcurve, const double *slitdeltas,
                     int osample, int y_lower_lim)
{
    const int ny = osample * (nrows + 1) + 1;
    int x, y, iy, shift = 0;
    double tmp;

    for (x = 0; x < ncols; x++)
    {
        if (slitcurve[curve_index(x, 1)] == 0.e0 && slitcurve[curve_index(x, 2)] == 0.e0 &&
            slitcurve[curve_index(x, 3)] == 0.e0 && slitcurve[curve_index(x, 4)] == 0.e0 &&
            slitcurve[curve_index(x, 5)] == 0.e0)
            continue;
        for (y = -y_lower_lim; y < nrows - y_lower_lim + 1; y++)
        {
            double y2 = y * y, y3 = y2 * y, y4 = y3 * y, y5 = y4 * y;
            tmp = ceil(fabs(y * slitcurve[curve_index(x, 1)] + y2 * slitcurve[curve_index(x, 2)] +
                            y3 * slitcurve[curve_index(x, 3)] + y4 * slitcurve[curve_index(x, 4)] +
                            y5 * slitcurve[curve_index(x, 5)]));
            shift = max(shift, tmp);
        }
    }
    for (iy = 0; iy < ny; iy++)
    {
        tmp = ceil(fabs(slitdeltas[iy]));
        shift = max(shift, tmp);
    }
    return shift;
}

int slitdec_margin(int ncols, int nrows, const double *slitcurve, const double *slitdeltas,
                   int osample, int y_lower_lim)
{
    if (y_lower_lim < 0 || y_lower_lim >= nrows)
        y_lower_lim = nrows / 2;
    return max_shift(ncols, nrows, slitcurve, slitdeltas, osample, y_lower_lim) +
           SLITDEC_EDGE;
}

int slitdec(        int ncols,
                    int nrows,
                    double *im,
                    double *pix_unc,
                    unsigned char *mask,
                    double *ycen,
                    double *slitcurve,
                    double *slitdeltas,
                    int osample,
                    double lambda_sP,
                    double lambda_sL,
                    int maxiter,
                    double kappa,
                    int use_preset,
                    int weight_mode,
                    double noise_floor,
                    double *sP,
                    double *sL,
                    double *model,
                    double *unc,
                    double *info,
                    int osample_x,
                    const double *mu_x,
                    int nk_x,
                    const double *mu_y,
                    int nk_y,
                    int y_lower_lim,
                    double *sP_fine,
                    int unc_mode,
                    int out_x,
                    int nlag,
                    double *cov,
                    double *chi2)
{
    /* The first 21 arguments are those of slitdec() before osample_x existed, in the
       same order. New ones:
       osample_x : fine spectrum bins per detector column (>= 1)
       mu_x[k-1] : spectrum penalty on the k-th derivative, k = 1..nk_x, in
                   physical units (see header comment); nk_x <= SLITDEC_OX_KMAX
       mu_y[k-1] : the same for the slit function, k = 1..nk_y. All zero (or
                   nk_y = 0): the slitdec penalty lambda_sL on first differences
                   of the y-subpixels, scaled by the mean diagonal (legacy)
       sP        : in: initial guess (ncols). out: detector-pixel spectrum
       sP_fine   : out: fine spectrum (ncols * osample_x), may be NULL
       info      : at least 5 values: success, sP change, status, iterations and
                   the margin of slitdec_margin(); info[5], info[6] are written if
                   info[4] is passed in as -7 (dropped zeta entries, sP band),
                   info[5..10] if it is passed in as -11: info[7], info[9] are
                   the damping ratio mu1/(2 sqrt(mu2)) in x and y (>= 1: the
                   penalty cannot ring), info[8], info[10] the nominal
                   resolution pi/nu_half [px] in x and y, nu_half being where
                   the penalty transfer 1/(1 + sum mu_k nu^2k) drops to 1/2. For
                   the legacy lambda_sL, info[9..10] use the equivalent mu1_y
                   (returned in info[9] as -mu1_y, since its damping is infinite).
       lambda_sP : slitdec's unscaled penalty on first differences, applied
                   between adjacent fine bins (0: off; prefer mu_x)
       unc_mode  : 0: unc from the data - model residuals (slitdec). 1: pixel variances
                   propagated through the last sP solve (propagate_unc), sL taken as
                   known; the variances are pix_unc^2, or the fitted a * model + b with
                   weight_mode 2. 2: as 1, scaled by the reduced chi^2 of the residuals.
                   Columns no unmasked pixel contributes to get NaN in unc and cov.
       out_x, nlag, cov : with unc_mode > 0 and cov != NULL, covariances on the grid of
                   out_x bins per column (out_x divides osample_x) for lags 0..nlag,
                   cov[(ncols*out_x) x (nlag+1)]
       chi2      : out, may be NULL: reduced chi^2 of the residuals against those
                   variances (unc_mode > 0), NaN if the propagation failed
       y_lower_lim : rows below the central row of the swath, i.e. the
                   trace sits at row y_lower_lim + ycen[x] - (int)ycen[x]
                   (IDL REDUCE convention). < 0 means nrows / 2 (slitdec). */
    int x, xx, y, yy, iy, n, m, nx, ny, nf, j;
    int bx;
    double norm, dev, lambda, diag_tot, ww, tmp;
    double sP_change, sP_stop, sP_med;
    int iter, delta_x, margin;
    unsigned int isum;
    int *ycen_offset;
    const int ox = osample_x < 1 ? 1 : osample_x;
    const int want_ext = (info[4] == -7 || info[4] == -11);
    const int want_ext2 = (info[4] == -11);
    double mux[SLITDEC_OX_KMAX], muy[SLITDEC_OX_KMAX];
    int kx = 0, ky = 0; /* highest order with a non-zero penalty */
    double mu1_y_legacy = 0.e0;
    int ndrop = 0;

    double *l_Aij, *l_bj, *p_Aij = NULL, *p_bj, *sPf;
    double *sP_old, *sP_diff, *norm_sq, *unc_norm;
    double *zw = NULL;
    int *zk = NULL;

    zeta_ref_ox *zeta = NULL;
    int *m_zeta = NULL;
    zeta_rng_ox *z_rng = NULL;
    /* per-pixel zeta list length, set by a counting pass over the geometry */
    int mzz = 0;

    int fast_flat;
    int *k0col = NULL;
    double *d1col = NULL, *flat_scratch = NULL, *swcol = NULL, *sw2col = NULL;

    double success, status;

    if (y_lower_lim < 0 || y_lower_lim >= nrows)
        y_lower_lim = nrows / 2;

    sP_stop = 5e-5;
    success = 1;
    status = 0;
    sP_change = INFINITY;
    ny = osample * (nrows + 1) + 1;
    nf = ncols * ox;
    for (m = 0; m < SLITDEC_OX_KMAX; m++)
    {
        mux[m] = (m < nk_x && mu_x != NULL && mu_x[m] > 0.e0) ? mu_x[m] : 0.e0;
        muy[m] = (m < nk_y && mu_y != NULL && mu_y[m] > 0.e0) ? mu_y[m] : 0.e0;
        if (mux[m] > 0.e0) kx = m + 1;
        if (muy[m] > 0.e0) ky = m + 1;
    }
    if (ky > 2 * osample) /* the sL band holds differences up to 2 * osample */
        ky = 2 * osample;

    /* delta_x: allocation, and the columns left out of the residual statistics.
       margin: the columns at either end a caller should not use (info[4]). */
    delta_x = max_shift(ncols, nrows, slitcurve, slitdeltas, osample, y_lower_lim);
    margin = delta_x + SLITDEC_EDGE;
    if (delta_x == 0 && (kx != 0 || lambda_sP != 0))
        delta_x = 1;
    if (4 * delta_x + 1 > ncols)
    {
        info[0] = 0;
        info[1] = sP_change;
        info[2] = -2;
        info[3] = 0;
        info[4] = margin;
        return -1;
    }

    /* The flat fast path only exists for the historic one-bin-per-column model */
    fast_flat = (ox == 1);
    for (x = 0; x < ncols && fast_flat; x++)
        for (m = 1; m < 6; m++)
            if (slitcurve[curve_index(x, m)] != 0.e0)
            {
                fast_flat = 0;
                break;
            }
    if (fast_flat)
        for (iy = 0; iy < ny; iy++)
            if (slitdeltas[iy] != 0.e0)
            {
                fast_flat = 0;
                break;
            }

    l_Aij = malloc((size_t)ny * (4 * osample + 1) * sizeof(double));
    l_bj = malloc(ny * sizeof(double));
    p_bj = malloc(nf * sizeof(double));
    sPf = malloc(nf * sizeof(double));
    if (fast_flat)
    {
        k0col = malloc(ncols * sizeof(int));
        d1col = malloc(ncols * sizeof(double));
        swcol = malloc(ncols * sizeof(double));
        sw2col = malloc(ncols * sizeof(double));
        flat_scratch = malloc(20 * (osample + 1) * sizeof(double));
    }
    else
    {
        m_zeta = malloc((size_t)ncols * nrows * sizeof(int));
        z_rng = malloc((size_t)ncols * nrows * sizeof(zeta_rng_ox));
        if (m_zeta != NULL && z_rng != NULL)
        {
            /* ycen is split into offset + fraction below; count with a copy */
            int *yo = malloc(ncols * sizeof(int));
            double *yc = malloc(ncols * sizeof(double));
            for (x = 0; x < ncols; x++)
            {
                yo[x] = ycen[x];
                yc[x] = ycen[x] - yo[x];
            }
            zeta_tensors_ox(ncols, nrows, ny, yc, yo, y_lower_lim, osample, ox,
                            slitcurve, slitdeltas, NULL, m_zeta, z_rng, INT_MAX, &ndrop);
            for (long i = 0; i < (long)ncols * nrows; i++)
                if (m_zeta[i] > mzz)
                    mzz = m_zeta[i];
            if (mzz < 1)
                mzz = 1;
            free(yo);
            free(yc);
            zeta = malloc((size_t)ncols * nrows * mzz * sizeof(zeta_ref_ox));
        }
        if (zeta == NULL || m_zeta == NULL || z_rng == NULL)
        {
            free(zeta); free(m_zeta); free(z_rng);
            free(l_Aij); free(l_bj); free(p_bj); free(sPf);
            info[0] = 0; info[1] = sP_change; info[2] = -4; info[3] = 0; info[4] = margin;
            return -1;
        }
    }
    ycen_offset = malloc(ncols * sizeof(int));
    sP_old = malloc(ncols * sizeof(double));
    sP_diff = malloc(ncols * sizeof(double));
    norm_sq = malloc(ncols * sizeof(double));
    unc_norm = malloc(ncols * sizeof(double));

    for (x = 0; x < ncols; x++)
    {
        ycen_offset[x] = ycen[x];
        ycen[x] = ycen[x] - ycen_offset[x];
    }

    /* Initial fine spectrum: the caller's guess, constant within a column */
    for (j = 0; j < nf; j++)
        sPf[j] = sP[j / ox];

    double *wbuf = NULL, *wpix = NULL;
    int fit_b = 0;
    const long npix = (long)nrows * ncols;
    if (weight_mode == 1)
    {
        wbuf = malloc((size_t)npix * sizeof(double));
        for (long i = 0; i < npix; i++)
        {
            const double u = pix_unc[i];
            wbuf[i] = (mask[i] && isfinite(u) && u > 0.e0) ? 1.e0 / (u * u) : 0.e0;
        }
        normalise_weights(wbuf, npix);
        wpix = wbuf;
    }
    else if (weight_mode == 3)
        wbuf = malloc((size_t)npix * sizeof(double));
    else if (weight_mode == 2)
    {
        wbuf = malloc((size_t)npix * sizeof(double));
        fit_b = !(noise_floor >= 0.e0);
        if (fit_b)
        {
            long nn = 0;
            for (long i = 0; i < npix; i++)
            {
                const double u = pix_unc[i];
                if (mask[i] && isfinite(u) && u > 0.e0)
                    wbuf[nn++] = u * u;
            }
            noise_floor = nn > 0 ? quick_select_percentile(wbuf, nn, 5.0) : 0.e0;
        }
    }

    if (fast_flat)
        flat_setup(ncols, osample, ycen, k0col, d1col);
    else
        zeta_tensors_ox(ncols, nrows, ny, ycen, ycen_offset, y_lower_lim, osample, ox,
                        slitcurve, slitdeltas, zeta, m_zeta, z_rng, mzz, &ndrop);

    /* sP band: widest fine-bin span of one pixel, at least the penalty order */
    {
        int span = 0;
        if (!fast_flat)
            for (x = 0; x < ncols; x++)
                for (y = 0; y < nrows; y++)
                {
                    if (m_zeta[mzeta_index(x, y)] <= 0)
                        continue;
                    const zeta_rng_ox *zr = &z_rng[mzeta_index(x, y)];
                    if (zr->max_x - zr->min_x > span)
                        span = zr->max_x - zr->min_x;
                }
        bx = (kx > 0 && span < kx) ? kx : span;
        if ((ox > 1 || lambda_sP > 0.e0) && bx < 1)
            bx = 1;
        nx = 2 * bx + 1;
        int zbuf = max(max(mzz, 2 * osample + 1), span + 1);
        zw = malloc(zbuf * sizeof(double));
        zk = malloc(zbuf * sizeof(int));
        p_Aij = malloc((size_t)nf * nx * sizeof(double));
    }

    if (use_preset)
    {
        norm = 0.e0;
        for (iy = 0; iy < ny; iy++)
            norm += sL[iy];
        norm /= osample;
        for (iy = 0; iy < ny; iy++)
            sL[iy] /= norm;
    }

    iter = 0;
    do
    {
        if (!use_preset)
        {
            for (iy = 0; iy < ny; iy++)
                l_bj[iy] = 0.e0;
            for (size_t i = 0; i < (size_t)ny * (4 * osample + 1); i++)
                l_Aij[i] = 0;

            if (fast_flat)
                flat_fill_sL(ncols, 0, nrows, osample, im, mask, wpix, sP, k0col, d1col,
                             flat_scratch, 0, l_Aij, l_bj);
            else
                for (yy = 0; yy < nrows; yy++)
                {
                    for (xx = 0; xx < ncols; xx++)
                    {
                        const int mz = m_zeta[mzeta_index(xx, yy)];
                        if (mz <= 0 || !mask[im_index(xx, yy)])
                            continue;
                        const zeta_ref_ox *zrow = &zeta[zeta_index(xx, yy, 0)];
                        const double imv = im[im_index(xx, yy)];
                        const double wv = wpix ? wpix[im_index(xx, yy)] : 1.e0;
                        const zeta_rng_ox *zr = &z_rng[mzeta_index(xx, yy)];
                        const int k0 = zr->min_iy;
                        const int rng = zr->max_iy - k0;
                        if (rng <= 2 * osample)
                        {
                            for (n = 0; n <= rng; n++)
                                zw[n] = 0.e0;
                            for (m = 0; m < mz; m++)
                                zw[zrow[m].iy - k0] += sPf[zrow[m].x] * zrow[m].w;
                            for (m = 0; m <= rng; m++)
                            {
                                const double um = wv * zw[m];
                                const double *restrict uv = zw + m;
                                double *restrict arow = &l_Aij[laij_index(k0 + m, 2 * osample)];
                                const int dmax = rng - m;
                                for (n = 0; n <= dmax; n++)
                                    arow[n] += um * uv[n];
                                l_bj[k0 + m] += imv * um;
                            }
                            continue;
                        }
                        int nk = 0;
                        for (m = 0; m < mz; m++)
                        {
                            const int key = zrow[m].iy;
                            const double v = sPf[zrow[m].x] * zrow[m].w;
                            for (n = 0; n < nk; n++)
                                if (zk[n] == key)
                                {
                                    zw[n] += v;
                                    break;
                                }
                            if (n == nk)
                            {
                                zk[nk] = key;
                                zw[nk++] = v;
                            }
                        }
                        for (m = 0; m < nk; m++)
                        {
                            iy = zk[m];
                            const double um = zw[m];
                            const double wum = wv * um;
                            l_Aij[laij_index(iy, 2 * osample)] += wum * um;
                            for (n = m + 1; n < nk; n++)
                            {
                                const int iyn = zk[n];
                                const int lo = min(iy, iyn);
                                const int d = abs(iyn - iy);
                                if (d > 2 * osample)
                                    continue;
                                l_Aij[laij_index(lo, d + 2 * osample)] += zw[n] * wum;
                            }
                            l_bj[iy] += imv * wum;
                        }
                    }
                }

            for (m = 1; m <= 2 * osample; m++)
                for (iy = 0; iy < ny - m; iy++)
                    l_Aij[laij_index(iy + m, 2 * osample - m)] = l_Aij[laij_index(iy, 2 * osample + m)];

            diag_tot = 0.e0;
            for (iy = 0; iy < ny; iy++)
                diag_tot += l_Aij[laij_index(iy, 2 * osample)];
            if (ky > 0 || want_ext2)
            {
                /* Penalty sum_k mu_y[k-1] * F * int (d^k sL/dy^k)^2 dy, as for sP:
                   coefficient mu * diag_med * osample^(2k+1) on k-th differences */
                double diag_med = median_positive(l_Aij + 2 * osample, ny, 4 * osample + 1);
                for (m = 1; m <= ky; m++)
                    add_difference_penalty(l_Aij, ny, 4 * osample + 1, 2 * osample, m,
                                           muy[m - 1] * diag_med * pow((double)osample, 2 * m + 1));
                if (ky == 0 && diag_med > 0.e0)
                    mu1_y_legacy = lambda_sL * diag_tot / ny / (diag_med * pow((double)osample, 3));
            }
            lambda = ky > 0 ? 0.e0 : lambda_sL * diag_tot / ny;
            if (lambda > 0.e0)
            {
            l_Aij[laij_index(0, 2 * osample)] += lambda;
            l_Aij[laij_index(0, 2 * osample + 1)] -= lambda;
            for (iy = 1; iy < ny - 1; iy++)
            {
                l_Aij[laij_index(iy, 2 * osample - 1)] -= lambda;
                l_Aij[laij_index(iy, 2 * osample)] += lambda * 2.e0;
                l_Aij[laij_index(iy, 2 * osample + 1)] -= lambda;
            }
            l_Aij[laij_index(ny - 1, 2 * osample - 1)] -= lambda;
            l_Aij[laij_index(ny - 1, 2 * osample)] += lambda;
            }
#if REGULARIZE_DIAGONAL
            {
                double max_diag = 0.0;
                for (iy = 0; iy < ny; iy++)
                    if (l_Aij[laij_index(iy, 2 * osample)] > max_diag)
                        max_diag = l_Aij[laij_index(iy, 2 * osample)];
                if (max_diag > 0.0)
                {
                    double min_diag = max_diag * 1.0e-10;
                    for (iy = 0; iy < ny; iy++)
                        if (l_Aij[laij_index(iy, 2 * osample)] < min_diag)
                            l_Aij[laij_index(iy, 2 * osample)] = min_diag;
                }
            }
#endif
            bandsol(l_Aij, l_bj, ny, 4 * osample + 1);
            norm = 0.e0;
            for (iy = 0; iy < ny; iy++)
            {
                sL[iy] = l_bj[iy];
                norm += sL[iy];
            }
            norm /= osample;
            for (iy = 0; iy < ny; iy++)
                sL[iy] /= norm;
        }

        /* Spectrum system on the fine grid */
        sp_fill(ncols, nrows, nf, nx, bx, osample, mzz, fast_flat, im, mask, wpix, sL,
                zeta, m_zeta, z_rng, k0col, d1col, flat_scratch, zw, p_Aij, p_bj);
        sp_penalties(p_Aij, nf, nx, bx, ox, kx, mux, lambda_sP);
        bandsol(p_Aij, p_bj, nf, nx);

        for (j = 0; j < nf; j++)
            sPf[j] = p_bj[j];
        for (x = 0; x < ncols; x++)
            sP_old[x] = sP[x];
        for (x = 0; x < ncols; x++)
        {
            double acc = 0.e0;
            for (m = 0; m < ox; m++)
                acc += sPf[x * ox + m];
            sP[x] = acc / ox;
        }
        for (x = 0; x < ncols; x++)
            sP_diff[x] = fabs(sP[x] - sP_old[x]);
        sP_change = quick_select_percentile(sP_diff, ncols, 99.0);
        for (x = 0; x < ncols; x++)
            sP_diff[x] = sP[x];
        sP_med = fabs(quick_select_median(sP_diff, ncols));

        if (fast_flat)
            flat_model(ncols, 0, nrows, osample, sP, sL, k0col, d1col, flat_scratch, model);
        else
            for (y = 0; y < nrows; y++)
                for (x = 0; x < ncols; x++)
                {
                    const zeta_ref_ox *zrow = &zeta[zeta_index(x, y, 0)];
                    const int mz = m_zeta[mzeta_index(x, y)];
                    double acc = 0.;
                    for (m = 0; m < mz; m++)
                        acc += sPf[zrow[m].x] * sL[zrow[m].iy] * zrow[m].w;
                    model[im_index(x, y)] = acc;
                }

        if (weight_mode == 2)
        {
            double va, vb;
            variance_fit(model, pix_unc, mask, npix, noise_floor, !fit_b, &va, &vb);
            for (long i = 0; i < npix; i++)
            {
                const double v = va * fmax(model[i], 0.e0) + vb;
                wbuf[i] = mask[i] && v > 0.e0 ? 1.e0 / v : 0.e0;
            }
            normalise_weights(wbuf, npix);
            wpix = wbuf;
        }
        else if (weight_mode == 3)
        {
            if (fast_flat)
            {
                double *A = flat_scratch, *B = A + osample + 1;
                for (y = 0; y < nrows; y++)
                {
                    flat_row_AB(y, osample, sL, A, B);
                    for (x = 0; x < ncols; x++)
                        wbuf[im_index(x, y)] = A[k0col[x]] + d1col[x] * B[k0col[x]];
                }
            }
            else
                for (y = 0; y < nrows; y++)
                    for (x = 0; x < ncols; x++)
                    {
                        const zeta_ref_ox *zrow = &zeta[zeta_index(x, y, 0)];
                        const int mz = m_zeta[mzeta_index(x, y)];
                        double acc = 0.;
                        for (m = 0; m < mz; m++)
                            acc += sL[zrow[m].iy] * zrow[m].w;
                        wbuf[im_index(x, y)] = acc;
                    }
            double vmax = 0.e0;
            for (long i = 0; i < npix; i++)
                if (mask[i] && wbuf[i] > vmax)
                    vmax = wbuf[i];
            const double vfloor = 1.e-3 * vmax;
            for (long i = 0; i < npix; i++)
                wbuf[i] = (mask[i] && vmax > 0.e0) ? 1.e0 / fmax(wbuf[i], vfloor) : 0.e0;
            normalise_weights(wbuf, npix);
            wpix = wbuf;
        }

        if (use_preset && kappa <= 0 && (weight_mode < 2 || iter > 0))
        {
            sP_change = 0;
            iter = weight_mode < 2 ? 1 : 2;
            break;
        }

        tmp = 0;
        isum = 0;
        for (y = 0; y < nrows; y++)
            for (x = delta_x; x < ncols - delta_x; x++)
                if (mask[im_index(x, y)])
                {
                    double resid = model[im_index(x, y)] - im[im_index(x, y)];
                    tmp += resid * resid;
                    isum++;
                }
        dev = sqrt(tmp / isum);

        if (kappa > 0)
            for (y = 0; y < nrows; y++)
                for (x = delta_x; x < ncols - delta_x; x++)
                    mask[im_index(x, y)] =
                        fabs(model[im_index(x, y)] - im[im_index(x, y)]) < kappa * dev ? 1 : 0;

    } while ((iter++ == 0) || ((iter <= maxiter) && (sP_change > sP_stop * sP_med)));

    if (iter > maxiter)
    {
        status = -1;
        success = 0;
    }
    else
        status = 1;
    if (!isfinite(sP_change))
    {
        status = -3;
        success = 0;
    }

    /* Uncertainty estimate, accumulated per detector column of the source bin */
    for (x = 0; x < ncols; x++)
    {
        unc[x] = 0.;
        unc_norm[x] = 0.;
        norm_sq[x] = 0.;
    }
    if (fast_flat)
    {
        const double s = 1.e0 / osample;
        for (x = 0; x < ncols; x++)
        {
            double sw = d1col[x], sw2 = d1col[x] * d1col[x];
            for (m = 1; m < osample; m++)
            {
                sw += s;
                sw2 += s * s;
            }
            sw += s - d1col[x];
            sw2 += (s - d1col[x]) * (s - d1col[x]);
            swcol[x] = sw;
            sw2col[x] = sw2;
        }
        flat_unc(ncols, nrows, im, model, mask, swcol, sw2col, unc, unc_norm, norm_sq);
    }
    else
        for (y = 0; y < nrows; y++)
            for (x = 0; x < ncols; x++)
            {
                if (!mask[im_index(x, y)])
                    continue;
                const zeta_ref_ox *zrow = &zeta[zeta_index(x, y, 0)];
                const int mz = m_zeta[mzeta_index(x, y)];
                tmp = im[im_index(x, y)] - model[im_index(x, y)];
                const double t2 = tmp * tmp;
                for (m = 0; m < mz; m++)
                {
                    xx = zrow[m].x / ox;
                    ww = zrow[m].w;
                    unc[xx] += t2 * ww;
                    unc_norm[xx] += ww;
                    norm_sq[xx] += ww * ww;
                }
            }
    for (x = 0; x < ncols; x++)
    {
        norm = unc_norm[x] - norm_sq[x] / unc_norm[x];
        unc[x] = sqrt(unc[x] / norm * nrows);
    }

    if (chi2 != NULL)
        *chi2 = NAN;
    if (unc_mode > 0 && out_x >= 1 && ox % out_x == 0 && nlag >= 0)
    {
        /* Pixel variances, then M = fill with weight^2 * variance and N rebuilt
           (bandsol overwrote it) from the final sL, weights and mask */
        double *vfac = malloc((size_t)npix * sizeof(double));
        double *Mb = malloc((size_t)nf * nx * sizeof(double));
        double va = 0.e0, vb = 0.e0, c2 = 0.e0;
        long nc2 = 0;
        if (weight_mode == 2)
            variance_fit(model, pix_unc, mask, npix, noise_floor, !fit_b, &va, &vb);
        for (y = 0; y < nrows; y++)
            for (x = 0; x < ncols; x++)
            {
                const long i = im_index(x, y);
                const double u = pix_unc[i];
                double v = weight_mode == 2 ? va * fmax(model[i], 0.e0) + vb
                                            : ((isfinite(u) && u > 0.e0) ? u * u : 0.e0);
                const double w = wpix ? wpix[i] : 1.e0;
                if (!mask[i] || !(v > 0.e0))
                    v = 0.e0;
                else if (x >= delta_x && x < ncols - delta_x)
                {
                    const double res = im[i] - model[i];
                    c2 += res * res / v;
                    nc2++;
                }
                vfac[i] = w * w * v;
            }
        nc2 -= (ncols - 2 * delta_x) + nrows;
        c2 = nc2 > 0 ? c2 / nc2 : NAN;

        sp_fill(ncols, nrows, nf, nx, bx, osample, mzz, fast_flat, im, mask, vfac, sL,
                zeta, m_zeta, z_rng, k0col, d1col, flat_scratch, zw, Mb, p_bj);
        sp_fill(ncols, nrows, nf, nx, bx, osample, mzz, fast_flat, im, mask, wpix, sL,
                zeta, m_zeta, z_rng, k0col, d1col, flat_scratch, zw, p_Aij, p_bj);
        sp_penalties(p_Aij, nf, nx, bx, ox, kx, mux, lambda_sP);
        if (propagate_unc(ncols, ox, out_x, nlag, nf, nx, bx, p_Aij, Mb, unc, cov) == 0)
        {
            if (chi2 != NULL)
                *chi2 = c2;
            if (unc_mode == 2 && isfinite(c2))
            {
                for (x = 0; x < ncols; x++)
                    unc[x] *= sqrt(c2);
                if (cov != NULL)
                    for (size_t i = 0; i < (size_t)ncols * out_x * (nlag + 1); i++)
                        cov[i] *= c2;
            }
            /* No unmasked pixel draws on the column: no measurement, so no
               error either, rather than the 0 the propagation gives */
            for (x = 0; x < ncols; x++)
            {
                int q, empty = 1;
                for (q = 0; q < ox && empty; q++)
                    if (Mb[paij_index(x * ox + q, bx)] != 0.e0)
                        empty = 0;
                if (!empty)
                    continue;
                unc[x] = NAN;
                if (cov != NULL)
                    for (q = 0; q < out_x * (nlag + 1); q++)
                        cov[(size_t)x * out_x * (nlag + 1) + q] = NAN;
            }
        }
        else if (cov != NULL)
            for (size_t i = 0; i < (size_t)ncols * out_x * (nlag + 1); i++)
                cov[i] = NAN;
        free(vfac);
        free(Mb);
    }

    for (x = 0; x < ncols; x++)
        if (!isfinite(sP[x]))
        {
            status = -3;
            success = 0;
            break;
        }
    if (sP_fine != NULL)
        for (j = 0; j < nf; j++)
            sP_fine[j] = sPf[j];

    free(sP_old);
    free(wbuf);
    free(sP_diff);
    free(norm_sq);
    free(unc_norm);
    free(l_Aij);
    free(p_Aij);
    free(p_bj);
    free(l_bj);
    free(sPf);
    free(zw);
    free(zk);
    free(zeta);
    free(m_zeta);
    free(z_rng);
    free(ycen_offset);
    free(k0col);
    free(d1col);
    free(swcol);
    free(sw2col);
    free(flat_scratch);

    info[0] = success;
    info[1] = sP_change;
    info[2] = status;
    info[3] = iter;
    info[4] = margin;
    if (want_ext)
    {
        info[5] = ndrop;
        info[6] = bx;
    }
    if (want_ext2)
    {
        info[7] = penalty_damping(mux, kx);
        info[8] = penalty_resolution(mux, kx);
        if (ky > 0)
        {
            info[9] = penalty_damping(muy, ky);
            info[10] = penalty_resolution(muy, ky);
        }
        else
        {
            info[9] = -mu1_y_legacy;
            info[10] = penalty_resolution(&mu1_y_legacy, mu1_y_legacy > 0.e0);
        }
    }
    return 0;
}
