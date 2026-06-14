#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include "slitdec.h"

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define signum(a) (((a) > 0) ? 1 : ((a) < 0) ? -1 : 0)
#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef REGULARIZE_DIAGONAL
#define REGULARIZE_DIAGONAL 1
#endif

// Store important sizes in global variables to make access easier
// When calculating the proper indices
// When not checking the indices just the variables directly
#if DEBUG
int _ncols = 0;
int _nrows = 0;
int _ny = 0;
int _nx = 0;
int _osample = 0;
int _n = 0;
int _nd = 0;
#else
#define _ncols ncols
#define _nrows nrows
#define _ny ny
#define _nx nx
#define _osample osample
#define _n n
#define _nd nd
#endif

// Define the sizes of each array
#define MAX_ZETA_X (_ncols)
#define MAX_ZETA_Y (_nrows)
#define MAX_ZETA_Z (3 * ((_osample) + 1))
#define MAX_ZETA (MAX_ZETA_X * MAX_ZETA_Y * MAX_ZETA_Z)
#define MAX_MZETA ((_ncols) * (_nrows))
#define MAX_CRV_X (_ncols)
#define MAX_CRV_Y (3)
#define MAX_CRV (MAX_CRV_X * MAX_CRV_Y)
#define MAX_A ((_n) * (_nd))
#define MAX_R (_n)
#define MAX_SP (_ncols)
#define MAX_SL (_ny)
#define MAX_LAIJ_X (_ny)
#define MAX_LAIJ_Y (4 * (_osample) + 1)
#define MAX_LAIJ (MAX_LAIJ_X * MAX_LAIJ_Y)
#define MAX_PAIJ_X (_ncols)
#define MAX_PAIJ_Y (_nx)
#define MAX_PAIJ (MAX_PAIJ_X * MAX_PAIJ_Y)
#define MAX_LBJ (_ny)
#define MAX_PBJ (_ncols)
#define MAX_IM ((_ncols) * (_nrows))

// If we want to check the index use functions to represent the index
// Otherwise a simpler define will do, which should be faster ?
#if DEBUG
static long zeta_index(long x, long y, long z)
{
    long i = z + y * MAX_ZETA_Z + x * MAX_ZETA_Z * _nrows;
    if ((i < 0) | (i >= MAX_ZETA))
    {
        printf("INDEX OUT OF BOUNDS. Zeta[%li, %li, %li]\n", x, y, z);
        return 0;
    }
    return i;
}

static long mzeta_index(long x, long y)
{
    long i = y + x * _nrows;
    if ((i < 0) | (i >= MAX_MZETA))
    {
        printf("INDEX OUT OF BOUNDS. Mzeta[%li, %li]\n", x, y);
        return 0;
    }
    return i;
}

static long curve_index(long x, long y)
{
    long i = ((x)*3 + (y));
    if ((i < 0) | (i >= MAX_CRV))
    {
        printf("INDEX OUT OF BOUNDS. PSF[%li, %li]\n", x, y);
        return 0;
    }
    return i;
}

static long a_index(long x, long y)
{
    long i = _nd * x + y;
    if ((i < 0) | (i >= MAX_A))
    {
        printf("INDEX OUT OF BOUNDS. a[%li, %li]\n", x, y);
        return 0;
    }
    return i;
}

static long r_index(long i)
{
    if ((i < 0) | (i >= MAX_R))
    {
        printf("INDEX OUT OF BOUNDS. r[%li]\n", i);
        return 0;
    }
    return i;
}

static long sp_index(long i)
{
    if ((i < 0) | (i >= MAX_SP))
    {
        printf("INDEX OUT OF BOUNDS. sP[%li]\n", i);
        return 0;
    }
    return i;
}

static long laij_index(long x, long y)
{
    long i = (x)*MAX_LAIJ_Y + (y);
    if ((i < 0) | (i >= MAX_LAIJ))
    {
        printf("INDEX OUT OF BOUNDS. l_Aij[%li, %li]\n", x, y);
        return 0;
    }
    return i;
}

static long paij_index(long x, long y)
{
    long i = (x)*_nx + (y);
    if ((i < 0) | (i >= MAX_PAIJ))
    {
        printf("INDEX OUT OF BOUNDS. p_Aij[%li, %li]\n", x, y);
        return 0;
    }
    return i;
}

static long lbj_index(long i)
{
    if ((i < 0) | (i >= MAX_LBJ))
    {
        printf("INDEX OUT OF BOUNDS. l_bj[%li]\n", i);
        return 0;
    }
    return i;
}

static long pbj_index(long i)
{
    if ((i < 0) | (i >= MAX_PBJ))
    {
        printf("INDEX OUT OF BOUNDS. p_bj[%li]\n", i);
        return 0;
    }
    return i;
}

static long im_index(long x, long y)
{
    long i = ((y)*_ncols) + (x);
    if ((i < 0) | (i >= MAX_IM))
    {
        printf("INDEX OUT OF BOUNDS. im[%li, %li]\n", x, y);
        return 0;
    }
    return i;
}

static long sl_index(long i)
{
    if ((i < 0) | (i >= MAX_SL))
    {
        printf("INDEX OUT OF BOUNDS. sL[%li]\n", i);
        return 0;
    }
    return i;
}
#else
#define zeta_index(x, y, z) ((z) + (y)*MAX_ZETA_Z + (x)*MAX_ZETA_Z * _nrows)
#define mzeta_index(x, y) ((y) + (x)*_nrows)
#define curve_index(x, y) ((x)*6 + (y))
// Band matrices are stored row-major (band entries for one row are
// contiguous): this matches the access pattern of both the SLE fill
// loops and bandsol, unlike the previous column-major layout.
#define a_index(x, y) ((x)*nd + (y))
#define r_index(i) (i)
#define sp_index(i) (i)
#define laij_index(x, y) ((x) * (4 * osample + 1) + (y))
#define paij_index(x, y) ((x)*nx + (y))
#define lbj_index(i) (i)
#define pbj_index(i) (i)
#define im_index(x, y) ((y)*ncols) + (x)
#define sl_index(i) (i)
#endif

int bandsol(double *a, double *r, int n, int nd)
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

double quick_select_median(double arr[], unsigned int n)
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
double quick_select_percentile(double arr[], unsigned int n, double percentile)
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

static inline void zeta_add(zeta_ref *zeta, int *m_zeta, zeta_rng *z_rng,
                     int ncols, int nrows, int osample,
                     int x, int iy, int xx, int yy, double w)
{
    if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && w > 0)
    {
        const int m = m_zeta[mzeta_index(xx, yy)];
        zeta_rng *zr = &z_rng[mzeta_index(xx, yy)];
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

int zeta_tensors(
    int ncols,
    int nrows,
    int ny,
    double *ycen,
    int *ycen_offset,
    int y_lower_lim,
    int osample,
    double *slitcurve,
    double *slitdeltas,
    zeta_ref *zeta,
    int *m_zeta,
    zeta_rng *z_rng)
{
    /*
    Create the zeta tensor, which describes the contribution of each subpixel
    of the oversampled slit function to each detector pixel, considering the
    curvature of the slit.

    Historically this routine also built the inverse mapping ("xi" tensor,
    subpixel -> detector pixels). Since the SLE fill loops in slitdec became
    pixel-centric, only zeta is needed, which also collapses the bookkeeping
    of which xi corner (LL/LR/UL/UR) a contribution belongs to: the zeta
    insertions were identical for all corner cases.

    Parameters
    ----------
    ncols : int
        Swath width in pixels
    nrows : int
        Extraction slit height in pixels
    ny : int
        Size of the slit function array: ny = osample * (nrows + 1) + 1
    ycen : double array of shape (ncols,)
        Order centre line offset from pixel row boundary
    ycen_offsets : int array of shape (ncols,)
        Order image column shift
    y_lower_lim : int
        Number of detector pixels below the pixel containing the central line ycen
    osample : int
        Subpixel ovsersampling factor
    slitcurve : double array of shape (ncols, 6)
        Polynomial fit to the slit image curvature.
        For column d_x = sum_k slitcurve[ncols][k] * d_y^k,
        where d_y is the offset from the central line ycen.
    slitdeltas : double array of shape (ny,)
        Additional per-subpixel horizontal offsets
    zeta : (out) zeta_ref array of shape (ncols, nrows, 3 * (osample + 1))
        Convolution tensor telling the coordinates of subpixels {x, iy} contributing
        to detector pixel {x, y}.
    m_zeta : (out) int array of shape (ncols, nrows)
        The actual number of contributing elements in zeta for each pixel

    Returns
    -------
    code : int
        0 on success, -1 on failure
    */
    int x, xx, y, yy, ix1, ix2, iy, iy1, iy2;
    double step, delta, dy, w, d1, d2;

    step = 1.e0 / osample;

    /* Clean zeta counts. The zeta entries themselves need no initialization:
       only the first m_zeta[x, y] entries of each list are ever read.
       Same for the key ranges: only read where m_zeta[x, y] > 0. */
    for (x = 0; x < ncols; x++)
        for (y = 0; y < nrows; y++)
        {
            zeta_rng *zr = &z_rng[mzeta_index(x, y)];
            m_zeta[mzeta_index(x, y)] = 0;
            zr->min_iy = INT_MAX;
            zr->max_iy = INT_MIN;
            zr->min_x = INT_MAX;
            zr->max_x = INT_MIN;
        }

    /*
    Construct the zeta tensor. It contains pixel references and contribution
    values coming from subpixels to a given detector pixel.
    Note that zeta is used in the equations for sL, sP and for the model but it
    does not involve the data, only the geometry. Thus it can be pre-computed once.
    */
    for (x = 0; x < ncols; x++)
    {
        /*
        I promised to reconsider the initial offset. Here it is. For the original layout
        (no column shifts and discontinuities in ycen) there is pixel y that contains the
        central line yc. There are two options here (by construction of ycen that can be 0
        but cannot be 1): (1) yc is inside pixel y and (2) yc falls at the boundary between
        pixels y and y-1. yc cannot be at the boundary of pixels y+1 and y because we would
        select y+1 to be pixel y in that case.

        Next we need to define starting and ending indices iy for sL subpixels that contribute
        to pixel y. I call them iy1 and iy2. For both cases we assume osample+1 subpixels covering
        pixel y (wierd). So for case 1 iy1 will be (y-1)*osample and iy2 == y*osample. Special
        treatment of the boundary subpixels will compensate for introducing extra subpixel in
        case 1. In case 2 things are more logical: iy1=(yc-y)*osample+(y-1)*osample;
        iy2=(y+1-yc)*osample)+(y-1)*osample. ycen is yc-y making things simpler. Note also that
        the same pattern repeates for all rows: we only need to initialize iy1 and iy2 and keep
        incrementing them by osample.
        */

        iy2 = osample - floor(ycen[x] * osample);
        iy1 = iy2 - osample;

        /*
        Handling partial subpixels cut by detector pixel rows is again tricky. Here we have three
        cases (mostly because of the decision to assume that we always have osample+1 subpixels
        per one detector pixel). Here d1 is the fraction of the subpixel iy1 inside detector pixel y.
        d2 is then the fraction of subpixel iy2 inside detector pixel y. By definition d1+d2==step.
        Case 1: ycen falls on the top boundary of each detector pixel (ycen == 1). Here we conclude
                that the first subpixel is fully contained inside pixel y and d1 is set to step.
        Case 2: ycen falls on the bottom boundary of each detector pixel (ycen == 0). Here we conclude
                that the first subpixel is totally outside of pixel y and d1 is set to 0.
        Case 3: ycen falls inside of each pixel (0>ycen>1). In this case d1 is set to the fraction of
                the first step contained inside of each pixel.
        And BTW, this also means that central line coinsides with the upper boundary of subpixel iy2
        when the y loop reaches pixel y_lower_lim. In other words:

        dy=(iy-(y_lower_lim+ycen[x])*osample)*step-0.5*step
        */

        d1 = fmod(ycen[x], step);
        if (d1 == 0)
            d1 = step;
        d2 = step - d1;

        /* Define initial distance from ycen       */
        /* It is given by the center of the first  */
        /* subpixel falling into pixel y_lower_lim */
        dy = ycen[x] - floor((y_lower_lim + ycen[x]) / step) * step - step;

        /*
        Now we go detector pixels x and y incrementing subpixels looking for their contributions
        to the current and adjacent pixels. Note that the curvature/tilt of the projected slit
        image could be so large that subpixel iy may not contribute to column x at all. On the
        other hand, subpixels around ycen by definition must contribute to pixel x,y.

        Each subpixel is assumed to be exactly 1 detector pixel wide; a horizontal shift delta
        divides its weight w between columns ix1=int(delta) and ix2=ix1+signum(delta) as
        (1-|delta-ix1|)*w and |delta-ix1|*w. The yy offset is required because the iy subpixel
        contributes to the yy row in the xx column of detector pixels where yy and y are in the
        same row. In the packed array this is not necessarily true. Instead, what we know is:
        y+ycen_offset[x] == yy+ycen_offset[xx]
        */

        for (y = 0; y < nrows; y++)
        {
            iy1 += osample; // Bottom subpixel falling in row y
            iy2 += osample; // Top subpixel falling in row y
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
                double t = dy - ycen[x];
                delta = t * (slitcurve[curve_index(x, 1)] +
                        t * (slitcurve[curve_index(x, 2)] +
                        t * (slitcurve[curve_index(x, 3)] +
                        t * (slitcurve[curve_index(x, 4)] +
                        t *  slitcurve[curve_index(x, 5)]))))
                        + slitdeltas[iy];
                ix1 = delta;
                ix2 = ix1 + signum(delta);

                if (ix1 < ix2) /* Subpixel iy shifts to the right from column x */
                {
                    if (x + ix1 >= 0 && x + ix2 < ncols)
                    {
                        xx = x + ix1;
                        yy = y + ycen_offset[x] - ycen_offset[xx];
                        zeta_add(zeta, m_zeta, z_rng, ncols, nrows, osample, x, iy, xx, yy,
                                 w - fabs(delta - ix1) * w);
                        xx = x + ix2;
                        yy = y + ycen_offset[x] - ycen_offset[xx];
                        zeta_add(zeta, m_zeta, z_rng, ncols, nrows, osample, x, iy, xx, yy,
                                 fabs(delta - ix1) * w);
                    }
                }
                else if (ix1 > ix2) /* Subpixel iy shifts to the left from column x */
                {
                    if (x + ix2 >= 0 && x + ix1 < ncols)
                    {
                        xx = x + ix2;
                        yy = y + ycen_offset[x] - ycen_offset[xx];
                        zeta_add(zeta, m_zeta, z_rng, ncols, nrows, osample, x, iy, xx, yy,
                                 fabs(delta - ix1) * w);
                        xx = x + ix1;
                        yy = y + ycen_offset[x] - ycen_offset[xx];
                        zeta_add(zeta, m_zeta, z_rng, ncols, nrows, osample, x, iy, xx, yy,
                                 w - fabs(delta - ix1) * w);
                    }
                }
                else /* Subpixel iy stays inside column x */
                {
                    xx = x + ix1;
                    yy = y + ycen_offset[x] - ycen_offset[xx];
                    zeta_add(zeta, m_zeta, z_rng, ncols, nrows, osample, x, iy, xx, yy, w);
                }
            }
        }
    }
    return 0;
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
                    double *sP,
                    double *sL,
                    double *model,
                    double *unc,
                    double *info)
{
    /*
    Extract the spectrum and slit illumination function for a curved slit

    This function does not assign or free any memory,
    therefore all working arrays are passed as parameters.
    The contents of which will be overriden however

    Parameters
    ----------
    ncols : int
        Swath width in pixels
    nrows : int
        Extraction slit height in pixels
    im : double array of shape (nrows, ncols)
        Image to be decomposed
    pix_unc : double array of shape (nrows, ncols)
        Individual pixel uncertainties. Set to zero if unknown.
    mask : byte array of shape (nrows, ncols)
        Initial and final mask for the swath, both in and output
    ycen : double array of shape (ncols,)
        Order centre line offset from pixel row boundary.
        Should only contain values between 0 and 1.
    slitcurve : double array of shape (ncols, 3)
        Slit curvature parameters for each point along the spectrum
    slitdeltas : double array of shape (nrows, ncols)
        Slit deltas for each point along the slit
    osample : int
        Subpixel ovsersampling factor
    lambda_sP : double
        Smoothing parameter for the spectrum, could be zero
    lambda_sL : double
        Smoothing parameter for the slit function, usually > 0
    sP : (out) double array of shape (ncols,)
        Spectrum resulting from decomposition
    sL : (out) double array of shape (ny,)
        Slit function resulting from decomposition
    model : (out) double array of shape (ncols, nrows)
        Model constructed from sp and sf
    unc : (out) double array of shape (ncols,)
        Spectrum uncertainties based on data - model and pix_unc
    info : (out) double array of shape (5,)
        Returns information about the fit results
    Returns
    -------
    code : int
        0 on success, -1 on failure (see also bandsol)
    */
    int x, xx, y, yy, iy, n, m, nx, ny;
    double norm, dev, lambda, diag_tot, ww, tmp;
    double sP_change, sP_stop, sP_med;
    int iter, delta_x;
    unsigned int isum;
    int *ycen_offset;
    int y_lower_lim = nrows / 2;

    // For the solving of the equation system
    double *l_Aij, *l_bj, *p_Aij, *p_bj;
    double *sP_old, *sP_diff;
    // Scratch buffers for per-pixel merged zeta weights (mz <= 3 * (osample + 1))
    double *zw;
    int *zk;

    // For the geometry
    zeta_ref *zeta;
    int *m_zeta;
    zeta_rng *z_rng;

    // The Optimization results
    double success, status;

    // maxiter = 20; // Maximum number of iterations
    sP_stop = 5e-5;  // Convergence threshold: 99th percentile spectrum change relative to median
    success = 1;
    status = 0;

    sP_change = INFINITY;
    ny = osample * (nrows + 1) + 1; /* The size of the sL array. Extra osample is because ycen can be between 0 and 1. */

#if DEBUG
    _ncols = ncols;
    _nrows = nrows;
    _ny = ny;
    _osample = osample;
    printf("ncols: %d, nrows: %d, ny: %d, osample: %d\n", _ncols, _nrows, _ny, _osample);
#endif

    // If we want to smooth the spectrum we need at least delta_x = 1
    // Otherwise delta_x = 0 works if there is no curvature
    delta_x = lambda_sP == 0 ? 0 : 1;
    for (x = 0; x < ncols; x++)
    {
        for (y = -y_lower_lim; y < nrows - y_lower_lim + 1; y++)
        {
            double y2 = y * y;
            double y3 = y2 * y;
            double y4 = y3 * y;
            double y5 = y4 * y;
            tmp = ceil(fabs(y * slitcurve[curve_index(x, 1)] +
                           y2 * slitcurve[curve_index(x, 2)] +
                           y3 * slitcurve[curve_index(x, 3)] +
                           y4 * slitcurve[curve_index(x, 4)] +
                           y5 * slitcurve[curve_index(x, 5)]));
            delta_x = max(delta_x, tmp);
        }
    }

    // Account for additional shift from slitdeltas
    for (int iy = 0; iy < ny; iy++)
    {
        tmp = ceil(fabs(slitdeltas[iy]));
        delta_x = max(delta_x, tmp);
    }

    nx = 4 * delta_x + 1; /* Maximum horizontal shift in detector pixels due to slit image curvature         */

#if DEBUG
    _nx = nx;
#endif

    // The curvature is larger than the number of columns
    // Usually that means that the curvature is messed up
    if (nx > ncols)
    {
        info[0] = 0;        //failed
        info[1] = sP_change; //INFINITY
        info[2] = -2;       // curvature to large
        info[3] = 0;
        info[4] = delta_x;
        return -1;
    }

    l_Aij = malloc(MAX_LAIJ * sizeof(double));
    p_Aij = malloc(MAX_PAIJ * sizeof(double));
    l_bj = malloc(MAX_LBJ * sizeof(double));
    p_bj = malloc(MAX_PBJ * sizeof(double));
    /* Scratch buffers for per-pixel merged zeta weights: large enough for
       both the slit-function window (2*osample+1 <= MAX_ZETA_Z) and the
       spectrum window (2*delta_x+1 <= nx) */
    int zbuf = max(MAX_ZETA_Z, nx);
    zw = malloc(zbuf * sizeof(double));
    zk = malloc(zbuf * sizeof(int));
    zeta = malloc(MAX_ZETA * sizeof(zeta_ref));
    m_zeta = malloc(MAX_MZETA * sizeof(int));
    z_rng = malloc(MAX_MZETA * sizeof(zeta_rng));
    ycen_offset = malloc(ncols * sizeof(int));
    sP_old = malloc(ncols * sizeof(double));
    sP_diff = malloc(ncols * sizeof(double));

        // remove integer values from ycen, put into ycen_offset
    for (x = 0; x < ncols; x++)
    {
        ycen_offset[x] = ycen[x];
        ycen[x] = ycen[x] - ycen_offset[x];
    }

    zeta_tensors(ncols, nrows, ny, ycen, ycen_offset, y_lower_lim, osample, slitcurve, slitdeltas, zeta, m_zeta, z_rng);

    /* Preset slit function: the caller supplies sL and we skip solving for
       it. Normalize the preset once here (to sum osample) so callers do not
       have to; inside the loop sL is then left untouched. */
    if (use_preset)
    {
        norm = 0.e0;
        for (iy = 0; iy < ny; iy++)
            norm += sL[sl_index(iy)];
        norm /= osample;
        for (iy = 0; iy < ny; iy++)
            sL[sl_index(iy)] /= norm;
    }

    /* Loop through sL , sP reconstruction until convergence is reached */
    iter = 0;
    do
    {
        /* Compute slit function sL (skipped when a preset sL is supplied) */
        if (!use_preset)
        {

        /* Prepare the RHS and the matrix */
        for (iy = 0; iy < MAX_LBJ; iy++)
            l_bj[lbj_index(iy)] = 0.e0; /* Clean RHS */
        for (iy = 0; iy < MAX_LAIJ; iy++)
            l_Aij[iy] = 0;

        /* Fill in SLE arrays for slit function.
           Both SLE matrices are sums over detector pixels of all pairs of
           subpixels contributing to that pixel, i.e. pairs of entries in the
           pixel's zeta list. Iterating pixel-centrically reads zeta
           sequentially and skips masked pixels entirely. Accumulation order
           differs from the historic xi-centric loop only at the rounding
           level. */
        for (xx = 0; xx < ncols; xx++)
        {
            for (yy = 0; yy < nrows; yy++)
            {
                const int mz = m_zeta[mzeta_index(xx, yy)];
                if (mz <= 0 || !mask[im_index(xx, yy)])
                    continue;
                const zeta_ref *zrow = &zeta[zeta_index(xx, yy, 0)];
                const double imv = im[im_index(xx, yy)];
                /* Merge entries sharing the same subpixel index iy: only the
                   summed weight enters both the matrix and the RHS. The iy of
                   one pixel span at most 2*osample+1 indices (the band width
                   assumed by the matrix), so merge into a dense window
                   zw[iy - k0] instead of searching a list of unique keys. */
                const zeta_rng *zr = &z_rng[mzeta_index(xx, yy)];
                const int k0 = zr->min_iy;
                const int rng = zr->max_iy - k0;
                if (rng <= 2 * osample)
                {
                    for (n = 0; n <= rng; n++)
                        zw[n] = 0.e0;
                    for (m = 0; m < mz; m++)
                        zw[zrow[m].iy - k0] += sP[sp_index(zrow[m].x)] * zrow[m].w;
                    /* The matrix is symmetric: accumulate each unordered pair
                       once into the upper bands (mirrored below after the
                       fill). Walking the window row-wise makes the inner loop
                       contiguous in both operands; window entries between
                       actual keys are zero and add exactly nothing. */
                    for (m = 0; m <= rng; m++)
                    {
                        const double um = zw[m];
                        const double *restrict uv = zw + m;
                        double *restrict arow = &l_Aij[laij_index(k0 + m, 2 * osample)];
                        const int dmax = rng - m;
                        for (n = 0; n <= dmax; n++)
                            arow[n] += um * uv[n];
                        l_bj[lbj_index(k0 + m)] += imv * um;
                    }
                    continue;
                }
                /* Over-wide list (extreme geometry): merge by searching
                   unique keys, as before */
                int nk = 0;
                for (m = 0; m < mz; m++)
                {
                    const int key = zrow[m].iy;
                    const double v = sP[sp_index(zrow[m].x)] * zrow[m].w;
                    for (n = 0; n < nk; n++)
                    {
                        if (zk[n] == key)
                        {
                            zw[n] += v;
                            break;
                        }
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
                    l_Aij[laij_index(iy, 2 * osample)] += um * um;
                    for (n = m + 1; n < nk; n++)
                    {
                        const int iyn = zk[n];
                        const int lo = min(iy, iyn);
                        const int d = abs(iyn - iy);
                        l_Aij[laij_index(lo, d + 2 * osample)] += zw[n] * um;
                    }
                    l_bj[lbj_index(iy)] += imv * um;
                }
            }
        }

        /* Mirror the upper bands into the lower bands: A[r+d, 2o-d] = A[r, 2o+d] */
        for (m = 1; m <= 2 * osample; m++)
            for (iy = 0; iy < ny - m; iy++)
                l_Aij[laij_index(iy + m, 2 * osample - m)] = l_Aij[laij_index(iy, 2 * osample + m)];

        diag_tot = 0.e0;
        for (iy = 0; iy < ny; iy++)
            diag_tot += l_Aij[laij_index(iy, 2 * osample)];

        /* Scale regularization parameters */
        lambda = lambda_sL * diag_tot / ny;

        /* Add regularization parts for the SLE matrix */

        l_Aij[laij_index(0, 2 * osample)] += lambda;     /* Main diagonal  */
        l_Aij[laij_index(0, 2 * osample + 1)] -= lambda; /* Upper diagonal */
        for (iy = 1; iy < ny - 1; iy++)
        {
            l_Aij[laij_index(iy, 2 * osample - 1)] -= lambda;    /* Lower diagonal */
            l_Aij[laij_index(iy, 2 * osample)] += lambda * 2.e0; /* Main diagonal  */
            l_Aij[laij_index(iy, 2 * osample + 1)] -= lambda;    /* Upper diagonal */
        }
        l_Aij[laij_index(ny - 1, 2 * osample - 1)] -= lambda; /* Lower diagonal */
        l_Aij[laij_index(ny - 1, 2 * osample)] += lambda;     /* Main diagonal  */

#if REGULARIZE_DIAGONAL
        /* Regularize diagonal to prevent singular matrix from fully masked rows */
        {
            double max_diag = 0.0;
            for (iy = 0; iy < ny; iy++)
            {
                if (l_Aij[laij_index(iy, 2 * osample)] > max_diag)
                    max_diag = l_Aij[laij_index(iy, 2 * osample)];
            }
            if (max_diag > 0.0)
            {
                double min_diag = max_diag * 1.0e-10;
                for (iy = 0; iy < ny; iy++)
                {
                    if (l_Aij[laij_index(iy, 2 * osample)] < min_diag)
                        l_Aij[laij_index(iy, 2 * osample)] = min_diag;
                }
            }
        }
#endif

        /* Solve the system of equations */
        bandsol(l_Aij, l_bj, MAX_LAIJ_X, MAX_LAIJ_Y);

        /* Normalize the slit function */

        norm = 0.e0;
        for (iy = 0; iy < ny; iy++)
        {
            sL[sl_index(iy)] = l_bj[lbj_index(iy)];
            norm += sL[sl_index(iy)];
        }
        norm /= osample;
        for (iy = 0; iy < ny; iy++)
            sL[sl_index(iy)] /= norm;

        } /* end if (!use_preset) */

        /* Compute spectrum sP */
        for (x = 0; x < MAX_PBJ; x++)
            p_bj[pbj_index(x)] = 0;
        for (x = 0; x < MAX_PAIJ; x++)
            p_Aij[x] = 0;

        /* Pixel-centric fill, see comment at the slit function SLE above */
        for (xx = 0; xx < ncols; xx++)
        {
            for (yy = 0; yy < nrows; yy++)
            {
                const int mz = m_zeta[mzeta_index(xx, yy)];
                if (mz <= 0 || !mask[im_index(xx, yy)])
                    continue;
                const zeta_ref *zrow = &zeta[zeta_index(xx, yy, 0)];
                const double imv = im[im_index(xx, yy)];
                /* Merge entries sharing the same source column x; with small
                   curvature this collapses the list to just a few entries.
                   Sources span at most 2*delta_x+1 columns (the band width of
                   the matrix), so merge into a dense window zw[x - k0]. */
                const zeta_rng *zr = &z_rng[mzeta_index(xx, yy)];
                const int k0 = zr->min_x;
                const int rng = zr->max_x - k0;
                if (rng <= 2 * delta_x)
                {
                    for (n = 0; n <= rng; n++)
                        zw[n] = 0.e0;
                    for (m = 0; m < mz; m++)
                        zw[zrow[m].x - k0] += sL[sl_index(zrow[m].iy)] * zrow[m].w;
                    /* Symmetric matrix: upper bands only, mirrored after the
                       fill. Window entries between keys are zero. */
                    for (m = 0; m <= rng; m++)
                    {
                        const double um = zw[m];
                        const double *restrict uv = zw + m;
                        double *restrict arow = &p_Aij[paij_index(k0 + m, 2 * delta_x)];
                        const int dmax = rng - m;
                        for (n = 0; n <= dmax; n++)
                            arow[n] += um * uv[n];
                        p_bj[pbj_index(k0 + m)] += imv * um;
                    }
                    continue;
                }
                /* Over-wide list: merge by searching unique keys */
                int nk = 0;
                for (m = 0; m < mz; m++)
                {
                    const int key = zrow[m].x;
                    const double v = sL[sl_index(zrow[m].iy)] * zrow[m].w;
                    for (n = 0; n < nk; n++)
                    {
                        if (zk[n] == key)
                        {
                            zw[n] += v;
                            break;
                        }
                    }
                    if (n == nk)
                    {
                        zk[nk] = key;
                        zw[nk++] = v;
                    }
                }
                for (m = 0; m < nk; m++)
                {
                    x = zk[m];
                    const double um = zw[m];
                    p_Aij[paij_index(x, 2 * delta_x)] += um * um;
                    for (n = m + 1; n < nk; n++)
                    {
                        const int xn = zk[n];
                        const int lo = min(x, xn);
                        const int d = abs(xn - x);
                        p_Aij[paij_index(lo, d + 2 * delta_x)] += zw[n] * um;
                    }
                    p_bj[pbj_index(x)] += imv * um;
                }
            }
        }

        /* Mirror the upper bands into the lower bands */
        for (m = 1; m <= 2 * delta_x; m++)
            for (x = 0; x < ncols - m; x++)
                p_Aij[paij_index(x + m, 2 * delta_x - m)] = p_Aij[paij_index(x, 2 * delta_x + m)];

        if (lambda_sP > 0.e0)
        {
            lambda = lambda_sP;

            p_Aij[paij_index(0, 2 * delta_x)] += lambda;     /* Main diagonal  */
            p_Aij[paij_index(0, 2 * delta_x + 1)] -= lambda; /* Upper diagonal */
            for (x = 1; x < ncols - 1; x++)
            {
                p_Aij[paij_index(x, 2 * delta_x - 1)] -= lambda;    /* Lower diagonal */
                p_Aij[paij_index(x, 2 * delta_x)] += lambda * 2.e0; /* Main diagonal  */
                p_Aij[paij_index(x, 2 * delta_x + 1)] -= lambda;    /* Upper diagonal */
            }
            p_Aij[paij_index(ncols - 1, 2 * delta_x - 1)] -= lambda; /* Lower diagonal */
            p_Aij[paij_index(ncols - 1, 2 * delta_x)] += lambda;     /* Main diagonal  */
        }

#if REGULARIZE_DIAGONAL
        /* Regularize diagonal to prevent singular matrix from fully masked columns.
           When a column has no valid data (all pixels masked), the corresponding
           row of the matrix is zero, causing division by zero in bandsol.
           We add a small regularization to the diagonal to make it non-singular.
           The resulting spectrum value for masked columns will be ~0 (from p_bj[x]/diag). */
        {
            double max_diag = 0.0;
            for (x = 0; x < ncols; x++)
            {
                if (p_Aij[paij_index(x, 2 * delta_x)] > max_diag)
                    max_diag = p_Aij[paij_index(x, 2 * delta_x)];
            }
            if (max_diag > 0.0)
            {
                double min_diag = max_diag * 1.0e-10;
                for (x = 0; x < ncols; x++)
                {
                    if (p_Aij[paij_index(x, 2 * delta_x)] < min_diag)
                        p_Aij[paij_index(x, 2 * delta_x)] = min_diag;
                }
            }
        }
#endif

        /* Solve the system of equations */
        bandsol(p_Aij, p_bj, MAX_PAIJ_X, MAX_PAIJ_Y);

        /* Save old spectrum, update, and compute change */
        for (x = 0; x < ncols; x++)
            sP_old[x] = sP[sp_index(x)];
        for (x = 0; x < ncols; x++)
            sP[sp_index(x)] = p_bj[pbj_index(x)];
        for (x = 0; x < ncols; x++)
            sP_diff[x] = fabs(sP[sp_index(x)] - sP_old[x]);

        /* Convergence: 99th percentile of change relative to median spectrum */
        sP_change = quick_select_percentile(sP_diff, ncols, 99.0);
        for (x = 0; x < ncols; x++)
            sP_diff[x] = sP[sp_index(x)];  /* reuse buffer for median calc */
        sP_med = fabs(quick_select_median(sP_diff, ncols));

        /* Compute the model.
           x is the outer loop so that the zeta tensor, by far the largest
           array, is read sequentially instead of with a large stride */
        for (x = 0; x < ncols; x++)
        {
            for (y = 0; y < nrows; y++)
            {
                const zeta_ref *zrow = &zeta[zeta_index(x, y, 0)];
                const int mz = m_zeta[mzeta_index(x, y)];
                double acc = 0.;
                for (m = 0; m < mz; m++)
                {
                    xx = zrow[m].x;
                    iy = zrow[m].iy;
                    ww = zrow[m].w;
                    acc += sP[xx] * sL[iy] * ww;
                }
                model[im_index(x, y)] = acc;
            }
        }

        /* Compare model and data */
        // We use the Median absolute derivation to estimate the distribution
        // The MAD is more robust than the usual STD as it uses the median
        // However the MAD << STD, since we are not dealing with a Gaussian
        // at all, but a distribution with heavy wings.
        // Therefore we use the factor 40, instead of 6 to estimate a reasonable range
        // of values. The cutoff is roughly the same.
        // Technically the distribution might best be described by a Voigt profile
        // which we then would have to fit to the distrubtion and then determine,
        // the range that covers 99% of the data.
        // Since that is much more complicated we just use the MAD.
        /* Compute sigma for outlier rejection (RMS of residuals) */
        tmp = 0;
        isum = 0;
        for (y = 0; y < nrows; y++)
        {
            for (x = delta_x; x < ncols - delta_x; x++)
            {
                if (mask[im_index(x, y)])
                {
                    double resid = model[im_index(x, y)] - im[im_index(x, y)];
                    tmp += resid * resid;
                    isum++;
                }
            }
        }
        dev = sqrt(tmp / isum);

        /* Adjust the mask marking outliers */
        if (kappa > 0)
        {
            for (y = 0; y < nrows; y++)
            {
                for (x = delta_x; x < ncols - delta_x; x++)
                {
                    if (fabs(model[im_index(x, y)] - im[im_index(x, y)]) < kappa * dev)
                        mask[im_index(x, y)] = 1;
                    else
                        mask[im_index(x, y)] = 0;
                }
            }
        }

#if DEBUG
        printf("Iteration: %i, sP_change: %g, sP_med: %g, threshold: %g\n",
               iter, sP_change, sP_med, sP_stop * sP_med);
#endif
        /* Check for convergence: stop when the 99th-percentile spectrum change
           drops below sP_stop * median(sP). Always do at least 2 iterations;
           maxiter is an unconditional upper bound. */
    } while ((iter++ == 0) || ((iter <= maxiter) && (sP_change > sP_stop * sP_med)));

    if (iter >= maxiter)
    {
        status = -1; // ran out of iterations
        success = 0;
    }
    else
        status = 1; // converged

    /* Uncertainty estimate */

    for (x = 0; x < ncols; x++)
    {
        unc[sp_index(x)] = 0.;
        p_bj[pbj_index(x)] = 0.;
        p_Aij[paij_index(x, 0)] = 0;
    }

    for (y = 0; y < nrows; y++)
    {
        for (x = 0; x < ncols; x++)
        {
            for (m = 0; m < m_zeta[mzeta_index(x, y)]; m++) // Loop through all pixels contributing to x,y
            {
                if (mask[im_index(x, y)])
                {
                    // Should pix_unc contribute here?
                    xx = zeta[zeta_index(x, y, m)].x;
                    iy = zeta[zeta_index(x, y, m)].iy;
                    ww = zeta[zeta_index(x, y, m)].w;
                    tmp = im[im_index(x, y)] - model[im_index(x, y)];
                    unc[sp_index(xx)] += tmp * tmp * ww;
                    p_bj[pbj_index(xx)] += ww;           // Norm
                    p_Aij[paij_index(xx, 0)] += ww * ww; // Norm squared
                }
            }
        }
    }

    for (x = 0; x < ncols; x++)
    {
        norm = p_bj[pbj_index(x)] - p_Aij[paij_index(x, 0)] / p_bj[pbj_index(x)];
        unc[sp_index(x)] = sqrt(unc[sp_index(x)] / norm * nrows);
    }

    for (x = 0; x < delta_x; x++)
    {
        sP[sp_index(x)] = unc[sp_index(x)] = 0;
    }
    for (x = ncols - delta_x; x < ncols; x++)
    {
        sP[sp_index(x)] = unc[sp_index(x)] = 0;
    }

    free(sP_old);
    free(sP_diff);
    free(l_Aij);
    free(p_Aij);
    free(p_bj);
    free(l_bj);
    free(zw);
    free(zk);

    free(zeta);
    free(m_zeta);
    free(z_rng);

    info[0] = success;
    info[1] = sP_change;
    info[2] = status;
    info[3] = iter;
    info[4] = delta_x;

    return 0;
}
