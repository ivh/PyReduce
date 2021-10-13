#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "slit_func_2d_xi_zeta_bd.h"

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define signum(a) (((a) > 0) ? 1 : ((a) < 0) ? -1 : 0)
#define DEBUG 0

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
#define MAX_XI ((_ncols) * (_ny)*4)
#define MAX_PSF_X (_ncols)
#define MAX_PSF_Y (3)
#define MAX_PSF (MAX_PSF_X * MAX_PSF_Y)
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

static long xi_index(long x, long y, long z)
{
    long i = z + 4 * y + _ny * 4 * x;
    if ((i < 0) | (i >= MAX_XI))
    {
        printf("INDEX OUT OF BOUNDS. Xi[%li, %li, %li]\n", x, y, z);
        return 0;
    }
    return i;
}

static long psf_index(long x, long y)
{
    long i = ((x)*3 + (y));
    if ((i < 0) | (i >= MAX_PSF))
    {
        printf("INDEX OUT OF BOUNDS. PSF[%li, %li]\n", x, y);
        return 0;
    }
    return i;
}

static long a_index(long x, long y)
{
    long i = _n * y + x;
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
    long i = ((y)*_ny) + (x);
    if ((i < 0) | (i >= MAX_LAIJ))
    {
        printf("INDEX OUT OF BOUNDS. l_Aij[%li, %li]\n", x, y);
        return 0;
    }
    return i;
}

static long paij_index(long x, long y)
{
    long i = ((y)*_ncols) + (x);
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
#define xi_index(x, y, z) ((z) + 4 * (y) + _ny * 4 * (x))
#define psf_index(x, y) ((x)*3 + (y))
#define a_index(x, y) ((y)*n + (x))
#define r_index(i) (i)
#define sp_index(i) (i)
#define laij_index(x, y) ((y)*ny) + (x)
#define paij_index(x, y) ((y)*ncols) + (x)
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

// This is the faster median determination method.
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

double median_absolute_deviation(double arr[], unsigned int n)
{
    double median = quick_select_median(arr, n);
    for (size_t i = 0; i < n; i++)
    {
        arr[i] = fabs(arr[i] - median);
    }
    double mad = quick_select_median(arr, n);
    return mad;
}

int xi_zeta_tensors(
    int ncols,
    int nrows,
    int ny,
    double *ycen,
    int *ycen_offset,
    int y_lower_lim,
    int osample,
    double *PSF_curve,
    xi_ref *xi,
    zeta_ref *zeta,
    int *m_zeta)
{
    /*
    Create the Xi and Zeta tensors, that describe the contribution of each pixel to the subpixels of the image,
    Considering the curvature of the slit.

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
    PSF_curve : double array of shape (ncols, 3)
        Parabolic fit to the slit image curvature.
        For column d_x = PSF_curve[ncols][0] +  PSF_curve[ncols][1] *d_y + PSF_curve[ncols][2] *d_y^2,
        where d_y is the offset from the central line ycen.
        Thus central subpixel of omega[x][y'][delta_x][iy'] does not stick out of column x.
    xi : (out) xi_ref array of shape (ncols, ny, 4)
        Convolution tensor telling the coordinates of detector
        pixels on which {x, iy} element falls and the corresponding projections.
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
    int x, xx, y, yy, ix, ix1, ix2, iy, iy1, iy2, m;
    double step, delta, dy, w, d1, d2;

    step = 1.e0 / osample;

    /* Clean xi */
    for (x = 0; x < ncols; x++)
    {
        for (iy = 0; iy < ny; iy++)
        {
            for (m = 0; m < 4; m++)
            {
                xi[xi_index(x, iy, m)].x = -1;
                xi[xi_index(x, iy, m)].y = -1;
                xi[xi_index(x, iy, m)].w = 0.;
            }
        }
    }

    /* Clean zeta */
    for (x = 0; x < ncols; x++)
    {
        for (y = 0; y < nrows; y++)
        {
            m_zeta[mzeta_index(x, y)] = 0;
            for (ix = 0; ix < MAX_ZETA_Z; ix++)
            {
                zeta[zeta_index(x, y, ix)].x = -1;
                zeta[zeta_index(x, y, ix)].iy = -1;
                zeta[zeta_index(x, y, ix)].w = 0.;
            }
        }
    }

    /*
    Construct the xi and zeta tensors. They contain pixel references and contribution.
    values going from a given subpixel to other pixels (xi) and coming from other subpixels
    to a given detector pixel (zeta).
    Note, that xi and zeta are used in the equations for sL, sP and for the model but they
    do not involve the data, only the geometry. Thus it can be pre-computed once.
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

        /*
        The final hurdle for 2D slit decomposition is to construct two 3D reference tensors. We proceed
        similar to 1D case except that now each iy subpixel can be shifted left or right following
        the curvature of the slit image on the detector. We assume for now that each subpixel is
        exactly 1 detector pixel wide. This may not be exactly true if the curvature changes accross
        the focal plane but will deal with it when the necessity will become apparent. For now we
        just assume that a shift delta the weight w assigned to subpixel iy is divided between
        ix1=int(delta) and ix2=int(delta)+signum(delta) as (1-|delta-ix1|)*w and |delta-ix1|*w.

        The curvature is given by a quadratic polynomial evaluated from an approximation for column
        x: delta = PSF_curve[x][0] + PSF_curve[x][1] * (y-yc[x]) + PSF_curve[x][2] * (y-yc[x])^2.
        It looks easy except that y and yc are set in the global detector coordinate system rather than
        in the shifted and cropped swath passed to slit_func_2d. One possible solution I will try here
        is to modify PSF_curve before the call such as:
        delta = PSF_curve'[x][0] + PSF_curve'[x][1] * (y'-ycen[x]) + PSF_curve'[x][2] * (y'-ycen[x])^2
        where y' = y - floor(yc).
        */

        /* Define initial distance from ycen       */
        /* It is given by the center of the first  */
        /* subpixel falling into pixel y_lower_lim */
        dy = ycen[x] - floor((y_lower_lim + ycen[x]) / step) * step - step;

        /*
        Now we go detector pixels x and y incrementing subpixels looking for their controibutions
        to the current and adjacent pixels. Note that the curvature/tilt of the projected slit
        image could be so large that subpixel iy may no contribute to column x at all. On the
        other hand, subpixels around ycen by definition must contribute to pixel x,y.
        3rd index in xi refers corners of pixel xx,y: 0:LL, 1:LR, 2:UL, 3:UR.
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
                delta = (PSF_curve[psf_index(x, 1)] + PSF_curve[psf_index(x, 2)] * (dy - ycen[x])) * (dy - ycen[x]);
                ix1 = delta;
                ix2 = ix1 + signum(delta);

                /* Three cases: subpixel on the bottom boundary of row y, intermediate subpixels and top boundary */

                if (iy == iy1) /* Case A: Subpixel iy is entering detector row y */
                {
                    if (ix1 < ix2) /* Subpixel iy shifts to the right from column x  */
                    {
                        if (x + ix1 >= 0 && x + ix2 < ncols)
                        {
                            xx = x + ix1; /* Upper right corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 3)].x = xx;
                            xi[xi_index(x, iy, 3)].y = yy;
                            xi[xi_index(x, iy, 3)].w = w - fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 3)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 3)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                            xx = x + ix2; /* Upper left corner of subpixel iy */
                            // This offset is required because the iy subpixel
                            // is going to contribute to the yy row in xx column
                            // of detector pixels where yy and y are in the same
                            // row. In the packed array this is not necessarily true.
                            // Instead, what we know is that:
                            // y+ycen_offset[x] == yy+ycen_offset[xx]
                            yy = y + ycen_offset[x] - ycen_offset[xx];

                            xi[xi_index(x, iy, 2)].x = xx;
                            xi[xi_index(x, iy, 2)].y = yy;
                            xi[xi_index(x, iy, 2)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 2)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 2)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else if (ix1 > ix2) /* Subpixel iy shifts to the left from column x */
                    {
                        if (x + ix2 >= 0 && x + ix1 < ncols)
                        {
                            xx = x + ix2; /* Upper left corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 2)].x = xx;
                            xi[xi_index(x, iy, 2)].y = yy;
                            xi[xi_index(x, iy, 2)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 2)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 2)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                            xx = x + ix1; /* Upper right corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 3)].x = xx;
                            xi[xi_index(x, iy, 3)].y = yy;
                            xi[xi_index(x, iy, 3)].w = w - fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 3)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 3)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else
                    {
                        xx = x + ix1; /* Subpixel iy stays inside column x */
                        yy = y + ycen_offset[x] - ycen_offset[xx];
                        xi[xi_index(x, iy, 2)].x = xx;
                        xi[xi_index(x, iy, 2)].y = yy;
                        xi[xi_index(x, iy, 2)].w = w;
                        if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && w > 0)
                        {
                            m = m_zeta[mzeta_index(xx, yy)];
                            zeta[zeta_index(xx, yy, m)].x = x;
                            zeta[zeta_index(xx, yy, m)].iy = iy;
                            zeta[zeta_index(xx, yy, m)].w = w;
                            m_zeta[mzeta_index(xx, yy)]++;
                        }
                    }
                }
                else if (iy == iy2) /* Case C: Subpixel iy is leaving detector row y */
                {
                    if (ix1 < ix2) /* Subpixel iy shifts to the right from column x */
                    {
                        if (x + ix1 >= 0 && x + ix2 < ncols)
                        {
                            xx = x + ix1; /* Bottom right corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 1)].x = xx;
                            xi[xi_index(x, iy, 1)].y = yy;
                            xi[xi_index(x, iy, 1)].w = w - fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 1)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 1)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                            xx = x + ix2; /* Bottom left corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 0)].x = xx;
                            xi[xi_index(x, iy, 0)].y = yy;
                            xi[xi_index(x, iy, 0)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 0)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 0)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else if (ix1 > ix2) /* Subpixel iy shifts to the left from column x */
                    {
                        if (x + ix2 >= 0 && x + ix1 < ncols)
                        {
                            xx = x + ix2; /* Bottom left corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 0)].x = xx;
                            xi[xi_index(x, iy, 0)].y = yy;
                            xi[xi_index(x, iy, 0)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 0)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 0)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                            xx = x + ix1; /* Bottom right corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 1)].x = xx;
                            xi[xi_index(x, iy, 1)].y = yy;
                            xi[xi_index(x, iy, 1)].w = w - fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 1)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 1)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else /* Subpixel iy stays inside column x        */
                    {
                        xx = x + ix1;
                        yy = y + ycen_offset[x] - ycen_offset[xx];
                        xi[xi_index(x, iy, 0)].x = xx;
                        xi[xi_index(x, iy, 0)].y = yy;
                        xi[xi_index(x, iy, 0)].w = w;
                        if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && w > 0)
                        {
                            m = m_zeta[mzeta_index(xx, yy)];
                            zeta[zeta_index(xx, yy, m)].x = x;
                            zeta[zeta_index(xx, yy, m)].iy = iy;
                            zeta[zeta_index(xx, yy, m)].w = w;
                            m_zeta[mzeta_index(xx, yy)]++;
                        }
                    }
                }
                else /* CASE B: Subpixel iy is fully inside detector row y */
                {
                    if (ix1 < ix2) /* Subpixel iy shifts to the right from column x      */
                    {
                        if (x + ix1 >= 0 && x + ix2 < ncols)
                        {
                            xx = x + ix1; /* Bottom right corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 1)].x = xx;
                            xi[xi_index(x, iy, 1)].y = yy;
                            xi[xi_index(x, iy, 1)].w = w - fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 1)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 1)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                            xx = x + ix2; /* Bottom left corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 0)].x = xx;
                            xi[xi_index(x, iy, 0)].y = yy;
                            xi[xi_index(x, iy, 0)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 0)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 0)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else if (ix1 > ix2) /* Subpixel iy shifts to the left from column x */
                    {
                        if (x + ix2 >= 0 && x + ix1 < ncols)
                        {
                            xx = x + ix2; /* Bottom right corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 1)].x = xx;
                            xi[xi_index(x, iy, 1)].y = yy;
                            xi[xi_index(x, iy, 1)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 1)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 1)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                            xx = x + ix1; /* Bottom left corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 0)].x = xx;
                            xi[xi_index(x, iy, 0)].y = yy;
                            xi[xi_index(x, iy, 0)].w = w - fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 0)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 0)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else /* Subpixel iy stays inside column x */
                    {
                        xx = x + ix2;
                        yy = y + ycen_offset[x] - ycen_offset[xx];
                        xi[xi_index(x, iy, 0)].x = xx;
                        xi[xi_index(x, iy, 0)].y = yy;
                        xi[xi_index(x, iy, 0)].w = w;
                        if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows && w > 0)
                        {
                            m = m_zeta[mzeta_index(xx, yy)];
                            zeta[zeta_index(xx, yy, m)].x = x;
                            zeta[zeta_index(xx, yy, m)].iy = iy;
                            zeta[zeta_index(xx, yy, m)].w = w;
                            m_zeta[mzeta_index(xx, yy)]++;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

int slit_func_curved(int ncols,
                     int nrows,
                     int ny,
                     double *im,
                     double *pix_unc,
                     unsigned char *mask,
                     double *ycen,
                     int *ycen_offset,
                     int y_lower_lim,
                     int osample,
                     double lambda_sP,
                     double lambda_sL,
                     int maxiter,
                     double *PSF_curve,
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
    nx : int
        Range of columns affected by PSF tilt: nx = 2 * delta_x + 1
    ny : int
        Size of the slit function array: ny = osample * (nrows + 1) + 1
    im : double array of shape (nrows, ncols)
        Image to be decomposed
    pix_unc : double array of shape (nrows, ncols)
        Individual pixel uncertainties. Set to zero if unknown.
    mask : byte array of shape (nrows, ncols)
        Initial and final mask for the swath, both in and output
    ycen : double array of shape (ncols,)
        Order centre line offset from pixel row boundary.
        Should only contain values between 0 and 1.
    ycen_offset : int array of shape (ncols,)
        Order image column shift
    y_lower_lim : int
        Number of detector pixels below the pixel containing
        the central line yc.
    osample : int
        Subpixel ovsersampling factor
    lambda_sP : double
        Smoothing parameter for the spectrum, could be zero
    lambda_sL : double
        Smoothing parameter for the slit function, usually > 0
    PSF_curve : double array of shape (ncols, 3)
        Slit curvature parameters for each point along the spectrum
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
    int x, xx, xxx, y, yy, iy, jy, n, m, nx;
    double norm, dev, lambda, diag_tot, ww, www;
    double cost_old, ftol, tmp;
    int iter, delta_x;
    unsigned int isum;

    // For the solving of the equation system
    double *l_Aij, *l_bj, *p_Aij, *p_bj;
    double *diff;

    // For the geometry
    xi_ref *xi;
    zeta_ref *zeta;
    int *m_zeta;

    // The Optimization results
    double success, status, cost;

    // maxiter = 20; // Maximum number of iterations
    ftol = 1e-7;  // Maximum cost difference between two iterations to stop convergence
    success = 1;
    status = 0;

    cost = INFINITY;
    ny = osample * (nrows + 1) + 1; /* The size of the sL array. Extra osample is because ycen can be between 0 and 1. */

#if DEBUG
    _ncols = ncols;
    _nrows = nrows;
    _ny = ny;
    _osample = osample;
#endif

    // If we want to smooth the spectrum we need at least delta_x = 1
    // Otherwise delta_x = 0 works if there is no curvature
    delta_x = lambda_sP == 0 ? 0 : 1;
    for (x = 0; x < ncols; x++)
    {
        for (y = -y_lower_lim; y < nrows - y_lower_lim + 1; y++)
        {
            tmp = ceil(fabs(y * PSF_curve[psf_index(x, 1)] + y * y * PSF_curve[psf_index(x, 2)]));
            delta_x = max(delta_x, tmp);
        }
    }
    nx = 4 * delta_x + 1; /* Maximum horizontal shift in detector pixels due to slit image curvature         */

#if DEBUG
    _nx = nx;
#endif

    // The curvature is larger than the number of columns
    // Usually that means that the curvature is messed up
    if (nx > ncols)
    {
        info[0] = 0;    //failed
        info[1] = cost; //INFINITY
        info[2] = -2;   // curvature to large
        info[3] = 0;
        info[4] = delta_x;
        return -1;
    }

    l_Aij = malloc(MAX_LAIJ * sizeof(double));
    p_Aij = malloc(MAX_PAIJ * sizeof(double));
    l_bj = malloc(MAX_LBJ * sizeof(double));
    p_bj = malloc(MAX_PBJ * sizeof(double));
    xi = malloc(MAX_XI * sizeof(xi_ref));
    zeta = malloc(MAX_ZETA * sizeof(zeta_ref));
    m_zeta = malloc(MAX_MZETA * sizeof(int));
    diff = malloc(MAX_IM * sizeof(double));

    xi_zeta_tensors(ncols, nrows, ny, ycen, ycen_offset, y_lower_lim, osample, PSF_curve, xi, zeta, m_zeta);

    /* Loop through sL , sP reconstruction until convergence is reached */
    iter = 0;
    do
    {
        // Save the total cost (chi-square) from the previous iteration
        cost_old = cost;

        /* Compute slit function sL */

        /* Prepare the RHS and the matrix */
        for (iy = 0; iy < MAX_LBJ; iy++)
            l_bj[lbj_index(iy)] = 0.e0; /* Clean RHS */
        for (iy = 0; iy < MAX_LAIJ; iy++)
            l_Aij[iy] = 0;

        /* Fill in SLE arrays for slit function */
        diag_tot = 0.e0;
        for (iy = 0; iy < ny; iy++)
        {
            for (x = 0; x < ncols; x++)
            {
                for (n = 0; n < 4; n++)
                {
                    ww = xi[xi_index(x, iy, n)].w;
                    if (ww > 0)
                    {
                        xx = xi[xi_index(x, iy, n)].x;
                        yy = xi[xi_index(x, iy, n)].y;
                        if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows)
                        {
                            if (m_zeta[mzeta_index(xx, yy)] > 0)
                            {
                                for (m = 0; m < m_zeta[mzeta_index(xx, yy)]; m++)
                                {
                                    xxx = zeta[zeta_index(xx, yy, m)].x;
                                    jy = zeta[zeta_index(xx, yy, m)].iy;
                                    www = zeta[zeta_index(xx, yy, m)].w;
                                    l_Aij[laij_index(iy, jy - iy + 2 * osample)] += sP[sp_index(xxx)] * sP[sp_index(x)] * www * ww * mask[im_index(xx, yy)];
                                }
                                l_bj[lbj_index(iy)] += im[im_index(xx, yy)] * mask[im_index(xx, yy)] * sP[sp_index(x)] * ww;
                            }
                        }
                    }
                }
            }
            diag_tot += l_Aij[laij_index(iy, 2 * osample)];
        }

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

        /* Compute spectrum sP */
        for (x = 0; x < MAX_PBJ; x++)
            p_bj[pbj_index(x)] = 0;
        for (x = 0; x < MAX_PAIJ; x++)
            p_Aij[x] = 0;

        for (x = 0; x < ncols; x++)
        {
            for (iy = 0; iy < ny; iy++)
            {
                for (n = 0; n < 4; n++)
                {
                    ww = xi[xi_index(x, iy, n)].w;
                    if (ww > 0)
                    {
                        xx = xi[xi_index(x, iy, n)].x;
                        yy = xi[xi_index(x, iy, n)].y;
                        if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows)
                        {
                            if (m_zeta[mzeta_index(xx, yy)] > 0)
                            {
                                for (m = 0; m < m_zeta[mzeta_index(xx, yy)]; m++)
                                {
                                    xxx = zeta[zeta_index(xx, yy, m)].x;
                                    jy = zeta[zeta_index(xx, yy, m)].iy;
                                    www = zeta[zeta_index(xx, yy, m)].w;
                                    p_Aij[paij_index(x, xxx - x + 2 * delta_x)] += sL[sl_index(jy)] * sL[sl_index(iy)] * www * ww * mask[im_index(xx, yy)];
                                }
                                p_bj[pbj_index(x)] += im[im_index(xx, yy)] * mask[im_index(xx, yy)] * sL[sl_index(iy)] * ww;
                            }
                        }
                    }
                }
            }
        }

        if (lambda_sP > 0.e0)
        {
            norm = 0.e0;
            for (x = 0; x < ncols; x++)
            {
                norm += sP[sp_index(x)];
            }
            norm /= ncols;
            lambda = lambda_sP * norm; /* Scale regularization parameter */

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

        /* Solve the system of equations */
        bandsol(p_Aij, p_bj, MAX_PAIJ_X, MAX_PAIJ_Y);

        for (x = 0; x < ncols; x++)
            sP[sp_index(x)] = p_bj[pbj_index(x)];

        /* Compute the model */
        for (x = 0; x < MAX_IM; x++)
        {
            model[x] = 0.;
        }

        for (y = 0; y < nrows; y++)
        {
            for (x = 0; x < ncols; x++)
            {
                for (m = 0; m < m_zeta[mzeta_index(x, y)]; m++)
                {
                    xx = zeta[zeta_index(x, y, m)].x;
                    iy = zeta[zeta_index(x, y, m)].iy;
                    ww = zeta[zeta_index(x, y, m)].w;
                    model[im_index(x, y)] += sP[xx] * sL[iy] * ww;
                }
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
        cost = 0;
        isum = 0;
        for (y = 0; y < nrows; y++)
        {
            for (x = delta_x; x < ncols - delta_x; x++)
            {
                if (mask[im_index(x, y)])
                {
                    tmp = model[im_index(x, y)] - im[im_index(x, y)];
                    diff[isum] = tmp;
                    tmp /= max(pix_unc[im_index(x, y)], 1);
                    cost += tmp * tmp;
                    isum++;
                }
            }
        }
        cost /= (isum - (ncols + ny));
        dev = median_absolute_deviation(diff, isum);
        // This is the "conversion" factor betweem MAD and STD
        // i.e. a perfect normal distribution has MAD = sqrt(2/pi) * STD
        dev *= 1.4826;

        /* Adjust the mask marking outlyers */
        for (y = 0; y < nrows; y++)
        {
            for (x = delta_x; x < ncols - delta_x; x++)
            {
                // The MAD is significantly smaller than the STD was, since it describes
                // only the central peak, not the distribution
                // The factor 40 was chosen, since it is roughly equal to 6 * STD
                if (fabs(model[im_index(x, y)] - im[im_index(x, y)]) < 40. * dev)
                    mask[im_index(x, y)] = 1;
                else
                    mask[im_index(x, y)] = 0;
            }
        }

#if DEBUG
        if (cost == 0)
        {
            printf("Iteration: %i, Reduced chi-square: %f\n", iter, cost);
            printf("dev: %f\n", dev);
            printf("isum: %i\n", isum);
            printf("iteration: %i\n", iter);
            printf("-----------\n");
        }
#endif
        /* Check for convergence */
    } while (((iter++ < maxiter) && (cost_old - cost > ftol)) || ((isfinite(cost) == 0) || ((isfinite(cost_old) == 0))));

    if (iter >= maxiter - 1)
    {
        status = -1; // ran out of iterations
        success = 0;
    }
    else if (cost_old - cost <= ftol)
        status = 1; // cost did not improve enough between iterations

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

    free(diff);
    free(l_Aij);
    free(p_Aij);
    free(p_bj);
    free(l_bj);

    free(xi);
    free(zeta);
    free(m_zeta);

    info[0] = success;
    info[1] = cost;
    info[2] = status;
    info[3] = iter;
    info[4] = delta_x;

    return 0;
}
