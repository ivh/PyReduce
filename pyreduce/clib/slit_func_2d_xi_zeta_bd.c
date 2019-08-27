#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "slit_func_2d_xi_zeta_bd.h"

#define CHECK_INDEX 1

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define signum(a) (((a) > 0) ? 1 : ((a) < 0) ? -1 : 0)

// Store important sizes in global variables to make access easier
// When calculating the proper indices
// When not checking the indices just the variables directly
#if CHECK_INDEX
int _ncols = 0;
int _nrows = 0;
int _ny = 0;
int _osample = 0;
int _n = 0;
int _nd = 0;
#else
#define _ncols ncols
#define _nrows nrows
#define _ny ny
#define _osample osample
#define _n n
#define _nd nd
#endif

// Define the sizes of each array
#define MAX_ZETA_Z (4 * ((_osample) + 1))
#define MAX_ZETA ((_ncols) * (_nrows) * MAX_ZETA_Z)
#define MAX_MZETA ((_ncols) * (_nrows))
#define MAX_XI ((_ncols) * (_ny) * 4)
#define MAX_PSF ((_ncols) * 3)
#define MAX_A ((_n) * (_nd))
#define MAX_R (_n)
#define MAX_SP (_ncols)
#define MAX_SL (_ny)
#define MAX_LAIJ_X (_ny)
#define MAX_LAIJ_Y (4 * (_osample) + 1)
#define MAX_LAIJ (MAX_LAIJ_X * MAX_LAIJ_Y)
#define MAX_PAIJ_X (_ncols)
#define MAX_PAIJ_Y (5)
#define MAX_PAIJ (MAX_PAIJ_X * MAX_PAIJ_Y)
#define MAX_LBJ (_ny)
#define MAX_PBJ (_ncols)
#define MAX_IM ((_ncols) * (_nrows))

// If we want to check the index use functions to represent the index
// Otherwise a simpler define will do, which should be faster ?
#if CHECK_INDEX
static int zeta_index(int x, int y, int z)
{
    int i = ((z)*_ncols * _nrows) + ((y)*_ncols) + (x);
    if ((i < 0) | (i >= MAX_ZETA))
    {
        printf("INDEX OUT OF BOUNDS. Zeta[%i, %i, %i]\n", x, y, z);
        return 0;
    }
    return i;
}

static int mzeta_index(int x, int y)
{
    int i = ((y)*_ncols) + (x);
    if ((i < 0) | (i >= MAX_MZETA))
    {
        printf("INDEX OUT OF BOUNDS. Mzeta[%i, %i]\n", x, y);
        return 0;
    }
    return i;
}

static int xi_index(int x, int y, int z)
{
    int i = ((z)*_ncols * _ny) + ((y)*_ncols) + (x);
    if ((i < 0) | (i >= MAX_XI))
    {
        printf("INDEX OUT OF BOUNDS. Xi[%i, %i, %i]\n", x, y, z);
        return 0;
    }
    return i;
}

static int psf_index(int x, int y)
{
    int i = ((y)*_ncols) + (x);
    if ((i < 0) | (i >= MAX_PSF))
    {
        printf("INDEX OUT OF BOUNDS. PSF[%i, %i]\n", x, y);
        return 0;
    }
    return i;
}

static int a_index(int x, int y)
{
    int i = x + _n * y;
    if ((i < 0) | (i >= MAX_A))
    {
        printf("INDEX OUT OF BOUNDS. a[%i, %i]\n", x, y);
        return 0;
    }
    return i;
}

static int r_index(int i)
{
    if ((i < 0) | (i >= MAX_R))
    {
        printf("INDEX OUT OF BOUNDS. r[%i]\n", i);
        return 0;
    }
    return i;
}

static int sp_index(int i)
{
    if ((i < 0) | (i >= MAX_SP))
    {
        printf("INDEX OUT OF BOUNDS. sP[%i]\n", i);
        return 0;
    }
    return i;
}

static int laij_index(int x, int y)
{
    int i = ((y)*MAX_LAIJ_X) + (x);
    if ((i < 0) | (i >= MAX_LAIJ))
    {
        printf("INDEX OUT OF BOUNDS. l_Aij[%i, %i]\n", x, y);
        return 0;
    }
    return i;
}

static int paij_index(int x, int y)
{
    int i = ((y)*_ncols) + (x);
    if ((i < 0) | (i >= MAX_PAIJ))
    {
        printf("INDEX OUT OF BOUNDS. p_Aij[%i, %i]\n", x, y);
        return 0;
    }
    return i;
}

static int lbj_index(int i)
{
    if ((i < 0) | (i >= MAX_LBJ))
    {
        printf("INDEX OUT OF BOUNDS. l_bj[%i]\n", i);
        return 0;
    }
    return i;
}

static int pbj_index(int i)
{
    if ((i < 0) | (i >= MAX_PBJ))
    {
        printf("INDEX OUT OF BOUNDS. p_bj[%i]\n", i);
        return 0;
    }
    return i;
}

static int im_index(int x, int y)
{
    int i = ((y)*_ncols) + (x);
    if ((i < 0) | (i >= MAX_IM))
    {
        printf("INDEX OUT OF BOUNDS. im[%i, %i]\n", x, y);
        return 0;
    }
    return i;
}

static int sl_index(int i)
{
    if ((i < 0) | (i >= MAX_SL))
    {
        printf("INDEX OUT OF BOUNDS. sL[%i]\n", i);
        return 0;
    }
    return i;
}
#else
#define zeta_index(x, y, z) ((z)*ncols * nrows) + ((y)*ncols) + (x)
#define mzeta_index(x, y) ((y)*ncols) + (x)
#define xi_index(x, y, z) ((z)*ncols * ny) + ((y)*ncols) + (x)
#define psf_index(x, y) ((y)*ncols) + (x)
#define a_index(x, y) ((x) + n * (y))
#define r_index(i) (i)
#define sp_index(i) (i)
#define laij_index(x, y) ((y)*ny) + (x)
#define paij_index(x, y) ((y)*ncols) + (x)
#define lbj_index(i) (i)
#define pbj_index(i) (i)
#define im_index(x, y) ((y)*ncols) + (x)
#define sl_index(i) (i)
#endif

/*----------------------------------------------------------------------------*/
/**
  @brief    Solve a sparse system of linear equations
  @param    a   2D array [n,nd]i
  @param    r   array of RHS of size n
  @param    n   number of equations
  @param    nd  width of the band (3 for tri-diagonal system)
  @return   0 on success, -1 on incorrect size of "a" and -4 on
            degenerate matrix

  Solve a sparse system of linear equations with band-diagonal matrix.
  Band is assumed to be symmetrix relative to the main diaginal.

  nd must be an odd number. The main diagonal should be in a(*,nd/2)
  The first lower subdiagonal should be in a(1:n-1,nd/2-1), the first
  upper subdiagonal is in a(0:n-2,nd/2+1) etc. For example:
                    / 0 0 X X X \
                    | 0 X X X X |
                    | X X X X X |
                    | X X X X X |
              A =   | X X X X X |
                    | X X X X X |
                    | X X X X X |
                    | X X X X 0 |
                    \ X X X 0 0 /
 */
/*----------------------------------------------------------------------------*/
static int bandsol(
    double *a,
    double *r,
    int n,
    int nd)
{
    double aa;
    int i, j, k;

    _n = n;
    _nd = nd;

    //if(fmod(nd,2)==0) return -1;

    /* Forward sweep */
    for (i = 0; i < n - 1; i++)
    {
        aa = a[a_index(i, nd / 2)];
        //if(aa==0.e0) return -3;
        r[r_index(i)] /= aa;
        for (j = 0; j < nd; j++)
            a[a_index(i, j)] /= aa;
        for (j = 1; j < min(nd / 2 + 1, n - i); j++)
        {
            aa = a[a_index(i + j, nd / 2 - j)];
            //if(aa==0.e0) return -j;
            r[r_index(i + j)] -= r[r_index(i)] * aa;
            for (k = 0; k < n * (nd - j); k += n)
                a[a_index(i + j + k, 0)] -= a[a_index(i + k, j)] * aa;
        }
    }

    /* Backward sweep */
    r[r_index(n - 1)] /= a[a_index(n - 1, nd / 2)];
    for (i = n - 1; i > 0; i--)
    {
        for (j = 1; j <= min(nd / 2, i); j++)
            r[r_index(i - j)] -= r[r_index(i)] * a[a_index(i - j, nd / 2 + j)];
        //if(a[i-1+n*(nd/2)]==0.e0) return -5;
        r[r_index(i - 1)] /= a[a_index(i - 1, nd / 2)];
    }

    //if(a[n*(nd/2)]==0.e0) return -6;
    r[r_index(0)] /= a[a_index(0, nd / 2)];
    return 0;
}

static int xi_zeta_tensors(
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
    int x, xx, y, yy, ix, ix1, ix2, iy, iy1, iy2;
    double step, delta, dy, w, d1, d2;

    /* Initialise */
    step = 1.e0 / osample;

    /* Clean xi   */
    for (x = 0; x < ncols; x++)
    {
        for (iy = 0; iy < ny; iy++)
        {
            xi[xi_index(x, iy, 0)].x = xi[xi_index(x, iy, 1)].x = xi[xi_index(x, iy, 2)].x = xi[xi_index(x, iy, 3)].x = 0;
            xi[xi_index(x, iy, 0)].y = xi[xi_index(x, iy, 1)].y = xi[xi_index(x, iy, 2)].y = xi[xi_index(x, iy, 3)].y = 0;
            xi[xi_index(x, iy, 0)].w = xi[xi_index(x, iy, 1)].w = xi[xi_index(x, iy, 2)].w = xi[xi_index(x, iy, 3)].w = 0.;
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
       Construct the xi and zeta tensors. They contain pixel references and
       contribution.
       values going from a given subpixel to other pixels (xi) and coming from
       other subpixels to a given detector pixel (zeta).
       Note, that xi and zeta are used in the equations for sL, sP and for
       the model but they do not involve the data, only the geometry.
       Thus it can be pre-computed once.
     */
    for (x = 0; x < ncols; x++)
    {
        /*
           I promised to reconsider the initial offset. Here it is. For the
           original layout (no column shifts and discontinuities in ycen)
           there is pixel y that contains the central line yc. There are two
           options here (by construction of ycen that can be 0 but cannot be
           1): (1) yc is inside pixel y and (2) yc falls at the boundary
           between pixels y and y-1. yc cannot be at the foundary of pixels
           y+1 and y because we would select y+1 to be pixel y in that case.

           Next we need to define starting and ending indices iy for sL
           subpixels that contribute to pixel y. I call them iy1 and iy2.
           For both cases we assume osample+1 subpixels covering pixel y
           (weird). So for case 1 iy1 will be (y-1)*osample and
           iy2 == y*osample. Special treatment of the boundary subpixels
           will compensate for introducing extra subpixel in case 1.
           In case 2 things are more logical: iy1=(yc-y)*osample+(y-1)*osample;
           iy2=(y+1-yc)*osample)+(y-1)*osample. ycen is yc-y making things
           simpler. Note also that the same pattern repeates for all rows:
           we only need to initialize iy1 and iy2 and keep incrementing them
           by osample.
         */
        iy2 = osample - (int)floor(ycen[sp_index(x)] / step) - 1;
        iy1 = iy2 - osample;

        /*
           Handling partial subpixels cut by detector pixel rows is again
           tricky.
           Here we have three cases (mostly because of the decision to assume
           that we always have osample+1 subpixels per one detector pixel).
           Here d1 is the fraction of the subpixel iy1 inside detector pixel y.
           d2 is then the fraction of subpixel iy2 inside detector pixel y.
           By definition d1+d2==step.
           Case 1: ycen falls on the top boundary of each detector pixel
           (ycen == 1). Here we conclude that the first subpixel is fully
           contained inside pixel y and d1 is set to step.
           Case 2: ycen falls on the bottom boundary of each detector pixel
           (ycen == 0). Here we conclude that the first subpixel is totally
           outside of pixel y and d1 is set to 0.
           Case 3: ycen falls inside of each pixel (0>ycen>1). In this case d1
           is set to the fraction of the first step contained inside of each
           pixel.  And BTW, this also means that central line coinsides with
           the upper boundary of subpixel iy2 when the y loop reaches pixel
           y_lower_lim. In other words:
           dy=(iy-(y_lower_lim+ycen[x])*osample)*step-0.5*step
         */

        d1 = fmod(ycen[sp_index(x)], step);
        if (d1 == 0)
            d1 = step;
        d2 = step - d1;

        /*
           The final hurdle for 2D slit decomposition is to construct two
           3D reference tensors. We proceed similar to 1D case except that
           now each iy subpixel can be shifted left or right following the
           curvature of the slit image on the detector. We assume for now that
           each subpixel is exactly 1 detector pixel wide. This may not be
           exactly true if the curvature changes accross the focal plane but
           will deal with it when the necessity will become apparent. For now
           we just assume that a shift delta the weight w assigned to subpixel
           iy is divided between ix1=int(delta) and
           ix2=int(delta)+signum(delta) as (1-|delta-ix1|)*w and |delta-ix1|*w.
           The curvature is given by a quadratic polynomial evaluated from
           an approximation for column x:
           delta = PSF_curve[x][0] + PSF_curve[x][1] * (y-yc[x]) +
               PSF_curve[x][2] * (y-yc[x])^2.
           It looks easy except that y and yc are set in the global detector
           coordinate system rather than in the shifted and cropped swath
           passed to slit_func_2d. One possible solution I will try here is
           to modify PSF_curve before the call such as:
           delta = PSF_curve'[x][0] + PSF_curve'[x][1] * (y'-ycen[x]) +
               PSF_curve'[x][2] * (y'-ycen[x])^2
            where y' = y - floor(yc).
         */

        dy = -(y_lower_lim * osample + floor(ycen[sp_index(x)] / step) + 0.5) * step;

        /* Define initial distance from ycen       */
        /* ie the center of the first subpixel falling into pixel y_lower_lim */

        /*
           Now we go detector pixels x and y incrementing subpixels looking
           for their controibutions to the current and adjacent pixels.
           Note that the curvature/tilt of the projected slit image could be
           so large that subpixel iy may no contribute to column x at all.
           On the other hand, subpixels around ycen by definition must
           contribute to pixel x,y.
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
                delta = (PSF_curve[psf_index(x, 1)] + PSF_curve[psf_index(x, 2)] * dy) * dy;
                ix1 = (int)delta;
                ix2 = ix1 + signum(delta);

                /* Three cases: bottom boundary of row y, intermediate i
                   subpixels and top boundary */

                if (iy == iy1)
                {
                    /* Subpixel iy is entering detector row y        */
                    if (ix1 < ix2)
                    {
                        /* Subpixel iy shifts to the right from column x */
                        if (x + ix1 >= 0 && x + ix2 < ncols)
                        {
                            xx = x + ix1;
                            yy = y + ycen_offset[sp_index(x)] - ycen_offset[sp_index(xx)];
                            xi[xi_index(x, iy, 1)].x = xx;
                            xi[xi_index(x, iy, 1)].y = yy;
                            xi[xi_index(x, iy, 1)].w = w - fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 1)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 1)].w;
                                (m_zeta[mzeta_index(xx, yy)])++;
                            }
                            xx = x + ix2;
                            yy = y + ycen_offset[sp_index(x)] - ycen_offset[sp_index(xx)];
                            xi[xi_index(x, iy, 0)].x = xx;
                            xi[xi_index(x, iy, 0)].y = yy;
                            xi[xi_index(x, iy, 0)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 0)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 0)].w;
                                (m_zeta[mzeta_index(xx, yy)])++;
                            }
                        }
                    }
                    else if (ix1 > ix2)
                    {
                        /* Subpixel iy shifts to the left from column x */
                        if (x + ix2 >= 0 && x + ix1 < ncols)
                        {
                            xx = x + ix2;
                            yy = y + ycen_offset[sp_index(x)] - ycen_offset[sp_index(xx)];
                            xi[xi_index(x, iy, 1)].x = xx;
                            xi[xi_index(x, iy, 1)].y = yy;
                            xi[xi_index(x, iy, 1)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 1)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 1)].w;
                                (m_zeta[mzeta_index(xx, yy)])++;
                            }
                            xx = x + ix1;
                            yy = y + ycen_offset[sp_index(x)] - ycen_offset[sp_index(xx)];
                            xi[xi_index(x, iy, 0)].x = xx;
                            xi[xi_index(x, iy, 0)].y = yy;
                            xi[xi_index(x, iy, 0)].w = w - fabs(delta - ix1) * w;

                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 0)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 0)].w;
                                (m_zeta[mzeta_index(xx, yy)])++;
                            }
                        }
                    }
                    else
                    {
                        /* Subpixel iy stays inside column x */
                        xx = x + ix1;
                        yy = y + ycen_offset[sp_index(x)] - ycen_offset[sp_index(xx)];
                        xi[xi_index(x, iy, 0)].x = xx;
                        xi[xi_index(x, iy, 0)].y = yy;
                        xi[xi_index(x, iy, 0)].w = w;
                        if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows &&
                            w > 0)
                        {
                            zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                            zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                            zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = w;
                            (m_zeta[mzeta_index(xx, yy)])++;
                        }
                    }
                }
                else if (iy == iy2)
                {
                    /* Subpixel iy is leaving detector row y    */
                    if (ix1 < ix2)
                    {
                        /* Subpixel iy shifts to the right from column x */
                        if (x + ix1 >= 0 && x + ix2 < ncols)
                        {
                            xx = x + ix1;
                            yy = y + ycen_offset[sp_index(x)] - ycen_offset[sp_index(xx)];
                            xi[xi_index(x, iy, 3)].x = xx;
                            xi[xi_index(x, iy, 3)].y = yy;
                            xi[xi_index(x, iy, 3)].w = w - fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 3)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 3)].w;
                                (m_zeta[mzeta_index(xx, yy)])++;
                            }
                            xx = x + ix2;
                            yy = y + ycen_offset[sp_index(x)] - ycen_offset[sp_index(xx)];
                            xi[xi_index(x, iy, 2)].x = xx;
                            xi[xi_index(x, iy, 2)].y = yy;
                            xi[xi_index(x, iy, 2)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 2)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 2)].w;
                                (m_zeta[mzeta_index(xx, yy)])++;
                            }
                        }
                    }
                    else if (ix1 > ix2)
                    {
                        /* Subpixel iy shifts to the left from column x */
                        if (x + ix2 >= 0 && x + ix1 < ncols)
                        {
                            xx = x + ix2;
                            yy = y + ycen_offset[sp_index(x)] - ycen_offset[sp_index(xx)];
                            xi[xi_index(x, iy, 3)].x = xx;
                            xi[xi_index(x, iy, 3)].y = yy;
                            xi[xi_index(x, iy, 3)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 3)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 3)].w;
                                (m_zeta[mzeta_index(xx, yy)])++;
                            }
                            xx = x + ix1;
                            yy = y + ycen_offset[sp_index(x)] - ycen_offset[sp_index(xx)];
                            xi[xi_index(x, iy, 2)].x = xx;
                            xi[xi_index(x, iy, 2)].y = yy;
                            xi[xi_index(x, iy, 2)].w = w - fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 2)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 2)].w;
                                (m_zeta[mzeta_index(xx, yy)])++;
                            }
                        }
                    }
                    else
                    {
                        /* Subpixel iy stays inside column x        */
                        xx = x + ix1;
                        yy = y + ycen_offset[sp_index(x)] - ycen_offset[sp_index(xx)];
                        xi[xi_index(x, iy, 2)].x = xx;
                        xi[xi_index(x, iy, 2)].y = yy;
                        xi[xi_index(x, iy, 2)].w = w;
                        if (xx >= 0 && xx < ncols && yy >= 0 &&
                            yy < nrows && w > 0)
                        {
                            zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                            zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                            zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = w;
                            (m_zeta[mzeta_index(xx, yy)])++;
                        }
                    }
                }
                else
                {
                    /* Subpixel iy is fully inside detector row y */
                    if (ix1 < ix2)
                    {
                        /* Subpixel iy shifts to the right from column x   */
                        if (x + ix1 >= 0 && x + ix2 < ncols)
                        {
                            xx = x + ix1;
                            yy = y + ycen_offset[sp_index(x)] - ycen_offset[sp_index(xx)];
                            xi[xi_index(x, iy, 1)].x = xx;
                            xi[xi_index(x, iy, 1)].y = yy;
                            xi[xi_index(x, iy, 1)].w = w - fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 1)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 1)].w;
                                (m_zeta[mzeta_index(xx, yy)])++;
                            }
                            xx = x + ix2;
                            yy = y + ycen_offset[sp_index(x)] - ycen_offset[sp_index(xx)];
                            xi[xi_index(x, iy, 0)].x = xx;
                            xi[xi_index(x, iy, 0)].y = yy;
                            xi[xi_index(x, iy, 0)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 0)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 0)].w;
                                (m_zeta[mzeta_index(xx, yy)])++;
                            }
                        }
                    }
                    else if (ix1 > ix2)
                    {
                        /* Subpixel iy shifts to the left from column x */
                        if (x + ix2 >= 0 && x + ix1 < ncols)
                        {
                            xx = x + ix2;
                            yy = y + ycen_offset[sp_index(x)] - ycen_offset[sp_index(xx)];
                            xi[xi_index(x, iy, 1)].x = xx;
                            xi[xi_index(x, iy, 1)].y = yy;
                            xi[xi_index(x, iy, 1)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 1)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 1)].w;
                                (m_zeta[mzeta_index(xx, yy)])++;
                            }
                            xx = x + ix1;
                            yy = y + ycen_offset[sp_index(x)] - ycen_offset[sp_index(xx)];
                            xi[xi_index(x, iy, 0)].x = xx;
                            xi[xi_index(x, iy, 0)].y = yy;
                            xi[xi_index(x, iy, 0)].w = w - fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 0)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 0)].w;
                                (m_zeta[mzeta_index(xx, yy)])++;
                            }
                        }
                    }
                    else
                    {
                        /* Subpixel iy stays inside column x */
                        xx = x + ix2;
                        yy = y + ycen_offset[sp_index(x)] - ycen_offset[sp_index(xx)];
                        xi[xi_index(x, iy, 0)].x = xx;
                        xi[xi_index(x, iy, 0)].y = yy;
                        xi[xi_index(x, iy, 0)].w = w;
                        if (xx >= 0 && xx < ncols && yy >= 0 &&
                            yy < nrows && w > 0)
                        {
                            zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                            zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                            zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = w;
                            (m_zeta[mzeta_index(xx, yy)])++;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

int slit_func_curved(int ncols,  /* Swath width in pixels                                 */
                     int nrows,  /* Extraction slit height in pixels                      */
                     double *im, /* Image to be decomposed              [nrows][ncols]                  */
                     double *pix_unc,
                     int *mask,        /* Initial and final mask for the swath [nrows][ncols]                 */
                     double *ycen,     /* Order centre line offset from pixel row boundary  [ncols]    */
                     int *ycen_offset, /* Order image column shift     [ncols]                         */
                     double *tilt,     /* slit tilt [ncols], that I later convert to PSF_curve array. */
                     double *shear,    /* slit tilt [ncols], that I later convert to PSF_curve array. */
                     int y_lower_lim,  /* Number of detector pixels below the pixel containing  */
                                       /* the central line yc.                                  */
                     int osample,      /* Subpixel ovsersampling factor                         */
                     double lambda_sP, /* Smoothing parameter for the spectrum, coiuld be zero  */
                     double lambda_sL, /* Smoothing parameter for the slit function, usually >0 */
                     double *sP,       /* Spectrum resulting from decomposition      [ncols]           */
                     double *sL,       /* Slit function resulting from decomposition    [ny]        */
                     double *model,    /* Model constructed from sp and sf        [nrows][ncols]              */
                     double *unc)      /* Spectrum uncertainties based on data - model   [ncols]       */
{
    int x, xx, xxx, y, yy, iy, jy, n, m, ny, y_upper_lim, i, maxiter, tmpi;
    double delta_x, tmp, sum, norm, dev, lambda, diag_tot, ww, www, sP_change, sP_max;
    int info, iter, isum;

    maxiter = 5;
    y_upper_lim = nrows - 1 - y_lower_lim;
    /* The size of the sL array. Extra osample is because ycen can be between 0 and 1. */
    ny = osample * (nrows + 1) + 1;
    _ncols = ncols;
    _nrows = nrows;
    _ny = ny;
    _osample = osample;

    double *sP_old = malloc(MAX_SP * sizeof(double));
    for (i = 0; i < MAX_SP; i++)
        sP_old[i] = 0;
    double *l_Aij = malloc(MAX_LAIJ * sizeof(double));
    for (i = 0; i < MAX_LAIJ; i++)
        l_Aij[i] = 0;
    double *p_Aij = malloc(MAX_PAIJ * sizeof(double));
    for (i = 0; i < MAX_PAIJ; i++)
        p_Aij[i] = 0;
    double *l_bj = malloc(MAX_LBJ * sizeof(double));
    for (i = 0; i < MAX_LBJ; i++)
        l_bj[i] = 0;
    double *p_bj = malloc(MAX_PBJ * sizeof(double));
    for (i = 0; i < MAX_PBJ; i++)
        p_bj[i] = 0;

    /*
      Convolution tensor telling the coordinates of detector pixels on which
      {x, iy} element falls and the corresponding projections. [ncols][ny][4]
    */
    xi_ref *xi = malloc(MAX_XI * sizeof(xi_ref));
    for (i = 0; i < MAX_XI; i++)
        xi[i].w = xi[i].x = xi[i].y = 0;
    /* Convolution tensor telling the coordinates of subpixels {x, iy}
      contributing to detector pixel {x, y}. [ncols][nrows][3*(osample+1)]
    */
    zeta_ref *zeta = malloc(MAX_ZETA * sizeof(zeta_ref));
    for (i = 0; i < MAX_ZETA; i++)
        zeta[i].w = zeta[i].x = zeta[i].iy = 0;

    /* The actual number of contributing elements in zeta  [ncols][nrows]  */
    int *m_zeta = malloc(MAX_MZETA * sizeof(int));
    for (i = 0; i < MAX_MZETA; i++)
        m_zeta[i] = 0;

    //[ncols][3];
    double *PSF_curve = malloc(MAX_PSF * sizeof(double));
    for (i = 0; i < MAX_PSF; i++)
        PSF_curve[i] = 0;

    /* Parabolic fit to the slit image curvature.            */
    /* For column d_x = PSF_curve[ncols][0] +                */
    /*                  PSF_curve[ncols][1] *d_y +           */
    /*                  PSF_curve[ncols][2] *d_y^2,          */
    /* where d_y is the offset from the central line ycen.   */
    /* Thus central subpixel of omega[x][y'][delta_x][iy']   */
    /* does not stick out of column x.                       */

    delta_x = 0.; /* Maximum horizontal shift in detector pixels due to slit image curvature         */
    for (i = 0; i < ncols; i++)
    {
        tmp = (0.5 / osample + y_lower_lim + ycen[sp_index(i)]);
        delta_x = max(delta_x, (int)(fabs(tilt[sp_index(i)] * tmp) + 1));
        tmp = (0.5 / osample + y_upper_lim + (1. - ycen[sp_index(i)]));
        delta_x = max(delta_x, (int)(fabs(tilt[sp_index(i)] * tmp) + 1));
        PSF_curve[psf_index(i, 0)] = 0.;
        PSF_curve[psf_index(i, 1)] = -tilt[sp_index(i)];
        PSF_curve[psf_index(i, 2)] = -shear[sp_index(i)];
    }

    i = xi_zeta_tensors(ncols, nrows, ny, ycen, ycen_offset, y_lower_lim, osample, PSF_curve, xi, zeta, m_zeta);

    /* Loop through sL , sP reconstruction until convergence is reached */
    iter = 0;
    do
    {
        /* Compute slit function sL */
        /* Prepare the RHS and the matrix */
        for (iy = 0; iy < ny; iy++)
        {
            l_bj[lbj_index(iy)] = 0.e0;
            /* Clean RHS                */
            for (jy = 0; jy < MAX_LAIJ_Y; jy++)
                l_Aij[laij_index(iy, jy)] = 0.e0;
        }
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
                                    tmpi = jy - iy + 2 * osample;
#if CHECK_INDEX
                                    if ((tmpi < 0) || (tmpi >= MAX_LAIJ_Y))
                                    {
                                        printf("Index out of Bounds l_Aij[%i, %i]\n", iy, tmpi);
                                        printf("ww = %f, www = %f\n", ww, www);
                                    }
                                    else
#endif
                                        l_Aij[laij_index(iy, tmpi)] +=
                                            sP[sp_index(xxx)] * sP[sp_index(x)] * www * ww * mask[im_index(xx, yy)];
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
        /* Main diagonal  */
        tmpi = 2 * osample; //Middle line
        l_Aij[laij_index(0, tmpi)] += lambda;
        /* Upper diagonal */
        l_Aij[laij_index(0, tmpi + 1)] -= lambda;
        for (iy = 1; iy < ny - 1; iy++)
        {
            /* Lower diagonal */
            l_Aij[laij_index(iy, tmpi - 1)] -= lambda;
            /* Main diagonal  */
            l_Aij[laij_index(iy, tmpi)] += lambda * 2.e0;
            /* Upper diagonal */
            l_Aij[laij_index(iy, tmpi + 1)] -= lambda;
        }
        /* Lower diagonal */
        l_Aij[laij_index(ny - 1, tmpi - 1)] -= lambda;
        /* Main diagonal  */
        l_Aij[laij_index(ny - 1, tmpi)] += lambda;

        /* Solve the system of equations */
        info = bandsol(l_Aij, l_bj, ny, 4 * osample + 1);
        if (info)
            printf("info(sL)=%d\n", info);

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

        /*  Compute spectrum sP */
        for (x = 0; x < ncols; x++)
        {
            for (xx = 0; xx < MAX_PAIJ_Y; xx++)
                p_Aij[paij_index(x, xx)] = 0.;
            p_bj[pbj_index(x)] = 0;
        }
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
                                    tmpi = xxx - x + 2;
#if CHECK_INDEX
                                    if ((tmpi < 0) || (tmpi >= MAX_PAIJ_Y))
                                    {
                                        printf("Index out of Bounds p_Aij[%i, %i]\n", x, tmpi);
                                        printf("ww = %f, www = %f\n", ww, www);
                                    }
                                    else
#endif
                                        p_Aij[paij_index(x, tmpi)] += sL[sl_index(jy)] * sL[sl_index(iy)] * www * ww * mask[im_index(xx, yy)];
                                }
                                p_bj[pbj_index(x)] += im[im_index(xx, yy)] * mask[im_index(xx, yy)] * sL[sl_index(iy)] * ww;
                            }
                        }
                    }
                }
            }
        }

        for (x = 0; x < ncols; x++)
            sP_old[sp_index(x)] = sP[sp_index(x)];
        if (lambda_sP > 0.e0)
        {
            norm = 0.e0;
            for (x = 0; x < ncols; x++)
            {
                norm += sP[sp_index(x)];
            }
            norm /= ncols;
            lambda = lambda_sP * norm;         /* Scale regularization parameter */
            p_Aij[paij_index(0, 2)] += lambda; /* Main diagonal  */
            p_Aij[paij_index(0, 3)] -= lambda; /* Upper diagonal */
            for (x = 1; x < ncols - 1; x++)
            {
                p_Aij[paij_index(x, 1)] -= lambda;        /* Lower diagonal */
                p_Aij[paij_index(x, 2)] += lambda * 2.e0; /* Main diagonal  */
                p_Aij[paij_index(x, 3)] -= lambda;        /* Upper diagonal */
            }
            p_Aij[paij_index(ncols - 1, 1)] -= lambda; /* Lower diagonal */
            p_Aij[paij_index(ncols - 1, 2)] += lambda; /* Main diagonal  */
        }

        /* Solve the system of equations */
        info = bandsol(p_Aij, p_bj, ncols, 5);
        if (info)
            printf("info(sP)=%d\n", info);
        for (x = 0; x < ncols; x++)
            sP[sp_index(x)] = p_bj[pbj_index(x)];

        /* Compute the model */
        for (y = 0; y < nrows; y++)
        {
            for (x = 0; x < ncols; x++)
            {
                model[im_index(x, y)] = 0.;
            }
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
                    model[im_index(x, y)] += sP[sp_index(xx)] * sL[sl_index(iy)] * ww;
                }
            }
        }
        /* Compare model and data */
        sum = 0.e0;
        isum = 0;
        for (y = 0; y < nrows; y++)
        {
            for (x = (int)delta_x; x < ncols - delta_x; x++)
            {
                sum += mask[im_index(x, y)] * (model[im_index(x, y)] - im[im_index(x, y)]) * (model[im_index(x, y)] - im[im_index(x, y)]);
                isum += mask[im_index(x, y)];
            }
        }
        dev = sqrt(sum / isum);
        /* Adjust the mask marking outlyers */
        for (y = 0; y < nrows; y++)
        {
            for (x = (int)delta_x; x < ncols - delta_x; x++)
            {
                if (fabs(model[im_index(x, y)] - im[im_index(x, y)]) > 6. * dev)
                    mask[im_index(x, y)] = 0;
                else
                    mask[im_index(x, y)] = 1;
            }
        }

        /* Compute the change in the spectrum */
        sP_change = 0.e0;
        sP_max = 1.e0;
        for (x = 0; x < ncols; x++)
        {
            if (sP[sp_index(x)] > sP_max)
                sP_max = sP[sp_index(x)];
            if (fabs(sP[sp_index(x)] - sP_old[sp_index(x)]) > sP_change)
                sP_change = fabs(sP[sp_index(x)] - sP_old[sp_index(x)]);
        }
        /* Check for convergence */
    } while (iter++ < maxiter && sP_change > 1e-6 * sP_max);

    /* Uncertainty estimate */
    for (x = 0; x < ncols; x++)
    {
        unc[sp_index(x)] = 0.;
        p_bj[pbj_index(x)] = 0.;
    }
    for (y = 0; y < nrows; y++)
    {
        for (x = 0; x < ncols; x++)
        {
            // Loop through all pixels contributing to x,y
            for (m = 0; m < m_zeta[mzeta_index(x, y)]; m++)
            {
                xx = zeta[zeta_index(x, y, m)].x;
                iy = zeta[zeta_index(x, y, m)].iy;
                ww = zeta[zeta_index(x, y, m)].w;
                unc[sp_index(xx)] += (im[im_index(x, y)] - model[im_index(x, y)]) *
                                     (im[im_index(x, y)] - model[im_index(x, y)]) *
                                     ww * mask[im_index(x, y)];
                unc[sp_index(xx)] += pix_unc[im_index(x, y)] * pix_unc[im_index(x, y)] *
                                     ww * mask[im_index(x, y)];
                // Norm
                p_bj[pbj_index(xx)] += ww * mask[im_index(x, y)];
            }
        }
    }
    for (x = 0; x < ncols; x++)
    {
        unc[sp_index(x)] = sqrt(unc[sp_index(x)] / p_bj[pbj_index(x)] * nrows);
    }
    free(sP_old);
    free(l_Aij);
    free(p_Aij);
    free(l_bj);
    free(p_bj);

    free(xi);
    free(zeta);
    free(m_zeta);
    free(PSF_curve);

    return 0;
}
