#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "slit_func_2d_xi_zeta_bd.h"

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define signum(a) (((a) > 0) ? 1 : ((a) < 0) ? -1 : 0)

#define zeta_index(x, y, z) (z * ncols * nrows) + (y * ncols) + x
#define mzeta_index(x, y) (y * ncols) + x
#define xi_index(x, y, z) (z * ncols * ny) + (y * ncols) + x

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

    //if(fmod(nd,2)==0) return -1;

    /* Forward sweep */
    for (i = 0; i < n - 1; i++)
    {
        aa = a[i + n * (nd / 2)];
        //if(aa==0.e0) return -3;
        r[i] /= aa;
        for (j = 0; j < nd; j++)
            a[i + j * n] /= aa;
        for (j = 1; j < min(nd / 2 + 1, n - i); j++)
        {
            aa = a[i + j + n * (nd / 2 - j)];
            //if(aa==0.e0) return -j;
            r[i + j] -= r[i] * aa;
            for (k = 0; k < n * (nd - j); k += n)
                a[i + j + k] -= a[i + k + n * j] * aa;
        }
    }

    /* Backward sweep */
    r[n - 1] /= a[n - 1 + n * (nd / 2)];
    for (i = n - 1; i > 0; i--)
    {
        for (j = 1; j <= min(nd / 2, i); j++)
            r[i - j] -= r[i] * a[i - j + n * (nd / 2 + j)];
        //if(a[i-1+n*(nd/2)]==0.e0) return -5;
        r[i - 1] /= a[i - 1 + n * (nd / 2)];
    }

    //if(a[n*(nd/2)]==0.e0) return -6;
    r[0] /= a[n * (nd / 2)];
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
    double PSF_curve[ncols][3],
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
            for (ix = 0; ix < 3 * (osample + 1); ix++)
            {
                zeta[zeta_index(x, y, ix)].x = 0;
                zeta[zeta_index(x, y, ix)].iy = 0;
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
        iy2 = osample - floor(ycen[x] / step) - 1;
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

        d1 = fmod(ycen[x], step);
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

        dy = -(y_lower_lim * osample + floor(ycen[x] / step) + 0.5) * step;

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
                delta = (PSF_curve[x][1] + PSF_curve[x][2] * dy) * dy;
                ix1 = delta;
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
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 1)].x = xx;
                            xi[xi_index(x, iy, 1)].y = yy;
                            xi[xi_index(x, iy, 1)].w = w - fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 1)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 1)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                            xx = x + ix2;
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 0)].x = xx;
                            xi[xi_index(x, iy, 0)].y = yy;
                            xi[xi_index(x, iy, 0)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 0)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 0)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else if (ix1 > ix2)
                    {
                        /* Subpixel iy shifts to the left from column x */
                        if (x + ix2 >= 0 && x + ix1 < ncols)
                        {
                            xx = x + ix2;
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 1)].x = xx;
                            xi[xi_index(x, iy, 1)].y = yy;
                            xi[xi_index(x, iy, 1)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 1)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 1)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                            xx = x + ix1;
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 0)].x = xx;
                            xi[xi_index(x, iy, 0)].y = yy;
                            xi[xi_index(x, iy, 0)].w = w - fabs(delta - ix1) * w;

                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 0)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 0)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else
                    {
                        /* Subpixel iy stays inside column x */
                        xx = x + ix1;
                        yy = y + ycen_offset[x] - ycen_offset[xx];
                        xi[xi_index(x, iy, 0)].x = xx;
                        xi[xi_index(x, iy, 0)].y = yy;
                        xi[xi_index(x, iy, 0)].w = w;
                        if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows &&
                            w > 0)
                        {
                            zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                            zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                            zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = w;
                            m_zeta[mzeta_index(xx, yy)]++;
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
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 3)].x = xx;
                            xi[xi_index(x, iy, 3)].y = yy;
                            xi[xi_index(x, iy, 3)].w = w - fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 3)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 3)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                            xx = x + ix2;
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 2)].x = xx;
                            xi[xi_index(x, iy, 2)].y = yy;
                            xi[xi_index(x, iy, 2)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 2)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 2)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else if (ix1 > ix2)
                    {
                        /* Subpixel iy shifts to the left from column x */
                        if (x + ix2 >= 0 && x + ix1 < ncols)
                        {
                            xx = x + ix2;
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 3)].x = xx;
                            xi[xi_index(x, iy, 3)].y = yy;
                            xi[xi_index(x, iy, 3)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 3)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 3)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                            xx = x + ix1;
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 2)].x = xx;
                            xi[xi_index(x, iy, 2)].y = yy;
                            xi[xi_index(x, iy, 2)].w = w - fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 2)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 2)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else
                    {
                        /* Subpixel iy stays inside column x        */
                        xx = x + ix1;
                        yy = y + ycen_offset[x] - ycen_offset[xx];
                        xi[xi_index(x, iy, 2)].x = xx;
                        xi[xi_index(x, iy, 2)].y = yy;
                        xi[xi_index(x, iy, 2)].w = w;
                        if (xx >= 0 && xx < ncols && yy >= 0 &&
                            yy < nrows && w > 0)
                        {
                            zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                            zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                            zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = w;
                            m_zeta[mzeta_index(xx, yy)]++;
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
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 1)].x = xx;
                            xi[xi_index(x, iy, 1)].y = yy;
                            xi[xi_index(x, iy, 1)].w = w - fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 1)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 1)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                            xx = x + ix2;
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 0)].x = xx;
                            xi[xi_index(x, iy, 0)].y = yy;
                            xi[xi_index(x, iy, 0)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 0)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 0)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else if (ix1 > ix2)
                    {
                        /* Subpixel iy shifts to the left from column x */
                        if (x + ix2 >= 0 && x + ix1 < ncols)
                        {
                            xx = x + ix2;
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 1)].x = xx;
                            xi[xi_index(x, iy, 1)].y = yy;
                            xi[xi_index(x, iy, 1)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 1)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 1)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                            xx = x + ix1;
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 0)].x = xx;
                            xi[xi_index(x, iy, 0)].y = yy;
                            xi[xi_index(x, iy, 0)].w = w - fabs(delta - ix1) * w;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows && xi[xi_index(x, iy, 0)].w > 0)
                            {
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                                zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = xi[xi_index(x, iy, 0)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else
                    {
                        /* Subpixel iy stays inside column x */
                        xx = x + ix2;
                        yy = y + ycen_offset[x] - ycen_offset[xx];
                        xi[xi_index(x, iy, 0)].x = xx;
                        xi[xi_index(x, iy, 0)].y = yy;
                        xi[xi_index(x, iy, 0)].w = w;
                        if (xx >= 0 && xx < ncols && yy >= 0 &&
                            yy < nrows && w > 0)
                        {
                            zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].x = x;
                            zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].iy = iy;
                            zeta[zeta_index(xx, yy, m_zeta[mzeta_index(xx, yy)])].w = w;
                            m_zeta[mzeta_index(xx, yy)]++;
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
    int x, xx, xxx, y, yy, iy, jy, n, m, ny, y_upper_lim, i;
    double delta_x, sum, norm, dev, lambda, diag_tot, ww, www, sP_change, sP_max;
    int info, iter, isum;

    int maxiter = 5;

    ny = osample * (nrows + 1) + 1; /* The size of the sL array. Extra osample is because ycen can be between 0 and 1. */

    double *sP_old = malloc(ncols * sizeof(double));
    double *l_Aij = malloc(ny * (4 * osample + 1) * sizeof(double));
    double *p_Aij = malloc(ncols * 5 * sizeof(double));
    double *l_bj = malloc(ny * sizeof(double));
    double *p_bj = malloc(ncols * sizeof(double));

    /*
      Convolution tensor telling the coordinates of detector pixels on which
      {x, iy} element falls and the corresponding projections. [ncols][ny][4]
    */
    xi_ref *xi = malloc(ncols * ny * 4 * sizeof(xi_ref));

    /* Convolution tensor telling the coordinates of subpixels {x, iy}
      contributing to detector pixel {x, y}. [ncols][nrows][3*(osample+1)]
    */
    zeta_ref *zeta = malloc(ncols * nrows * 3 * (osample + 1) * sizeof(zeta_ref));

    /* The actual number of contributing elements in zeta  [ncols][nrows]  */
    int *m_zeta = malloc(ncols * nrows * sizeof(int));

    double PSF_curve[ncols][3]; /* Parabolic fit to the slit image curvature. [ncols][3]           */
                                /* For column d_x = PSF_curve[ncols][0] +                */
                                /*                  PSF_curve[ncols][1] *d_y +           */
                                /*                  PSF_curve[ncols][2] *d_y^2,          */
                                /* where d_y is the offset from the central line ycen.   */
                                /* Thus central subpixel of omega[x][y'][delta_x][iy']   */
                                /* does not stick out of column x.                       */

    y_upper_lim = nrows - 1 - y_lower_lim;
    delta_x = 0.; /* Maximum horizontal shift in detector pixels due to slit image curvature         */
    for (i = 0; i < ncols; i++)
    {
        delta_x = max(delta_x, (int)(fabs(tilt[i] * (0.5 / osample + y_lower_lim + ycen[i])) + 1));
        delta_x = max(delta_x, (int)(fabs(tilt[i] * (0.5 / osample + y_upper_lim + (1. - ycen[i]))) + 1));
        PSF_curve[i][0] = 0.;
        PSF_curve[i][1] = -tilt[i];
        PSF_curve[i][2] = -shear[i];
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
            l_bj[iy] = 0.e0;
            /* Clean RHS                */
            for (jy = 0; jy <= 4 * osample; jy++)
                l_Aij[iy + ny * jy] = 0.e0;
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
                        if (m_zeta[mzeta_index(xx, yy)] > 0 && xx >= 0 && xx < ncols &&
                            yy >= 0 && yy < nrows)
                        {
                            for (m = 0; m < m_zeta[mzeta_index(xx, yy)]; m++)
                            {
                                xxx = zeta[zeta_index(xx, yy, m)].x;
                                jy = zeta[zeta_index(xx, yy, m)].iy;
                                www = zeta[zeta_index(xx, yy, m)].w;
                                l_Aij[iy + ny * (jy - iy + 2 * osample)] +=
                                    sP[xxx] * sP[x] * www * ww * mask[yy * ncols + xx];
                            }
                            l_bj[iy] += im[yy * ncols + xx] * mask[yy * ncols + xx] * sP[x] * ww;
                        }
                    }
                }
            }
            diag_tot += l_Aij[iy + ny * 2 * osample];
        }
        /* Scale regularization parameters */
        lambda = lambda_sL * diag_tot / ny;
        /* Add regularization parts for the SLE matrix */
        /* Main diagonal  */
        l_Aij[ny * 2 * osample] += lambda;
        /* Upper diagonal */
        l_Aij[ny * (2 * osample + 1)] -= lambda;
        for (iy = 1; iy < ny - 1; iy++)
        {
            /* Lower diagonal */
            l_Aij[iy + ny * (2 * osample - 1)] -= lambda;
            /* Main diagonal  */
            l_Aij[iy + ny * 2 * osample] += lambda * 2.e0;
            /* Upper diagonal */
            l_Aij[iy + ny * (2 * osample + 1)] -= lambda;
        }
        /* Lower diagonal */
        l_Aij[ny - 1 + ny * (2 * osample - 1)] -= lambda;
        /* Main diagonal  */
        l_Aij[ny - 1 + ny * 2 * osample] += lambda;

        /* Solve the system of equations */
        info = bandsol(l_Aij, l_bj, ny, 4 * osample + 1);
        if (info)
            printf("info(sL)=%d\n", info);

        /* Normalize the slit function */
        norm = 0.e0;
        for (iy = 0; iy < ny; iy++)
        {
            sL[iy] = l_bj[iy];
            norm += sL[iy];
        }
        norm /= osample;
        for (iy = 0; iy < ny; iy++)
            sL[iy] /= norm;

        /*  Compute spectrum sP */
        for (x = 0; x < ncols; x++)
        {
            for (xx = 0; xx < 5; xx++)
                p_Aij[xx * ncols + x] = 0.;
            p_bj[x] = 0;
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
                        if (m_zeta[mzeta_index(xx, yy)] > 0 && xx >= 0 && xx < ncols &&
                            yy >= 0 && yy < nrows)
                        {
                            for (m = 0; m < m_zeta[mzeta_index(xx, yy)]; m++)
                            {
                                xxx = zeta[zeta_index(xx, yy, m)].x;
                                jy = zeta[zeta_index(xx, yy, m)].iy;
                                www = zeta[zeta_index(xx, yy, m)].w;
                                p_Aij[x + ncols * (xxx - x + 2)] += sL[jy] *
                                                                    sL[iy] * www * ww * mask[yy * ncols + xx];
                            }
                            p_bj[x] += im[yy * ncols + xx] * mask[yy * ncols + xx] * sL[iy] * ww;
                        }
                    }
                }
            }
        }

        for (x = 0; x < ncols; x++)
            sP_old[x] = sP[x];
        if (lambda_sP > 0.e0)
        {
            norm = 0.e0;
            for (x = 0; x < ncols; x++)
            {
                norm += sP[x];
            }
            norm /= ncols;
            lambda = lambda_sP * norm;  /* Scale regularization parameter */
            p_Aij[ncols * 2] += lambda; /* Main diagonal  */
            p_Aij[ncols * 3] -= lambda; /* Upper diagonal */
            for (x = 1; x < ncols - 1; x++)
            {
                p_Aij[x + ncols] -= lambda;            /* Lower diagonal */
                p_Aij[x + ncols * 2] += lambda * 2.e0; /* Main diagonal  */
                p_Aij[x + ncols * 3] -= lambda;        /* Upper diagonal */
            }
            p_Aij[ncols - 1 + ncols] -= lambda;     /* Lower diagonal */
            p_Aij[ncols - 1 + ncols * 2] += lambda; /* Main diagonal  */
        }

        /* Solve the system of equations */
        info = bandsol(p_Aij, p_bj, ncols, 5);
        if (info)
            printf("info(sP)=%d\n", info);
        for (x = 0; x < ncols; x++)
            sP[x] = p_bj[x];

        /* Compute the model */
        for (y = 0; y < nrows; y++)
        {
            for (x = 0; x < ncols; x++)
            {
                model[y * ncols + x] = 0.;
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
                    model[y * ncols + x] += sP[xx] * sL[iy] * ww;
                }
            }
        }
        /* Compare model and data */
        sum = 0.e0;
        isum = 0;
        for (y = 0; y < nrows; y++)
        {
            for (x = delta_x; x < ncols - delta_x; x++)
            {
                sum += mask[y * ncols + x] * (model[y * ncols + x] - im[y * ncols + x]) * (model[y * ncols + x] - im[y * ncols + x]);
                isum += mask[y * ncols + x];
            }
        }
        dev = sqrt(sum / isum);
        /* Adjust the mask marking outlyers */
        for (y = 0; y < nrows; y++)
        {
            for (x = delta_x; x < ncols - delta_x; x++)
            {
                if (fabs(model[y * ncols + x] - im[y * ncols + x]) > 6. * dev)
                    mask[y * ncols + x] = 0;
                else
                    mask[y * ncols + x] = 1;
            }
        }

        /* Compute the change in the spectrum */
        sP_change = 0.e0;
        sP_max = 1.e0;
        for (x = 0; x < ncols; x++)
        {
            if (sP[x] > sP_max)
                sP_max = sP[x];
            if (fabs(sP[x] - sP_old[x]) > sP_change)
                sP_change = fabs(sP[x] - sP_old[x]);
        }
        /* Check for convergence */
    } while (iter++ < maxiter && sP_change > 1e-6 * sP_max);

    /* Uncertainty estimate */
    for (x = 0; x < ncols; x++)
    {
        unc[x] = 0.;
        p_bj[x] = 0.;
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
                unc[xx] += (im[y * ncols + x] - model[y * ncols + x]) *
                           (im[y * ncols + x] - model[y * ncols + x]) *
                           ww * mask[y * ncols + x];
                unc[xx] += pix_unc[y * ncols + x] * pix_unc[y * ncols + x] *
                           ww * mask[y * ncols + x];
                // Norm
                p_bj[xx] += ww * mask[y * ncols + x];
            }
        }
    }
    for (x = 0; x < ncols; x++)
    {
        unc[x] = sqrt(unc[x] / p_bj[x] * nrows);
    }
    free(sP_old);
    free(l_Aij);
    free(p_Aij);
    free(l_bj);
    free(p_bj);

    free(xi);
    free(zeta);
    free(m_zeta);
    return 0;
}
