#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

#define omega_index(x, y, z) z + (y * ny) + (x * ny * nrows)
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

int slit_func_vert(int ncols,        /* Swath width in pixels                                 */
                   int nrows,        /* Extraction slit height in pixels                      */
                   double *im,       /* Image to be decomposed                                */
                   double *pix_unc,  /* Image Uncertianty to be decomposed                                */
                   int *mask,        /* Initial and final mask for the swath                  */
                   double *ycen,     /* Order centre line offset from pixel row boundary      */
                   int osample,      /* Subpixel ovsersampling factor                         */
                   double lambda_sP, /* Smoothing parameter for the spectrum, coiuld be zero  */
                   double lambda_sL, /* Smoothing parameter for the slit function, usually >0 */
                   double *sP,       /* Spectrum resulting from decomposition                 */
                   double *sL,       /* Slit function resulting from decomposition            */
                   double *model,    /* Model constructed from sp and sf                      */
                   double *unc       /* Spectrum uncertainties                                */
)
{
  int x, y, iy, jy, iy1, iy2, ny, nd;
  double step, d1, d2, sum, norm, dev, lambda, diag_tot, sP_change, sP_max;
  int info, iter, isum;
  /* Initialise */
  nd = 2 * osample + 1;
  ny = osample * (nrows + 1) + 1; /* The size of the sf array */
  step = 1.e0 / osample;
  double *E = malloc(ncols * sizeof(double));                    // double E[ncols];
  double *sP_old = malloc(ncols * sizeof(double));               // double sP_old[ncols];
  double *Aij = malloc(ny * (2 * osample + 1) * sizeof(double)); // double Aij[ny*ny];
  double *bj = malloc(ny * sizeof(double));                      // double bj[ny];
  double *Adiag = malloc(ncols * 3 * sizeof(double));            // double Adiag[ncols*3];
  double *omega = malloc(ny * nrows * ncols * sizeof(double));   // double omega[ny][nrows][ncols];
  double *p_bj = malloc(ncols * sizeof(double));                 // index as: [iy+(y*ny)+(x*ny*nrows)]

  for (int k = 0; k < ncols * 3; k++)
    Adiag[k] = 0; // Initialize to zero

  /*
  Construct the omega tensor. Normally it has the dimensionality of
  ny*nrows*ncols.
  The tensor is mostly empty and can be easily compressed to ny*nx, but
  this will complicate matrix operations at later stages. I will keep
  it as it is for now.
  Note, that omega is used in in the equations for sL, sP and for the model
  but it does not involve the data, only the geometry. Thus it can be
  pre-computed once.
  */
  for (x = 0; x < ncols; x++)
  {
    iy2 = osample - floor(ycen[x] / step) - 1;
    iy1 = iy2 - osample;

    d1 = fmod(ycen[x], step);
    if (d1 == 0)
      d1 = step;
    d2 = step - d1;

    for (y = 0; y < nrows; y++)
    {
      iy1 += osample;
      iy2 += osample;
      for (iy = 0; iy < ny; iy++)
      {
        if (iy < iy1)
          omega[omega_index(x, y, iy)] = 0.;
        else if (iy == iy1)
          omega[omega_index(x, y, iy)] = d1;
        else if (iy > iy1 && iy < iy2)
          omega[omega_index(x, y, iy)] = step;
        else if (iy == iy2)
          omega[omega_index(x, y, iy)] = d2;
        else
          omega[omega_index(x, y, iy)] = 0.;
      }
    }
  }

  /* Loop through sL , sP reconstruction until convergence is reached */
  iter = 0;
  do
  {
    /* Compute slit function sL */
    /* Fill in SLE arrays */
    diag_tot = 0.e0;
    for (iy = 0; iy < ny; iy++)
    {
      bj[iy] = 0.e0;
      for (jy = max(iy - osample, 0); jy <= min(iy + osample, ny - 1); jy++)
      {
        Aij[iy + ny * (jy - iy + osample)] = 0.e0;
        for (x = 0; x < ncols; x++)
        {
          sum = 0.e0;
          for (y = 0; y < nrows; y++)
            sum += omega[omega_index(x, y, iy)] *
                   omega[omega_index(x, y, jy)] * mask[y * ncols + x];
          Aij[iy + ny * (jy - iy + osample)] += sum * sP[x] * sP[x];
        }
      }
      for (x = 0; x < ncols; x++)
      {
        sum = 0.e0;
        for (y = 0; y < nrows; y++)
          sum += omega[omega_index(x, y, iy)] *
                 mask[y * ncols + x] * im[y * ncols + x];
        bj[iy] += sum * sP[x];
      }
      diag_tot += Aij[iy + ny * osample];
    }

    /* Scale regularization parameters */
    lambda = lambda_sL * diag_tot / ny;

    /* Add regularization parts for the slit function */
    Aij[ny * osample] += lambda;       /* Main diagonal  */
    Aij[ny * (osample + 1)] -= lambda; /* Upper diagonal */
    for (iy = 1; iy < ny - 1; iy++)
    {
      Aij[iy + ny * (osample - 1)] -= lambda;  /* Lower diagonal */
      Aij[iy + ny * osample] += lambda * 2.e0; /* Main diagonal  */
      Aij[iy + ny * (osample + 1)] -= lambda;  /* Upper diagonal */
    }
    Aij[ny - 1 + ny * (osample - 1)] -= lambda; /* Lower diagonal */
    Aij[ny - 1 + ny * osample] += lambda;       /* Main diagonal  */

    /* Solve the system of equations */
    info = bandsol(Aij, bj, ny, nd);
    if (info)
      printf("Bandsol exited with %d", info);

    /* Normalize the slit function */
    norm = 0.e0;
    for (iy = 0; iy < ny; iy++)
    {
      sL[iy] = bj[iy];
      norm += sL[iy];
    }
    norm /= osample;
    for (iy = 0; iy < ny; iy++)
      sL[iy] /= norm;

    /* Compute spectrum sP */
    for (x = 0; x < ncols; x++)
    {
      Adiag[x + ncols] = 0.e0;
      E[x] = 0.e0;
      for (y = 0; y < nrows; y++)
      {
        sum = 0.e0;
        for (iy = 0; iy < ny; iy++)
        {
          sum += omega[omega_index(x, y, iy)] * sL[iy];
        }

        Adiag[x + ncols] += sum * sum * mask[y * ncols + x];
        E[x] += sum * im[y * ncols + x] * mask[y * ncols + x];
      }
    }
    if (lambda_sP > 0.e0)
    {
      norm = 0.e0;
      for (x = 0; x < ncols; x++)
      {
        sP_old[x] = sP[x];
        norm += sP[x];
      }
      norm /= ncols;
      lambda = lambda_sP * norm;
      Adiag[0] = 0.e0;
      Adiag[0 + ncols] += lambda;
      Adiag[0 + ncols * 2] -= lambda;
      for (x = 1; x < ncols - 1; x++)
      {
        Adiag[x] = -lambda;
        Adiag[x + ncols] += 2.e0 * lambda;
        Adiag[x + ncols * 2] = -lambda;
      }
      Adiag[ncols - 1] -= lambda;
      Adiag[ncols - 1 + ncols] += lambda;
      Adiag[ncols - 1 + 2 * ncols] = 0.e0;

      info = bandsol(Adiag, E, ncols, 3);
      for (x = 0; x < ncols; x++)
        sP[x] = E[x];
    }
    else
    {
      for (x = 0; x < ncols; x++)
      {
        sP_old[x] = sP[x];
        sP[x] = E[x] / Adiag[x + ncols];
      }
    }

    /* Compute the model */
    for (y = 0; y < nrows; y++)
    {
      for (x = 0; x < ncols; x++)
      {
        sum = 0.e0;
        for (iy = 0; iy < ny; iy++)
          sum += omega[omega_index(x, y, iy)] * sL[iy];
        model[y * ncols + x] = sum * sP[x];
      }
    }

    /* Compare model and data */
    sum = 0.e0;
    isum = 0;
    for (y = 0; y < nrows; y++)
    {
      for (x = 0; x < ncols; x++)
      {
        sum += mask[y * ncols + x] * (model[y * ncols + x] - im[y * ncols + x]) *
               (model[y * ncols + x] - im[y * ncols + x]);
        isum += mask[y * ncols + x];
      }
    }
    dev = sqrt(sum / isum);

    /* Adjust the mask marking outlyers */
    for (y = 0; y < nrows; y++)
    {
      for (x = 0; x < ncols; x++)
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
    /* Check the convergence */
  } while (iter++ < 5 && sP_change > 1e-5 * sP_max);

  /* Uncertainty estimate */
  for (x = 0; x < ncols; x++)
  {
    unc[x] = 0.;
    p_bj[x] = 0.;
  }
  for (x = 0; x < ncols; x++)
  {
    // Loop through all pixels contributing to x
    for (y = 0; y < nrows; y++)
    {
      unc[x] += (im[y * ncols + x] - model[y * ncols + x]) *
                (im[y * ncols + x] - model[y * ncols + x]) *
                sL[y] * mask[y * ncols + x];
      unc[x] += pix_unc[y * ncols + x] * pix_unc[y * ncols + x] *
                sL[y] * mask[y * ncols + x];
      // Norm
      p_bj[x] += sL[y] * mask[y * ncols + x];
    }
  }
  for (x = 0; x < ncols; x++)
  {
    unc[x] = sqrt(unc[x] / p_bj[x] * nrows);
  }

  if (!(sP[0] == sP[0]))
  {
    printf("SOMETHING WENT WRONG STOP HERE!!!\n");
    // raise(SIGINT);
  }

  free(E);
  free(sP_old);
  free(omega);
  free(Aij);
  free(bj);
  free(Adiag);
  free(p_bj);

  return 0;
}
