/* Copied from CharSlit (github.com/ivh/CharSlit), commit 093dbd3.
   Plain-C slit-decomposition algorithm, compiled here via CFFI.
   Do not edit here: change CharSlit, then run tools/sync_slitdec.py.
   Only divergence: trailing whitespace is stripped.
   PYREDUCE_EXTRACTION=charslit runs the external charslit package instead. */

/* Slit decomposition, optionally with the spectrum oversampled along x (see slitdec.c). */

typedef struct
{
    int x;    /* Contributing fine spectrum bin (x * osample_x + p) */
    int iy;   /* Contributing slit-function subpixel                */
    double w; /* Contribution weight                                */
} zeta_ref_ox;

typedef struct
{
    int min_iy, max_iy;
    int min_x, max_x;   /* in fine-bin units */
} zeta_rng_ox;

#define SLITDEC_OX_KMAX 15 /* highest derivative order of the penalties */

/* Columns at either end of a swath solution that should not be used, counted beyond
   the largest slit-image shift: they can be off by many times the flux. slitdec()
   returns them as solved; give them zero weight when merging swaths. */
#define SLITDEC_EDGE 2

/* Number of such columns per end, as slitdec() reports in info[4]: the largest shift
   from slitcurve (ncols x 6) and slitdeltas (osample * (nrows + 1) + 1), rounded up,
   plus SLITDEC_EDGE. y_lower_lim < 0 means nrows / 2. */
int slitdec_margin(int ncols, int nrows, const double *slitcurve, const double *slitdeltas,
                   int osample, int y_lower_lim);

/* The first 21 arguments are the historic slitdec() ones. Passing osample_x = 1,
   mu_x = NULL, nk_x = 0, mu_y = NULL, nk_y = 0, y_lower_lim = -1,
   sP_fine = NULL, unc_mode = 0, out_x = 1, nlag = 0, cov = NULL, chi2 = NULL for the rest
   gives the historic result, except that the tilt is evaluated at the subpixel
   centre (the historic code used its lower edge).
   osample_x: fine spectrum bins per column. mu_x, mu_y: penalties
   sum_k mu[k-1] * F * int (d^k s)^2 in physical units for the spectrum and the slit
   function (none for the slit function: lambda_sL). lambda_sP: the unscaled
   first-difference penalty, between fine bins. y_lower_lim: rows below the trace row
   in the swath, < 0 means nrows / 2. sP_fine (may be NULL): ncols * osample_x.
   unc_mode 0: residual-based unc; 1: propagated pixel variances; 2: propagated, scaled
   by chi^2. cov (may be NULL): (ncols*out_x) x (nlag+1) covariances on the out_x grid. */
int slitdec(int ncols, int nrows, double *im, double *pix_unc,
            unsigned char *mask, double *ycen, double *slitcurve,
            double *slitdeltas, int osample, double lambda_sP, double lambda_sL,
            int maxiter, double kappa, int use_preset, int weight_mode,
            double noise_floor, double *sP, double *sL, double *model,
            double *unc, double *info,
            int osample_x, const double *mu_x, int nk_x, const double *mu_y, int nk_y,
            int y_lower_lim, double *sP_fine,
            int unc_mode, int out_x, int nlag, double *cov, double *chi2);
