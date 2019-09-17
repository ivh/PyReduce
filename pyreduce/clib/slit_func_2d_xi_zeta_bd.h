typedef struct
{
    int x;
    int y;    /* Coordinates of target pixel x,y  */
    double w; /* Contribution weight <= 1/osample */
} xi_ref;

typedef struct
{
    int x;
    int iy;   /* Contributing subpixel  x,iy      */
    double w; /* Contribution weight <= 1/osample */
} zeta_ref;

int slit_func_curved(int ncols,
                     int nrows,
                     int nx,
                     int ny,
                     double *im,
                     double *pix_unc,
                     int *mask_orig,
                     double *ycen,
                     int *ycen_offset,
                     int y_lower_lim,
                     int osample,
                     double lambda_sP,
                     double lambda_sL,
                     double *PSF_curve,
                     double *sP,
                     double *sL,
                     double *model,
                     double *unc,
                     int *mask,
                     double *l_Aij,
                     double *l_bj,
                     double *p_Aij,
                     double *p_bj);
