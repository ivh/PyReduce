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
                     double *info);

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
                    int *m_zeta);

int create_spectral_model(
    int ncols, 
    int nrows, 
    int osample,
    xi_ref* xi, 
    double* spec, 
    double* slitfunc,
    double* img);