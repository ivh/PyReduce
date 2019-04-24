
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

int slit_func_curved(int ncols,  /* Swath width in pixels */
                     int nrows,  /* Extraction slit height in pixels */
                     double *im, /* Image to be decomposed [nrows][ncols] */
                     double *pix_unc,
                     int *mask,        /* Initial and final mask for the swath [nrows][ncols] */
                     double *ycen,     /* Order centre line offset from pixel row boundary [ncols] */
                     int *ycen_offset, /* Order image column shift [ncols] */
                     double *tilt,     /* slit tilt [ncols], that I later convert to PSF_curve array */
                     double *shear,    /* slit tilt [ncols], that I later convert to PSF_curve array */
                     int y_lower_lim,  /* Number of detector pixels below the pixel containing */
                                       /* the central line yc */
                     int osample,      /* Subpixel ovsersampling factor */
                     double lambda_sP, /* Smoothing parameter for the spectrum, coiuld be zero */
                     double lambda_sL, /* Smoothing parameter for the slit function, usually >0 */
                     double *sP,       /* Spectrum resulting from decomposition [ncols] */
                     double *sL,       /* Slit function resulting from decomposition [ny] */
                     double *model,    /* Model constructed from sp and sf [nrows][ncols] */
                     double *unc       /* Spectrum uncertainties based on data - model [ncols] */
);