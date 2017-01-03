#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cpl.h>

typedef unsigned char byte;
#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))


// returns model
cpl_image * slit_func_vert(int ncols,             /* Swath width in pixels                                 */ 
                   int nrows,                     /* Extraction slit height in pixels                      */
                   int osample,                   /* Subpixel ovsersampling factor                         */ 
                   cpl_image * im_cpl,            /* Image to be decomposed                                */
                   cpl_vector * ycen_cpl,         /* Order centre line offset from pixel row boundary      */
                   cpl_vector * sL_cpl,           /* Slit function resulting from decomposition, start     */
                                                  /* guess is input, gets overwriteten with result         */
                   cpl_vector * sP_cpl,           /* Spectrum resulting from decomposition                 */
                   double lambda_sP,              /* Smoothing parameter for the spectrum, coiuld be zero  */
                   double lambda_sL,              /* Smoothing parameter for the slit function, usually >0 */
                   double sP_stop,                /* Fraction of spectyrum change, stop condition          */
                   int maxiter                   /* Max number of iterations                              */
    ) {
	int x, y, iy, jy, iy1, iy2, ny;
	double step, d1, d2, sum, norm, dev, lambda, diag_tot, sP_change, sP_max;
	int info, iter, isum;
	double rcond, ferr, berr, rpivot;
	char equed[3];
    cpl_matrix *Aij_cpl, *bj_cpl;
    double *Aij, *bj, *sP, *sL, *ycen; // raw data of cpl vec and matrices

	ny=osample*(nrows+1)+1; /* The size of the sf array */
    if ( ny != (int)cpl_vector_get_size(sL_cpl) ) {
        cpl_msg_error(__func__, "Size for sL does not match! %d %d",ny,(int)cpl_vector_get_size(sL_cpl));
    }
    step=1.e0/osample;
    double omega[ny][nrows][ncols];
    double im[nrows][ncols];
    int mask[nrows][ncols];
    double model[nrows][ncols];
    double E[ncols];
    double Adiag[ncols];
    double sP_old[ncols];

    Aij_cpl = cpl_matrix_new(ny, ny);
    Aij = cpl_matrix_get_data(Aij_cpl);
    bj_cpl = cpl_matrix_new(ny, 1);
    bj = cpl_matrix_get_data(bj_cpl);
    memcpy(im, cpl_image_get_data(im_cpl), sizeof(im));
    sL = cpl_vector_get_data(sL_cpl);
    sP = cpl_vector_get_data(sP_cpl);
    ycen = cpl_vector_get_data(ycen_cpl);

/*
reconstruct "mask" which is the inverse of the bad-pixel-mask attached to the image
*/
    memcpy( mask, cpl_mask_get_data(cpl_image_get_bpm(im)), sizeof(mask) );

/*
   Construct the omega tensor. Normally it has the dimensionality of ny*nrows*ncols. 
   The tensor is mostly empty and can be easily compressed to ny*nx, but this will
   complicate matrix operations at later stages. I will keep it as it is for now.
   Note, that omega is used in in the equations for sL, sP and for the model but it
   does not involve the data, only the geometry. Thus it can be pre-computed once.
*/

	for(x=0; x<ncols; x++)
	{
		iy2=(1.e0-ycen[x])*osample; /* The initial offset should be reconsidered. It looks fine but needs theory. */
		iy1=iy2-osample;
		
		if(iy2==0) d1=step;
		else if(iy1==0) d1=0.e0;
		else d1=fmod(ycen[x], step);
		d2=step-d1;
		for(y=0; y<nrows; y++)
		{
			iy1+=osample;
			iy2+=osample;
			for(iy=0; iy<ny; iy++)
			{
				if(iy<iy1) omega[iy][y][x]=0.;
				else if(iy==iy1) omega[iy][y][x]=d1;
				else if(iy>iy1 && iy<iy2) omega[iy][y][x]=step;
				else if(iy==iy2) omega[iy][y][x]=d2;
				else omega[iy][y][x]=0.;
			}
		}
	}

/* Loop through sL , sP reconstruction until convergence is reached */
	iter=0;
    do
    {
/*
  Compute slit function sL
*/

/* Fill in SLE arrays */

    	diag_tot=0.e0;
        for(iy=0; iy<ny; iy++)
        {
            bj[iy]=0.e0;
            for(jy=0; jy<ny; jy++)
            {
           	    Aij[iy+ny*jy]=0.e0;
           	    for(x=0; x<ncols; x++)
           	    {
          	 	   sum=0.e0;
                   for(y=0; y<nrows; y++) sum+=omega[iy][y][x]*omega[jy][y][x]*mask[y][x];
                   Aij[iy+ny*jy]+=sum*sP[x]*sP[x];
                }
            }
            for(x=0; x<ncols; x++)
           	{
           		sum=0.e0;
                for(y=0; y<nrows; y++) sum+=omega[iy][y][x]*mask[y][x]*im[y][x];
                bj[iy]+=sum*sP[x];
            }
            diag_tot+=Aij[iy+ny*iy];
        }

/* Scale regularization parameters */

	    lambda=lambda_sL*diag_tot/ny;

/* Add regularization parts for the slit function */

        for(iy=1; iy<ny-1; iy++)
        {
        	Aij[iy+ny*(iy-1)]-=lambda;      /* Upper diagonal */
        	Aij[iy+ny*iy    ]+=lambda*2.e0; /* Main diagonal  */
        	Aij[iy-1+ny*iy  ]-=lambda;      /* Lower diagonal */
    
        }
      	Aij[ny-1+ny*(ny-2)]-=lambda;
       	Aij[0             ]+=lambda;
       	Aij[ny-1+ny*(ny-1)]+=lambda;
      	Aij[ny-2+ny*(ny-1)]-=lambda;

/* Solve the system of equations */

        sL_cpl = cpl_matrix_solve(Aij_cpl, bj_cpl);
        sL = cpl_matrix_get_data(sL_cpl);
        cpl_matrix_delete(sL_cpl);

/* Normalize the slit function */

        norm=0.e0;
        for(iy=0; iy<ny; iy++) norm+=sL[iy];
        norm/=osample;
        for(iy=0; iy<ny; iy++) sL[iy]/=norm;

/*
  Compute spectrum sP
*/
        for(x=0; x<ncols; x++)
        {
        	E[x]=0.e0;
        	for(y=0; y<nrows; y++)
            {
            	sum=0.e0;
        	    for(iy=0; iy<ny; iy++)
        	    {
                    sum+=omega[iy][y][x]*sL[iy];
        	    }
                E[x]+=sum*im[y][x]*mask[y][x];
            }
        }
    
        if(lambda_sP>0.e0)
        {
        	norm=0.e0;
        	for(x=0; x<ncols; x++)
        	{
        		sP_old[x]=sP[x];
        		norm+=sP[x];
        	}
        	norm/=ncols;
        	lambda=lambda_sP*norm;
        	Adiag[0] =-lambda;
        	for(x=1; x<ncols-1; x++)
        	{
        		Adiag[x]=-lambda;
        	}

            for(x=0; x<ncols; x++) sP[x]=E[x];
            /* Solver */
            cpl_matrix_solve()
        }	
        else
        {
        	for(x=0; x<ncols; x++)
        	{
        	    sP_old[x]=sP[x];
                sP[x]=E[x]/Adiag[x+ncols];	
        	} 
        }

/* Compute the model */

  	    for(y=0; y<nrows; y++)
  	    {
            for(x=0; x<ncols; x++)
            {
        	    sum=0.e0;
        	    for(iy=0; iy<ny; iy++) sum+=omega[iy][y][x]*sL[iy];
        	    model[y][x]=sum*sP[x];
            }
        }

/* Compare model and data */

        sum=0.e0;
        isum=0;
        for(y=0; y<nrows; y++)
        {
        	for(x=0;x<ncols; x++)
        	{
                sum+=mask[y][x]*(model[y][x]-im[y][x])*(model[y][x]-im[y][x]);
                isum+=mask[y][x];
        	}
        }
        dev=sqrt(sum/isum);

/* Adjust the mask marking outlyers */

        for(y=0; y<nrows; y++)
        {
        	for(x=0;x<ncols; x++)
        	{
                if(fabs(model[y][x]-im[y][x])>6.*dev) mask[y][x]=0; else mask[y][x]=1;
        	}
        }


/* Compute the change in the spectrum */

        sP_change=0.e0;
        sP_max=1.e0;
        for(x=0; x<ncols; x++)
        {
            if(sP[x]>sP_max) sP_max=sP[x];
            if(fabs(sP[x]-sP_old[x])>sP_change) sP_change=fabs(sP[x]-sP_old[x]);
        }

/* Check the convergence */

    } while(iter++ < maxiter && sP_change > sP_stop*sP_max);


    cpl_matrix_unwrap(Aij_cpl);


    return cpl_image_wrap_double(ncols, nrows, model);
}

#define NCOLS 768
#define NROWS  15
#define NY    161

int main(int nArgs, void *Args[])
{
    int ncols, nrows, osample, ny, i, j, k;
    FILE *datafile;
    double norm;
    static byte mask_data[NROWS][NCOLS];
    static double im_data[NROWS][NCOLS], ycen_data[NCOLS];

    datafile=fopen("slit_func1.dat", "rb");
    fread(&osample, sizeof(int), 1, datafile);
    fread(&ncols, sizeof(int), 1, datafile);
    fread(&nrows, sizeof(int), 1, datafile);
    fread(&ny, sizeof(int), 1, datafile);
    printf("%d %d %d\n", osample,ncols,nrows);

    fread(mask_data, sizeof(byte), nrows*ncols, datafile);
    fread(im_data, sizeof(double), nrows*ncols, datafile);
    fread(ycen_data, sizeof(double), ncols, datafile);
    printf("nrows=%d, ncols=%d, double=%d\n", nrows, ncols, sizeof(double));
    fclose(datafile);

    cpl_init(CPL_INIT_DEFAULT);
    cpl_image *model, *im, *tmp;
    cpl_vector *ycen, *sL, *sP;
    cpl_mask *mask;

    model = cpl_image_new(ncols, nrows, CPL_TYPE_DOUBLE);
    im = cpl_image_wrap_double(ncols, nrows, im_data);
    ycen = cpl_vector_wrap(ncols, ycen_data);
    sL = cpl_vector_new(ny);
    mask = cpl_mask_new(ncols, nrows);
    for (i=0;i < nrows;i++) { // convert to CPL mask, _1 means masked, inverse to mask_data!
        for (j=0;j < ncols;j++) {
            if (mask_data[i][j] == 1) {
                cpl_mask_set(mask, i, j, CPL_BINARY_0);
                } else {
                cpl_mask_set(mask, i, j, CPL_BINARY_1);
                }
        }
    }

    
    if (cpl_image_set_bpm(im, mask) != NULL) { return 1; }
    tmp = cpl_image_collapse_median_create(im , 0, 0, 0);
    sP = cpl_vector_new_from_image_row(tmp,1);
    cpl_image_delete(tmp);

    for (i=0;i < ny;i++) {
        printf("%lf, ",cpl_vector_get(sP,i));
        }
   

    model = slit_func_vert(ncols, nrows, osample, im, ycen, 
                        sL, sP, 
                        0.e-6, 1.e0, 1.e-5, 20);
 
    return 0;

    datafile=fopen("dump.bin", "wb");
    fwrite(sL, sizeof(double), ny, datafile);
    fwrite(sP, sizeof(double), ncols, datafile);
    fwrite(sP, sizeof(double), ncols, datafile);
    fwrite(im, sizeof(double), ncols*nrows, datafile);
    fwrite(model, sizeof(double), ncols*nrows, datafile);
    fclose(datafile);

    return 0;
}
