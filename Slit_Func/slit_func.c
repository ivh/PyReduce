#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cpl.h>

typedef unsigned char byte;
#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

/*
int slit_func(int ncols, int nrows, int Osample, double image[nrows][ncols], byte mask[nrows][ncols],
              double sL[(nrows+1)*Osample+1],  double sP[ncols], float im_model[nrows][ncols],
              double omega[][], double oo[][], double Ajxjy[][], float Biy[],
              double Dd[ncols])
{

}
*/

int slit_func_vert(int ncols,                     /* Swath width in pixels                                 */ 
                   int nrows,                     /* Extraction slit height in pixels                      */
                   int ny,                        /* Size of the slit function array: ny=osample(nrows+1)+1*/
                   double im[nrows][ncols],       /* Image to be decomposed                                */
                   byte mask[nrows][ncols],       /* Initial and final mask for the swath                  */
                   double ycen[ncols],            /* Order centre line offset from pixel row boundary      */
                   int osample,                   /* Subpixel ovsersampling factor                         */ 
                   double lambda_sP,              /* Smoothing parameter for the spectrum, coiuld be zero  */
                   double lambda_sL,              /* Smoothing parameter for the slit function, usually >0 */
                   double sP[ncols],              /* Spectrum resulting from decomposition                 */
                   double sL[ny],                 /* Slit function resulting from decomposition            */
                   double model[nrows][ncols],    /* Model constructed from sp and sf                      */
                   double omega[ny][nrows][ncols],/* Work array telling what fruction of subpixel iy falls */
                                                  /* into pixel {x,y}.                                     */
                   double sP_old[ncols],          /* Work array to control the convergence                 */
                   double Aij[],                  /*                          */
                   double Aij_work[],             /* ny*ny                                                 */
                   double bj[],                   /* ny                                                    */
                   int    ipivot[],               /* ny                                                    */
                   double r[],                    /* ny                                                    */
                   double c[],                    /* ny                                                    */
                   double Adiag[],                /* Arrays for solving the tridiagonal SLE for sP (ncols) */
                   double Bdiag[],                /* main diagonal (ncols)                                 */
                   double Cdiag[],                /* lower diagonal (ncols)                                */
                   double E[])                    /* RHS (ncols)                                           */
{
	int x, y, iy, jy, iy1, iy2;
	double step, d1, d2, sum, norm, dev, lambda, diag_tot, sP_change, sP_max;
	int info, iter, isum;
	double rcond, ferr, berr, rpivot;
	char equed[3];

	ny=osample*(nrows+1)+1; /* The size of the sf array */
    step=1.e0/osample;

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
                Aij_work[iy+ny*jy]=Aij[iy+ny*jy];
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

/* // LAPACK version to solve
        equed[0]='N'; equed[1]='\0';
        info=LAPACKE_dgesvx(LAPACK_COL_MAJOR, 'E', 'N', ny, 1, Aij, ny,
                            Aij_work, ny, ipivot, equed, r, c, bj, ny, sL, ny,
                            &rcond, &ferr, &berr, &rpivot);
        printf("info(sL)=%d\n", info);
*/


        int i;
        for (i=0;i < sizeof (sL);i++) {
            printf("%lf ",sL[i]);
            }
        printf("\n");

        cpl_matrix *Aij_cpl, *bj_cpl, *sL_cpl;
        Aij_cpl = cpl_matrix_wrap(ny, ny, Aij);
        bj_cpl = cpl_matrix_wrap(ny, 1, bj);
        sL_cpl = cpl_matrix_wrap(ny, 1, sL);

        sL_cpl = cpl_matrix_solve_normal(Aij_cpl, bj_cpl);
        sL = cpl_matrix_get_data(sL_cpl);
        for (i=0;i < sizeof (sL);i++) {
            printf("%lf",sL[i]);
            }
        cpl_matrix_unwrap(Aij_cpl);
        cpl_matrix_unwrap(bj_cpl);
        cpl_matrix_unwrap(sL_cpl);

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
        	Bdiag[x]=0.e0;
        	E[x]=0.e0;
        	for(y=0; y<nrows; y++)
            {
            	sum=0.e0;
        	    for(iy=0; iy<ny; iy++)
        	    {
                    sum+=omega[iy][y][x]*sL[iy];
        	    }
                Bdiag[x]+=sum*sum*mask[y][x];
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
        	Bdiag[0]+=lambda;
        	Cdiag[0] =-lambda;
        	for(x=1; x<ncols-1; x++)
        	{
        		Adiag[x]=-lambda;
        		Bdiag[x]+=2.e0*lambda;
        		Cdiag[x]=-lambda;
        	}
        	Bdiag[ncols-1]+=lambda;
/*        	info=LAPACKE_dpttrf(ncols, Bdiag, Cdiag);
            printf("info(sP1)=%d\n", info);
        	info=LAPACKE_dpttrs(LAPACK_COL_MAJOR, ncols, 1, Bdiag, Cdiag, E, ncols);
            printf("info(sP2)=%d\n", info);
    */
        	for(x=0; x<ncols; x++) sP[x]=E[x];
        }	
        else
        {
        	for(x=0; x<ncols; x++)
        	{
        	    sP_old[x]=sP[x];
                sP[x]=E[x]/Bdiag[x];	
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

    } while(iter++<20 && sP_change>1.e-5*sP_max);

	return 0;
}

#define NCOLS 768
#define NROWS  15
#define NY    161

int main(int nArgs, void *Args[])
{
  int ncols, nrows, osample, ny, iret, i, j, k;
  FILE *datafile;
  double norm;

  cpl_init(CPL_INIT_DEFAULT);

  datafile=fopen("slit_func1.dat", "rb");
  fread(&osample, sizeof(int), 1, datafile);
  fread(&ncols, sizeof(int), 1, datafile);
  fread(&nrows, sizeof(int), 1, datafile);
  fread(&ny, sizeof(int), 1, datafile);
  printf("%d %d %d %d\n", osample,ncols,nrows,ny);

  {
    static byte mask[NROWS][NCOLS];
    static double im[NROWS][NCOLS], ycen[NCOLS];
    static double omega[NY][NROWS][NCOLS], sP[NCOLS], sP_old[NCOLS], sL[NY], model[NROWS][NCOLS];

    static double Aij[NY*NY];            /* Various LAPACK arrays                                 */
    static double Aij_work[NY*NY];
    static double bj[NY];
    static int    ipivot[NY];
    static double r[NY];
    static double c[NY];
    static double Adiag[NCOLS-1], Bdiag[NCOLS], Cdiag[NCOLS-1], E[NCOLS];

    fread(mask, sizeof(byte), nrows*ncols, datafile);
    fread(im, sizeof(double), nrows*ncols, datafile);
    fread(ycen, sizeof(double), ncols, datafile);
    fclose(datafile);

    printf("nrows=%d, ncols=%d, double=%d\n", nrows, ncols, sizeof(double));

    for(i=0; i<ncols; i++)
    {
        sP[i]=0.e0;
        norm=0.e0;
        for(j=0; j<nrows; j++)
        {
        	norm+=mask[j][i];
        	sP[i]+=im[j][i]*mask[j][i];
        }
        norm/=nrows;
        if(norm>0) sP[i]/=norm; else sP[i]=-1000.; /* I should handle these columns eventually */
    }

    iret=slit_func_vert(ncols, nrows, ny, im, mask, ycen, osample, 0.e-6, 1.e0,
                        sP, sL, model, omega, sP_old, Aij, Aij_work, bj, ipivot, r, c,
                        Adiag, Bdiag, Cdiag, E);

    datafile=fopen("dump.bin", "wb");
    fwrite(sL, sizeof(double), ny, datafile);
    fwrite(sP, sizeof(double), ncols, datafile);
    fwrite(sP_old, sizeof(double), ncols, datafile);
    fwrite(im, sizeof(double), ncols*nrows, datafile);
    fwrite(model, sizeof(double), ncols*nrows, datafile);
    fclose(datafile);
  }
  return 0;
}
