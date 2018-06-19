#include <stdio.h>
#include <stdlib.h>


#define SWAP(r,s)  do{int t=r; r=s; s=t; } while(0)

void siftDown(int *a, int *i, int start, int end)
{
  int root = start;

  while(root*2+1 < end)
  {
    int child = 2*root + 1;
    if((child+1 < end) && (a[child] < a[child+1]))
    {
      child++;
    }
    if(a[root] < a[child])
    {
      SWAP(a[child], a[root]);
      SWAP(i[child], i[root]);
      root = child;
    }
    else return;
  }
}

void isort(int *a, int *i, int count)
{
  int start, end;

  for(start=0; start<count; start++) i[start]=start;

  for(start=(count-2)/2; start>=0; start--)
  {
    siftDown(a, i, start, count);
  }

  for (end=count-1; end > 0; end--)
  {
    SWAP(a[end], a[0]);
    SWAP(i[end], i[0]);
    siftDown(a, i, 0, end);
  }
}

int *diag_sort(int *x, int *y, int *index, int n, int nX, int nY)
{
  int i, diag;

  if(nX<=nY)
  {
    for(i=0;i<n;i++)
    {
      diag=x[i]+y[i]+1;
      if(diag<nX-1)
      {
        index[i]=diag*(diag+1)/2-x[i]-1;
      }
      else if(diag>=nX-1 && diag<nY)
      {
        index[i]=nX*(nX-1)/2+(diag-nX)*nX+nX-x[i]-1;
      }
      else if(diag>=nY)
      {
        index[i]=nX*nY-(nX+nY-diag)*(nX+nY-diag+1)/2+nX-x[i]-1;
      }
    }
  }
  else
  {
    for(i=0;i<n;i++)
    {
      diag=x[i]+y[i]+1;
      if(diag<nY)
      {
        index[i]=diag*(diag+1)/2-x[i]-1;
      }
      else if(diag>=nY && diag<=nX)
      {
        index[i]=nY*(nY-1)/2+(diag-nY)*nY+y[i];
      }
      else if(diag>nX)
      {
        index[i]=nX*nY-(nX+nY-diag)*(nX+nY-diag+1)/2+nX-x[i]-1;
      }
    }
  }
  return index;
}

int  locate_clusters(int argc, void *argv[])
{
/*
 locate_clusters takes a 2D integer array with horizontal structures
 (spectral orders) in it  and returns X and Y coodinates of pixels
 "with signal" in two integer arrays. Pixel selection is based on a
 simple comparizon of a box-car filtered image with the original. The
 filtering (smoothing) is done in vertical direction. Pixels in the
 original image that have higher counts than the smoothed image are
 considered to have "signal". An optional "noise" parameter is used to
 control the selection threshold thus allowing for more robust
 discrimination. On exit locate_clusters returns the number of pixels
 "with signal" (int) or negative number indicating an error condition.
 This function has 7 mandatory and two optional
 parameters passed by pointers. The mandatory parameters are:
   nX     - (input, int) number of columns in the input array;
   nY     - (input, int) number of rows in the input array;
   filter - (input, int) filter window. Filtered value in pixel [x,y] is
            1/filter * sum(i=x-filter/2,x+filter/2) Image[x,i];
   im     - (input, *int) image;
   nmax   - (input, int) size of the allocated arrays for pixels with signal;
   x      - (ouput, *int) array that will hold X-coordinates of pixels with
            signal. This array must be allocated prior to calling locate
            clusters. The allocated size is passed
   y      - (ouput, *int) array that will hold Y-coordinates of pixels with
            signal;

 Optional parameters are:
   noise  - (input, float) extra margin for selecting pixels with signal.
            The selection condition is thus: im[x,y] > im_smoothed[x,y]+noise.
            The default value is 1;
   mask   - (input, *unsigned char) mask of bad pixels. 0 - bad pixel, good
            otherwise.

 Returning negative number indicates an error:
   -1     - not all mandatory arguments were specified;
   -2     -

 WARNING: locate_clusters do not check the consitency between the actual size
          of the arrays and their given dimensions.

  A typical call from IDL may look like this:

  nx=size(im[*,0])
  ny=size(im[0,*])
  nmax=nx*ny
  x=lonarr(nmax)
  y=lonarr(nmax)
  filter=25L
  noise=8.
  n=call_external(<DLL name>,'locate_clusters',nx,ny,filter $
                  ,long(im),nmax,x,y,noise)

 History: 17-Jul-2000 N.Piskunov wrote the IDL version.
          02-Jan-2014 N.Piskunov ported this routine to C.
*/
  int nX, nY, *im, *x, *y, filter, nmax, n;
  unsigned char *mask;
  float noise;
  int iX, iY, half, has_mask;
  float offset, box, nbox;

  if(argc<7) return -1;
  nX    =*(int *)argv[0];
  nY    =*(int *)argv[1];
  filter=*(int *)argv[2];
  im    = (int *)argv[3];
  nmax  =*(int *)argv[4];
  x     = (int *)argv[5];
  y     = (int *)argv[6];
  noise = (argc>7)?*(float *)argv[7]:1.;
  if(argc>8)
  {
    mask  = (unsigned char *)argv[8];
    has_mask=1;
  }
  else has_mask=0;

  n=0;
  half=filter/2;
  if(has_mask)
  {
    for(iX=0; iX<nX; iX++)
    {
      box=0.; nbox=0;
      for(iY=0; iY<half; iY++) {box+=*(im+iY*nX+iX); nbox++;}
      for(iY=0; iY<nY; iY++)
      {
        if(iY+half<nY) {box+=*(im+(iY+half)*nX+iX); nbox++;}
        if(iY-half>=0) {box-=*(im+(iY-half)*nX+iX); nbox--;}
        if(*(im+iY*nX+iX)>box/nbox+noise && *(mask+iY*nX+iX))
        {
          if(n==nmax) return -2;
          x[n]=iX; y[n]=iY; n++;
        }
      }
    }
  }
  else
  {
    for(iX=0; iX<nX; iX++)
    {
      box=0.; nbox=0;
      for(iY=0; iY<half; iY++) {box+=*(im+iY*nX+iX); nbox++;}
      for(iY=0; iY<nY; iY++)
      {
        if(iY+half<nY) {box+=*(im+(iY+half)*nX+iX); nbox++;}
        if(iY-half>=0) {box-=*(im+(iY-half)*nX+iX); nbox--;}
        if(*(im+iY*nX+iX)>box/nbox+noise)
        {
          if(n==nmax) return -2;
          x[n]=iX; y[n]=iY; n++;
        }
      }
    }
  }
  return n;
}

// Original call
//int  cluster(int argc, void *argv[])
int cluster(int *x, int *y, int n, int nX, int nY, int thres, int *index)
{
/*
 Cluster takes two 1D integer arrays with X and Y coodinates of
 pixels somehow selected in a 2D (Nx,Ny) array and identifies
 clusters of pixels. X and Y should be Y-sorted, that is X should run faster
 while Y should increase monotonously. If that is not the case, it could be
 achieved with the following commands:
 i=sort(Y*2L^15+X) & Y=Y(i) & X=X(i)
 A cluster is defined so that every pixel in a cluster is adjacent to at
 list one other pixel from the same cluster (their X and Y coordinates do
 not differ by more then 1) and all the pixels that are adjacent to any
 pixel in a cluster belong ; to the same cluster.
 The function returns a 1D integer array of the same size as X and Y
 that contains cluster number for each pixel. Non-cluster members
 are marked with 0.
 Other parameters:
   thres    - (input) consider clusters with "threshold"
              or smaller number of members to be non-clustered (index eq 0)
   index    - (output) integer array of the same size as x and y containing
              cluster number that contains pixel with coordinates [x, y]
   nregions - (output) contain the number of identified clusters also
              counting non-cluster pixels (if any) as a separate cluster

 History: 17-July-2000 N.Piskunov wrote the version optimized for
                       clusters oriented preferentially along rows or
                       columns.
          21-July-2000 N.Piskunov modified to handle arbitrary oriented
                       clusters in the optimal way. It is slower than the
                       original version by about 10% for spectral orders
                       which are nearly horizontal/vertical.
*/

//  int *x, *y, n, nX, nY, thres, *index;
  int *Xsort, *i2X, *X2i, *Ysort, *i2Y, *Y2i,
      *Lsort, *i2L, *L2i, *Rsort, *i2R, *R2i,
      *dummy, *dummy1;
  int i, j[9], jj, nj, njj, j1, j2, iX, iY, n_branches;
  int threshold, nregions;
  int min_clr, max_clr, clrs[9], *uniq_clr, *translation;


  //if(argc<7) return -1;
  //x=(int *)argv[0];
  //y=(int *)argv[1];
  //n=*(int *)argv[2];
  if(n<=0) return -2;
  //nX=*(int *)argv[3];
  //nY=*(int *)argv[4];
  if(n>nX*nY) return -4;
  //thres=*(int *)argv[5];
  //index=(int *)argv[6];

  dummy =(int *)malloc(n*sizeof(int));
  dummy1=(int *)malloc(n*sizeof(int));

  i2X   =(int *)malloc(n*sizeof(int));
  X2i   =(int *)malloc(n*sizeof(int));

  i2Y   =(int *)malloc(n*sizeof(int));
  Y2i   =(int *)malloc(n*sizeof(int));

  i2L   =(int *)malloc(n*sizeof(int));
  L2i   =(int *)malloc(n*sizeof(int));

  i2R   =(int *)malloc(n*sizeof(int));
  R2i   =(int *)malloc(n*sizeof(int));

  for(i=0; i<n; i++) index[i]=0;

/*
   The initial X and Y are Y-sorted, which
   means that X runs faster while Y is
   increasing monotonously.
*/

  for(i=0; i<n; i++) dummy[i]=x[i]+y[i]*nX;
  isort(dummy, X2i, n);                  /* X-sorted indices for each i     */
  for(i=0;i<n;i++) i2X[X2i[i]]=i;        /* i-value for each X-sorted index */

  for(i=0; i<n; i++) dummy[i]=y[i]+x[i]*nY;
  isort(dummy, Y2i, n);                  /* Y-sorted indices for each i     */
  for(i=0;i<n;i++) i2Y[Y2i[i]]=i;        /* i-value for each Y-sorted index */

  diag_sort(x, y, dummy, n, nX, nY);
  isort(dummy, L2i, n);                  /* Left-diag-sorted indices for each i     */
  for(i=0; i<n; i++) i2L[L2i[i]]=i;      /* i-value for each left-diag-sorted index */

  for(i=0; i<n; i++) dummy1[i]=nX-1-x[i];
  diag_sort(dummy1, y, dummy, n, nX, nY);
  isort(dummy, R2i, n);                  /* Right-diag-sorted indices for each i     */
  for(i=0; i<n; i++) i2R[R2i[i]]=i;      /* i-value for each right-diag-sorted index */

  free(dummy);
  free(dummy1);

  translation=(int *)calloc(n+1, sizeof(int)); /* Color look-up table */

  n_branches=0;                          /* Branch counter */

  jj=0;
  for(i=0;i<n;i++)                       /* Loop through pixels */
  {
    nj=1;
    j[0]= i;                             /* Mark the current pixel i    */

    j1=i2X[i];                           /* X-sorted number of pixel i  */
    j2=j1-1;                             /* Previous X-sorted number    */
    if(j2>=0 && x[X2i[j2]]+1==x[i] && y[X2i[j2]]==y[i]) j[nj++]=X2i[j2];
    j2=j1+1;                             /* Next X-sorted number        */
    if(j2<n  && x[X2i[j2]]-1==x[i] && y[X2i[j2]]==y[i]) j[nj++]=X2i[j2];

    j1=i2Y[i];                           /* X-sorted number of pixel i  */
    j2=j1-1;                             /* Previous X-sorted number    */
    if(j2>=0 && x[Y2i[j2]]==x[i] && y[Y2i[j2]]+1==y[i]) j[nj++]=Y2i[j2];
    j2=j1+1;                             /* Next X-sorted number        */
    if(j2<n  && x[Y2i[j2]]==x[i] && y[Y2i[j2]]-1==y[i]) j[nj++]=Y2i[j2];

    j1=i2L[i];                           /* X-sorted number of pixel i  */
    j2=j1-1;                             /* Previous X-sorted number    */
    if(j2>=0 && x[L2i[j2]]-1==x[i] && y[L2i[j2]]+1==y[i]) j[nj++]=L2i[j2];
    j2=j1+1;                             /* Next X-sorted number        */
    if(j2<n  && x[L2i[j2]]+1==x[i] && y[L2i[j2]]-1==y[i]) j[nj++]=L2i[j2];

    j1=i2R[i];                           /* X-sorted number of pixel i  */
    j2=j1-1;                             /* Previous X-sorted number    */
    if(j2>=0 && x[R2i[j2]]+1==x[i] && y[R2i[j2]]+1==y[i]) j[nj++]=R2i[j2];
    j2=j1+1;                             /* Next X-sorted number        */
    if(j2<n  && x[R2i[j2]]-1==x[i] && y[R2i[j2]]-1==y[i]) j[nj++]=R2i[j2];

    njj=0;
    min_clr=n+1;                         /* Initialize minimum color     */
    for(j1=0; j1<nj; j1++)               /* Find minimum color           */
    {                                    /* any existing cluster member  */
      j2=index[j[j1]];                   /* Color of this pixel          */
      if(j2 > 0 && j2 < min_clr) min_clr=j2;
    }

	if(min_clr == n+1)                   /* None found (only uncolored pixels)  */
    {
      n_branches++;
      for(j1=0; j1<nj; j1++) index[j[j1]]=n_branches;
      translation[n_branches]=n_branches;
    }
    else                                  /* Found colored pixels, re-paint all  */
    {                                     /* pixels with smallest non-zero color */
      for(j1=0;j1<nj;j1++)
      {
        j2=index[j[j1]];                  /* Color of this pixel          */
        if(j2>min_clr) translation[j2]=min_clr; /* Adjust the look-up table     */
        index[j[j1]]=min_clr;             /* Re-color this pixel          */
      }
    }
  }
  free(X2i); free(i2X);
  free(Y2i); free(i2Y);
  free(L2i); free(i2L);
  free(R2i); free(i2R);

  for(i=0; i<n; i++)                      /* Reduce reference chains in look-up  */
  {                                       /* table to single direct references   */
    translation[i]=translation[translation[i]];
  }

  for(i=0; i<n; i++)                      /* Apply look-up table                 */
  {
    index[i]=translation[index[i]];
  }

  threshold=thres>1?thres:1;              /* Prepare for measuring cluster sizes */

  max_clr=0;                              /* Find maximum color index            */
  for(i=0;i<n;i++) if(index[i]>max_clr) max_clr=index[i];
  max_clr++;

  uniq_clr=(int *)calloc(max_clr, sizeof(int));

  for(i=0;i<n;i++) uniq_clr[index[i]]++;

  j1=0;
  translation[0]=0;
  for(i=1;i<max_clr;i++)
  {
    if(uniq_clr[i]>=threshold)
    {
      j1++;
      translation[i]=j1;
    }
    else translation[i]=0;
  }

  nregions=j1;
  free(uniq_clr);

  for(i=0;i<n;i++) index[i]=translation[index[i]];

  free(translation);

  return nregions;
}
