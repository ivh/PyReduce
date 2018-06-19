
//int  locate_clusters(int argc, void *argv[]); 

int locate_clusters(int nX, int nY, int filter, int * im, int nmax, int *x, int *y, float noise, int * mask);
int cluster(int *x, int *y, int n, int nX, int nY, int thres, int *index);
