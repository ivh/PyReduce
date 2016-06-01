#include <stdio.h>
#include <stdlib.h>
#include <cpl.h>

void * memcpy ( void * destination, const void * source, size_t num );

void medfilt(double *in, double *out, int len, int hw) {
    double *dbl;
    cpl_vector *vec;
    cpl_init(CPL_INIT_DEFAULT);
    vec = cpl_vector_wrap(len,in);
    dbl = (double*)cpl_vector_unwrap( cpl_vector_filter_median_create(vec, hw) );
    memcpy(out,dbl,len*sizeof(double));
    free(dbl);
    //cpl_vector_delete(vec); //No, this nulls the input-array!
}
