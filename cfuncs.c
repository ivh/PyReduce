#include <stdio.h>
#include <stdlib.h>
#include <cpl.h>

void copy(double *in, double *out, int len) {
    int i=0;
    for (i=0; i<len; i++) {
        out[i] = in[i];
    }
}

void double2vec(double * in, cpl_vector * out, int len) {
    int i=0;
    for (i=0; i<len; i++) {
        cpl_vector_set(out,i,in[i]);
    }
}

void vec2double(cpl_vector * in, double * out, int len) {
    int i=0;
    for (i=0; i<len; i++) {
        out[i] = cpl_vector_get(in,i);
    }
}
void medfilt(double *in, double *out, int len, int hw) {
    cpl_vector *vec;
    cpl_init(CPL_INIT_DEFAULT);
    //vec1 = cpl_vector_new(len);
    //double2vec(in,vec1,len);
    vec = cpl_vector_wrap(len,in);
    vec = cpl_vector_filter_median_create(vec, hw);
    out = cpl_vector_unwrap(vec);
    //vec2double(vec,out,len);
    //cpl_vector_delete(vec1);
    //cpl_vector_delete(vec2);

}
