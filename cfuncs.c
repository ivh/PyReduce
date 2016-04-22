#include <stdio.h>
#include <stdlib.h>

void copy(float *in, float *out, int len) {
    int i=0;
    for (i=0; i<len; i++) {
        out[i] = in[i];
    }
}
