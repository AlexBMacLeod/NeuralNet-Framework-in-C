#include <stdio.h>
#include <stdlib.h>
#include "../include/nn.h"


int main(void)
{
    float *in = malloc(sizeof(float)*5);
    //float *in = abc;
    for(int i=0;i<5;i++) in[i]=i*2.0;
    //net_add_layer("relu", 5, 5);
    //net_add_layer("relu", 5, 5);
    nn_linear("relu", 5, 5);
    nn_linear("relu", 5, 5);
    //nn_forward(in, in);
    nn_delete();
    //for(int i=0;i<5;i++) printf("%f\t", in[i]);
    free(in);
    return 0;
}