#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/nn.h"

float xin[4][3] = {
                        {1,0,1},
                        {0,1,1},
                        {0,0,1},
                        {1,1,1}
                    };
float y[4] = {1, 1, 0, 0};

int main(void)
{
    float *in = malloc(sizeof(float)*3);
    for(int i=0;i<5;i++) in[i]=i*2.0;
    nn_linear("relu", 3, 5);
    nn_linear("relu", 5, 1);
    for(int iteration=0;iteration<60;iteration++)
    {
        float error=0;
        float *yhat=0;
        for(int i=0;i<4;i++)
        {
            for(int j=0;j<3;j++) in[j]=xin[i][j]; 
            nn_forward(in, yhat);
            error += pow(*yhat-y[i],2);
            nn_backward(yhat);
        }
        if(iteration%9==9) printf("Error: %f", error);
    }
    nn_delete();
    free(in);
    return 0;
}