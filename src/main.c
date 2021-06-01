#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/nn.h"

float xin[4][3] =   {
                        {1,0,1},
                        {0,1,1},
                        {0,0,1},
                        {1,1,1}
                    };
float y[4] = {1, 1, 0, 0};

int main(void)
{
    float *in = malloc(sizeof(float)*3);
    float *yhat = malloc(sizeof(float));
    nn_linear("relu", 3, 5);
    nn_linear("none", 5, 1);
    for(int iteration=0;iteration<60;iteration++)
    {
        float error=0;
        *yhat = 0;
        for(int i=0;i<4;i++)
        {
            for(int j=0;j<3;j++) in[j]=xin[i][j]; 
            nn_forward(in, yhat);
            error += pow(*yhat-y[i],2);
            //nn_backward(yhat);
        }
        if(iteration%10==9) printf("Error: %f\n", error);
    }
    nn_delete();
    free(in);
    free(yhat);
    return 0;
}