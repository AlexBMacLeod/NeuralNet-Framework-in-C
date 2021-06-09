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
    float *in = (float*)malloc(sizeof(float)*3);
    float yhat=0;
    InsertAtHead("relu", 3, 4);
    //InsertAtHead("relu", 5, 5);
    InsertAtHead("none", 4, 1);
    for(int iteration=0;iteration<60;iteration++)
    {
        float error=0;
        for(int i=0;i<4;i++)
        {
            for(int j=0;j<3;j++) in[j]=xin[i][j]; 
            Forward(in, &yhat);
            error += pow(yhat-y[i],2);
            Backward(&y[i]);
        }
        if(iteration%10==9) printf("Error: %f\n", error);
    }
    //printf("%f", yhat);
    Delete();
    free(in);
    //free(yhat);
    return 0;
}