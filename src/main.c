#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/nn.h"
#include "../include/common.h"


int main(void)
{
    NeuralNet nn = createNetwork(.005f, 784);
    nn.add_linear_layer("relu", 512);
    nn.add_linear_layer("relu", 256);
    nn.add_linear_layer("none", 10);

    float *data_raw, *data, *labels;
    int *labels_raw;
    char file[] = "../data/train.csv";
    int len=0;
    checkLen(file, &len);
    //data_raw = calloc((len)*785, sizeof(float));
    data = calloc((len)*784, sizeof(float));
    labels_raw = calloc(len, sizeof(float));
    labels = calloc(len*10, sizeof(float));
    load_data(file, data, labels_raw);
    one_hot_encoder(labels_raw, labels, len);
    free_all(labels_raw);
    float *y_hat = calloc(10, sizeof(float));
    float *in = calloc(784, sizeof(float));
    float *y = calloc(10, sizeof(float));


    
    for(int iteration=0;iteration<600;iteration++)
    {
        printf("Iteration: %d\n",iteration);
        float error=0;
        for(int i=0;i<1000;i++)
        {
            memmove(in, (data+(i*784)), sizeof(float)*784); 
            nn.forward_pass(in, y_hat);
            memmove(y, (labels+(i*10)), sizeof(float)*10);
            for(int j=0;j<10;j++)error += pow(y_hat[j]-y[j],2);
            nn.backward_pass(y);
        }
        if(iteration%10==9) printf("Error: %f\n", error);
    }

    nn.clean_up();
    free_all(in, y_hat, y, data, labels);
    return 0;
}