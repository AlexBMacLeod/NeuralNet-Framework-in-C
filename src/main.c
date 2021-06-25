#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/nn.h"
#include "../include/common.h"

#define TEST_TRAIN_SPLIT .75


int main(void)
{
    NeuralNet nn = createNetwork(.001f, 784);
    nn.add_linear_layer("relu", 512);
    nn.add_linear_layer("relu", 256);
    nn.add_linear_layer("none", 10);
    char file[] = "../data/train.csv";

    float *y_hat = calloc(10, sizeof(float));
    float *in = calloc(784, sizeof(float));
    float *y = calloc(10, sizeof(float));
    struct mnist mnist_data = load_mnist(file, TEST_TRAIN_SPLIT);

    
    for(int iteration=0;iteration<600;iteration++)
    {
        printf("Iteration: %d\n",iteration);
        float error=0;
        for(int i=0;i<1000;i++)
        {
            memmove(in, (mnist_data.train_data+(i*784)), sizeof(float)*784);
            memmove(y, (mnist_data.train_labels+(i*10)), sizeof(float)*10);
            nn.forward_pass(in, y_hat);
            for(int j=0;j<10;j++)error += pow(y_hat[j]-y[j],2);
            nn.backward_pass(y);
        }
        if(iteration%10==9){
            printf("Error: %f\n", error);
            printf("Correct Count: %d/1000\n", validation_run( mnist_data.test_data, mnist_data.test_labels, mnist_data.len_test, nn));}
    }

    nn.clean_up();
    free_all(in, y_hat, y, mnist_data.test_data, mnist_data.test_labels, mnist_data.train_data, mnist_data.train_labels);
    return 0;
}