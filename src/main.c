#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/nn.h"
#include "../include/common.h"

#define TEST_TRAIN_SPLIT .75
#define BATCH_SIZE 32


int main(void)
{
    NeuralNet nn = createNetwork(.001f, 784, BATCH_SIZE);
    nn.add_linear_layer("relu", 512);
    nn.add_linear_layer("relu", 256);
    nn.add_linear_layer("softmax", 10);
    char file[] = "../data/train.csv";

    float *y_hat = calloc(10*BATCH_SIZE, sizeof(float));
    float *in = calloc(784*BATCH_SIZE, sizeof(float));
    float *y = calloc(10*BATCH_SIZE, sizeof(float));
    struct mnist mnist_data = load_mnist(file, TEST_TRAIN_SPLIT);

    
    for(int iteration=0;iteration<600;iteration++)
    {
        printf("Iteration: %d\n",iteration);
        float error=0;
        for(int i=0;i<floor(1000/BATCH_SIZE);i++)
        {
            memmove(in, (mnist_data.train_data+(i*784*BATCH_SIZE)), sizeof(float)*784*BATCH_SIZE);
            memmove(y, (mnist_data.train_labels+(i*10*BATCH_SIZE)), sizeof(float)*10*BATCH_SIZE);
            nn.forward_pass(in, y_hat);
            error += calc_batch_error(y, y_hat, 10, BATCH_SIZE);
            nn.backward_pass(y);
        }
        if(iteration%10==9){
            printf("Error: %f\n", error);
            printf("Correct Count: %d/%d\n", validation_run( mnist_data.test_data, mnist_data.test_labels, mnist_data.len_test, BATCH_SIZE, nn), (int)(floor(1000/BATCH_SIZE)*BATCH_SIZE));}
    }

    nn.clean_up();
    free_all(in, y_hat, y, mnist_data.test_data, mnist_data.test_labels, mnist_data.train_data, mnist_data.train_labels);
    return 0;
}