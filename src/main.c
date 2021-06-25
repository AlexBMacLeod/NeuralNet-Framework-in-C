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

    float *data, *labels, *train, *test, *train_labels, *test_labels;
    int *labels_raw;
    int test_size, train_size, len=0;
    char file[] = "../data/train.csv";
    checkLen(file, &len);
    test_size = len*(1-TEST_TRAIN_SPLIT);
    train_size = len*TEST_TRAIN_SPLIT;
    data = calloc(len*784, sizeof(float));
    train = calloc(train_size*784, sizeof(float));
    test = calloc(test_size*784, sizeof(float));
    labels_raw = calloc(len, sizeof(int));
    labels = calloc(len*10, sizeof(float));
    train_labels = calloc(train_size*10, sizeof(float));
    test_labels = calloc(test_size*10, sizeof(float));
    load_data(file, data, labels_raw);
    one_hot_encoder(labels_raw, labels, len);
    test_train_split(data, labels, train, test, train_labels, test_labels, len, train_size, test_size);
    float *y_hat = calloc(10, sizeof(float));
    float *in = calloc(784, sizeof(float));
    float *y = calloc(10, sizeof(float));
    free_all(labels_raw, data, labels);


    
    for(int iteration=0;iteration<600;iteration++)
    {
        printf("Iteration: %d\n",iteration);
        float error=0;
        for(int i=0;i<1000;i++)
        {
            memmove(in, (train+(i*784)), sizeof(float)*784);
            memmove(y, (train_labels+(i*10)), sizeof(float)*10);
            nn.forward_pass(in, y_hat);
            for(int j=0;j<10;j++)error += pow(y_hat[j]-y[j],2);
            nn.backward_pass(y);
        }
        if(iteration%10==9){
            printf("Error: %f\n", error);
            printf("Correct Count: %d/1000\n", validation_run( test, test_labels, test_size, nn));}
    }

    nn.clean_up();
    free_all(in, y_hat, y, train_labels, test_labels, train, test);
    return 0;
}