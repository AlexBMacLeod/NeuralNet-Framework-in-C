#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#include "../include/common.h"

void checkLen(char file[], int *len)
{
    int rows=0;
    FILE *in = fopen(file, "r");
    if (!in) {
        fprintf(stderr, "Could not open file: %s\n", file);
        exit(1);
    }else{

        char line[8192];
        while (!feof(in) && fgets(line, 8192, in)) {
            ++rows;
        }
        printf("Loading %d rows from %s\n", rows, file);
        *len = rows;
        fclose(in);
    }
}

void load_data(char file[], float* data, int* labels)
{
    FILE *in = fopen(file, "r");
    if (!in) {
        fprintf(stderr, "Could not open file: %s\n", file);
        exit(1);
    }else{
        char line[8192];
        char* token = NULL;
        long index = 0;
    
        int row = 0;
        long rowInner = 0;
        int column = 0;
        int colInner = 0;
    
        while (!feof(in) && fgets(line, 8192, in)) {
            row++;
            if (row == 1)
                continue;
            token = strtok(line, ",");
            column = 0;
            colInner = 0;
            while (token) {
                if(column==0){ 
                    labels[rowInner]=atoi(token);
                }else{
                    index = rowInner*784+colInner;
                    data[index] = atof(token)/255.0f;
                }
                column++;
                colInner++;
                token = strtok(NULL, ",");
            }
            rowInner++;
        }
    fclose(in);
    }
}

void splitLabels(float *data, float *training_data, int *labels, int len)
{
    int y = 0;
    long index, indexOne, indexTwo;
    for(int i=0; i<len; i++)
    {
        y=0;
        for(int j=0; j<785; j++)
        {
            if(j==0){
                index = 785*i;
                labels[i] = (int)data[index];
            }else{
                indexOne = 785*i+j;
                indexTwo = 784*i+y;
                training_data[indexTwo] = data[indexOne];
                y++;
            }
        }
    }
}

void one_hot_encoder(int *data, float *one_hot_encoded, int len)
{
    for(int i=0; i<len; i++)
    {
        one_hot_encoded[i*10+data[i]] = 1.0f;
    }
}

int argmax(float *y_hat, float *y, int len)
{
    int y_hat_col = 0; 
    int y_col = 0;
    for(int i=0; i<len; i++){
        if(y_hat[i]>y_hat[y_hat_col]) y_hat_col=i;
        if(y[i]>y[y_col]) y_col = i;
    }
    if(y_hat_col==y_col){ return 1;
    }else return 0;
}

void test_train_split(float *data, float* labels, float *train, float *test, float *train_labels, float *test_labels, int len, int train_size, int test_size)
{
    long indexOne = train_size*784;
    long indexTwo = train_size*10;
    memmove(train, data, sizeof(float)*train_size*784);
    memmove(test, data+indexOne, sizeof(float)*test_size*784);//<-here
    memmove(train_labels, labels, sizeof(float)*train_size*10);
    memmove(test_labels, labels+indexTwo, sizeof(float)*test_size*10);
}

struct mnist load_mnist(char file[], float testTrainSplit)
{
    struct mnist out;
    float *data, *labels, *train, *test, *train_labels, *test_labels;
    int *labels_raw;
    int test_size, train_size, len=0;
    checkLen(file, &len);
    test_size = len*(1-testTrainSplit);
    train_size = len*testTrainSplit;
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
    out.len_test=test_size;
    out.len_train=train_size;
    out.test_data=test;
    out.test_labels=test_labels;
    out.train_data=train;
    out.train_labels=train_labels;
    free_all(labels_raw, data, labels);
    return out;
}

float calc_batch_error(float *y, float *y_hat, int out_size, int batch_size)
{
    float error = 0.0f;
    for(int i=0; i<batch_size; i++)
    {
        for(int j=0; j<out_size; j++) error += pow(y_hat[i*out_size+j]-y[i*out_size+j],2);
    }
    return error;
}

int argmax_batch(float *y_hat, float *y, int len, int batch_size)
{
    int y_col, y_hat_col, correct = 0;
    for(int i=0; i<batch_size; i++)
    {
        y_col=0;y_hat_col=0;
        for(int j=0; j<len; j++) 
        {
            if(y_hat[i*len+j]>y_hat[i*len+y_hat_col]) y_hat_col=j;
            if(y[i*len+j]>y[i*len+y_col]) y_col = j;
        }
    if(y_hat_col==y_col) correct++;
    }
    return correct;
}