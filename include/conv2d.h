#ifndef _CONV2D_H
#define _CONV2D_H
#include <stdbool.h>

#include "tensor.h"
#include "shape.h"



typedef struct conv2DLayer{
    Matrix *weights;
    Matrix *output;
    Matrix *deriv;
    Matrix *input;
    Matrix *delta;
    Matrix *nextDelta;
    Matrix *nextWeights;
    Matrix *kernels;
    float lr;
    bool padding;
    int kernel_size;
    int batch_size;
    int stride;
    struct Shape in;
    struct Shape out;
    void (*actFunc)(struct conv2DLayer*);
    void (*derivFunc)(struct conv2DLayer*);
    void (*forward_pass)(struct conv2DLayer*);
    void (*backward_weights)(struct conv2DLayer*);
    void (*backward_delta)(struct conv2DLayer*, float*);
    void (*free_layer)(struct conv2DLayer*);
} conv2DLayer;


void makeWeights( Matrix*);

float* makeKernels(int, int, int);

void freeConv2DLayer(conv2DLayer*);

conv2DLayer* createConv2DLayer(char[], struct Shape, int, int, int, int, int, int, bool);

void forwardConv2D( conv2DLayer*);
//
void backwardConv2D( conv2DLayer*);

void deltaConv2D(conv2DLayer*, float*);

#endif //_CONV2D_H