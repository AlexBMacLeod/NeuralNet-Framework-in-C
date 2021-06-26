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
    float lr;
    bool padding;
    int kernel_size;
    int batch_size;
    Shape in;
    Shape out;
    void (*actFunc)(struct conv2DLayer*);
    void (*derivFunc)(struct conv2DLayer*);
    void (*forward_pass)(struct conv2DLayer*);
    void (*backward_weights)(struct conv2DLayer*);
    void (*backward_delta)(struct conv2DLayer*, float*);
    void (*free_layer)(struct conv2DLayer*);
} conv2DLayer;


void makeWeights( Matrix*);

float* makeKernels(int, int, int);

void freeLayer(conv2DLayer*);

//void initLinear( layer*, int, int, activation *funcs);
void initLayer( conv2DLayer**, char[], int, int, bool);

conv2DLayer* createConv2DLayer(char[], Shape, int, int, bool);

void forward( conv2DLayer*);
//
void backward( conv2DLayer*);

void delta(conv2DLayer*, float*);

#endif //_CONV2D_H