//
// Created by alex on 30/05/21.
//
#ifndef _LINEAR_H
#define _LINEAR_H

#include "tensor.h"
//#include "activation_functions.h"



typedef struct LinearLayer{
    Matrix *weights;
    Matrix *output;
    Matrix *deriv;
    Matrix *input;
    Matrix *delta;
    Matrix *nextDelta;
    Matrix *nextWeights;
    float lr;
    int in;
    int out;
    void (*actFunc)(struct LinearLayer*);
    void (*derivFunc)(struct LinearLayer*);
    void (*forward_pass)(struct LinearLayer*);
    void (*backward_weights)(struct LinearLayer*);
    void (*backward_delta)(struct LinearLayer*, float*);
    void (*free_layer)(struct LinearLayer*);
} LinearLayer;



/*
The idea with layer is to encapsulate the data and funcitons needed within one struct. 
Here we both store the information needed for each layer, as well as the functions
to both create and destroy the layer, as well as to work with the data inside.
While the end result doesn't include self like a python function might, it works well enough.
An example for a forward pass for a layer named layer would be
layer.forward_pass(&layer, input);
The backward pass would then be:
layer.backward_pass(&layer, delta);
*/

void makeWeights( Matrix*);

void freeLayer(LinearLayer*);

//void initLinear( layer*, int, int, activation *funcs);
void initLayer( LinearLayer**, char[], int, int);

LinearLayer* createLayer(char[], int, int);

void forward( LinearLayer*);
//
void backward( LinearLayer*);

void delta(LinearLayer*, float*);

#endif //_LINEAR_H