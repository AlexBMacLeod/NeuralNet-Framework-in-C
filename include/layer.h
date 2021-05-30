//
// Created by alex on 30/05/21.
//
#ifndef _LAYER_H
#define _LAYER_H

#include "activation_functions.h"
#include "tensor.h"



typedef struct linearLayer{
    Matrix *weights;
    //struct Matrix *derivative;
    Matrix *output;
    Matrix *deriv;
    Matrix *input;
    int in;
    int out;
    void (*actFunc)(struct linearLayer*);
    void (*derivFunc)(struct linearLayer*);
    Matrix* (*forward_pass)(struct linearLayer*, Matrix*);
    //void (*backward_pass)(struct linearLayer*, float*);
    void (*free_layer)(struct linearLayer*);
} linearLayer;




/*
The idea with linearLayer is to encapsulate the data and funcitons needed within one struct. 
Here we both store the information needed for each linear layer, as well as the functions
to both create and destroy the layer, as well as to work with the data inside.
While the end result doesn't include self like a python function might, it works well enough.
An example for a forward pass for a linearLayer named layer would be
layer.forward_pass(&layer, input);
The backward pass would then be:
layer.backward_pass(&layer, delta);
*/

static void freeLayer(linearLayer*);

//void initLinear( linearLayer*, int, int, activation *funcs);
void initLinear( linearLayer**, int x, int y);

Matrix *forward( linearLayer*, Matrix*);
//
//void backward( linearLayer*, float*);

#endif //_LAYER_H