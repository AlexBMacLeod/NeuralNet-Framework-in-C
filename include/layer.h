//
// Created by alex on 30/05/21.
//
#ifndef _LAYER_H
#define _LAYER_H

#include "tensor.h"
//#include "activation_functions.h"



typedef struct Layer{
    Matrix *weights;
    //struct Matrix *derivative;
    Matrix *output;
    Matrix *deriv;
    Matrix *input;
    int in;
    int out;
    void (*actFunc)(struct Layer*);
    void (*derivFunc)(struct Layer*);
    Matrix* (*forward_pass)(struct Layer*, Matrix*);
    //void (*backward_pass)(struct layer*, float*);
    void (*free_layer)(struct Layer*);
} Layer;




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

void freeLayer(Layer*);

//void initLinear( layer*, int, int, activation *funcs);
void initLayer( Layer**, char[], int, int);

Layer* createLayer(char[], int, int);

Matrix *forward( Layer*, Matrix*);
//
//void backward( layer*, float*);

#endif //_LAYER_H