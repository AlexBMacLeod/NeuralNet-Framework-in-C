#ifndef _INPUT_H_
#define _INPUT_H_

#include "tensor.h"
#include "shape.h"

/*
Just a basic struct to store basic information such as input share, 
learning rate, and batch. Before I had been using a linear layer but
ran in to some issues with dimensionality of output specifically with flattening
*/

typedef struct inputLayer{
    Matrix *output;
    float lr;
    int flat;
    struct Shape out;
    void (*flatten)(struct inputLayer*);
    void (*free_layer)(struct inputLayer*);
} inputLayer;

void freeInput(inputLayer*);

inputLayer* createInput(struct Shape, float);

#endif