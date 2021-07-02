//
// Created by alex on 3/21/21.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

#include "../include/activation_functions.h"
#include "../include/linear.h"
#include "../include/mmath.h"
//#include "../include/nn.h"




void freeLayer(LinearLayer* layer)
{
    layer->weights->freeMem(layer->weights);
    layer->output->freeMem(layer->output);
    if(layer->deriv!=NULL) layer->deriv->freeMem(layer->deriv);
    layer->delta->freeMem(layer->delta);
    if(layer->nextWeights==NULL) layer->nextDelta->freeMem(layer->nextDelta);
    free(layer);
}

void makeWeights( Matrix* matrix)
{
    srand(time(NULL));
    for(int i = 0; i < matrix->shape.n; i++)
    {
        for(int j = 0; j < matrix->shape.y; j++)
        {
            matrix->data[i*matrix->shape.y+j] = .2f*(((float)rand()/(float)(RAND_MAX)))-.1f;
        }
    }
}

void forward(LinearLayer *layer)
{
    //memset(layer->output->data, 0, layer->out*sizeof(float));
    matrixMultiplication(layer->input, layer->weights, layer->output);
    layer->actFunc(layer);
    layer->derivFunc(layer);

}


LinearLayer* createLayer(char activation[], struct Shape in, int out, int batch_size)
{
    LinearLayer *layer = malloc(sizeof(LinearLayer));

    //layer->flat = in.x*in.y*in.z;

    layer->deriv = NULL;
    layer->weights = createMatrix( in.y, 1, out, 1);
    layer->output = createMatrix( batch_size, 1, out, 1);
    layer->delta = createMatrix( batch_size, 1, in.y, 1);

    layer->batch_size = batch_size;
    layer->in = in;
    layer->out = out;
    makeWeights( layer->weights);
    if(strcmp(activation, "relu") == 0)
    {
        layer->actFunc = relu;
        layer->derivFunc = relu_deriv;
        layer->deriv = createMatrix( batch_size, 1, out, 1);
    }else if(strcmp(activation, "softmax")==0){
        layer->actFunc = softMax;
        layer->derivFunc = none;
    }else if(strcmp(activation, "tanh")==0){
        layer->actFunc = tanhAct;
        layer->derivFunc = tanhAct_deriv;
        layer->deriv = createMatrix( batch_size, 1, out, 1);
    }else{
        layer->actFunc = none;
        layer->derivFunc = none;
    }

    layer->free_layer = freeLayer;
    layer->forward_pass = forward;
    layer->backward_weights = backward;
    layer->backward_delta = delta;
    layer->nextDelta = NULL;
    layer->nextWeights = NULL;
    return layer;
}

void delta(struct LinearLayer* layer, float* y)
{
    if(layer->nextWeights==NULL)
    {   
        layer->nextDelta = createMatrix(layer->batch_size, 1 , layer->out, 1);
        float b_size = layer->batch_size;
        for(int i=0; i<layer->nextDelta->shape.n; i++)
        {
            for(int j=0; j<layer->nextDelta->shape.y; j++)
            {
                layer->nextDelta->data[i*layer->out+j] = (layer->output->data[i*layer->out+j] - y[i*layer->out+j])/b_size;
            }
        }
    }
    if(layer->deriv!=NULL) elemMatrixMultInPlace(layer->nextDelta, layer->deriv);
    Matrix *invWeights = createInverse(layer->weights);
    matrixMultiplication(layer->nextDelta, invWeights, layer->delta);
    invWeights->freeMem(invWeights);
    //if(layer->derivFunc!=none){
    //}
}


void backward(struct LinearLayer* layer)
{
    Matrix *invInput = createInverse(layer->input);
    Matrix *weightsDelta = createMatrix(layer->in.y, 1, layer->out, 1);
    matrixMultiplication(invInput, layer->nextDelta, weightsDelta);
    matrixScalarMultiplicationInPlace(weightsDelta, layer->lr);
    matrixSubtraction(layer->weights, weightsDelta);
    weightsDelta->freeMem(weightsDelta);
    invInput->freeMem(invInput);
}