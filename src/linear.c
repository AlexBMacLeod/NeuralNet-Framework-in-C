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
    layer->deriv->freeMem(layer->deriv);
    layer->delta->freeMem(layer->delta);
    free(layer);
}

void makeWeights( Matrix* matrix)
{
    srand(time(NULL));
    for(int i = 0; i < matrix->shape.x; i++)
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


LinearLayer* createLayer(char activation[], int in, int out, int batch_size)
{
    LinearLayer *layer = malloc(sizeof(LinearLayer));

    layer->deriv = createMatrix( batch_size, 1, out, 1);
    layer->weights = createMatrix( 1, in, out, 1);
    layer->output = createMatrix( batch_size, 1, out, 1);
    layer->delta = createMatrix( batch_size, 1, out, 1);

    layer->batch_size = batch_size;
    layer->in = in;
    layer->out = out;
    makeWeights( layer->weights);
    if(strcmp(activation, "relu") == 0)
    {
        layer->actFunc = relu;
        layer->derivFunc = relu_deriv;
    }else if(strcmp(activation, "softmax")==0){
        layer->actFunc = softMax;
        layer->derivFunc = none;
    }else if(strcmp(activation, "tanh")==0){
        layer->actFunc = tanhAct;
        layer->derivFunc = tanhAct_deriv;
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
    if(layer->nextDelta==NULL)
    {   
        float b_size = layer->batch_size;
        for(int i=0; i<layer->delta->shape.x; i++)
        {
            for(int j=0; j<layer->delta->shape.y; j++)
            {
                layer->delta->data[i*layer->out+j] = (layer->output->data[i*layer->out+j] - y[i*layer->out+j])/b_size;
            }
        }
    }else{
    Matrix *invWeights = createInverse(layer->nextWeights);
    matrixMultiplication(layer->nextDelta, invWeights, layer->delta);
    invWeights->freeMem(invWeights);
    }
    if(layer->derivFunc!=none){
        elemMatrixMultInPlace(layer->delta, layer->deriv);
    }
}


void backward(struct LinearLayer* layer)
{
    Matrix *invInput = createInverse(layer->input);
    Matrix *weightsDelta = createMatrix(1, layer->in, layer->out, 1);
    matrixMultiplication(invInput, layer->delta, weightsDelta);
    matrixScalarMultiplication(weightsDelta, layer->lr);
    matrixSubtraction(layer->weights, weightsDelta);
    weightsDelta->freeMem(weightsDelta);
    invInput->freeMem(invInput);
}