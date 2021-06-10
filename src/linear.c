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


#define Fn_apply(type, fn, ...) {                                   \
void *stopper_for_apply = (int[]){0};                               \
type **list_for_apply = (type*[]){__VA_ARGS__, stopper_for_apply};  \
for (int i=0; list_for_apply[i] != stopper_for_apply; i++)          \
fn(list_for_apply[i]);                                              \
}

#define free_all(...) apply_func(void, free, __VA_ARGS__);


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
            matrix->data[i*matrix->shape.y+j] = 2.0f*(((float)rand()/(float)(RAND_MAX)))-1.0f;
        }
    }
}

void forward(LinearLayer *layer)
{
    //memset(layer->output->data, 0, layer->out*sizeof(float));
    vecMatrixMultiplication(layer->input, layer->weights, layer->output);
    layer->actFunc(layer);
    layer->derivFunc(layer);

}


LinearLayer* createLayer(char activation[], int in, int out)
{
    LinearLayer *layer = malloc(sizeof(LinearLayer));

    layer->deriv = createMatrix( 1, out);
    layer->weights = createMatrix( in, out);
    layer->output = createMatrix( 1, out);
    layer->delta = createMatrix(1, out);


    layer->in = in;
    layer->out = out;
    makeWeights( layer->weights);
    if(strcmp(activation, "relu") == 0)
    {
        layer->actFunc = relu;
        layer->derivFunc = relu_deriv;
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

void delta(struct LinearLayer* layer, float* y_hat)
{
    if(layer->nextDelta==NULL)
    {   
        if(layer->out==1)
        {
            *(layer->delta->data) = *(layer->output->data) - *y_hat;
        }else{
            for(int i=0;i<layer->out;i++) layer->delta->data[i] = layer->output->data[i] - y_hat[i];
        }
    }else{
    Matrix *invWeights = createInverse(layer->nextWeights);
    vecMatrixMultiplication(layer->nextDelta, invWeights, layer->delta);
    vecElemMultiplication(layer->delta, layer->deriv);
    invWeights->freeMem(invWeights);
    }
}


void backward(struct LinearLayer* layer)
{
    Matrix *invInput = createInverse(layer->input);
    Matrix *weightsDelta = createMatrix(layer->in, layer->out);
    vecVecMultiplication(invInput, layer->delta, weightsDelta);
    matrixScalarMultiplication(weightsDelta, layer->lr);
    matrixSubtraction(layer->weights, weightsDelta);
    weightsDelta->freeMem(weightsDelta);
    invInput->freeMem(invInput);
}