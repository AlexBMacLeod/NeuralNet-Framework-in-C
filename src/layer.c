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
#include "../include/layer.h"
//#include "../include/nn.h"

#define ALPHA .02


#define Fn_apply(type, fn, ...) {                                   \
void *stopper_for_apply = (int[]){0};                               \
type **list_for_apply = (type*[]){__VA_ARGS__, stopper_for_apply};  \
for (int i=0; list_for_apply[i] != stopper_for_apply; i++)          \
fn(list_for_apply[i]);                                              \
}

#define free_all(...) apply_func(void, free, __VA_ARGS__);


void freeLayer(Layer* layer)
{
    layer->weights->freeMem(layer->weights);
    layer->output->freeMem(layer->output);
    //layer->input->freeMem(layer->input);
    layer->deriv->freeMem(layer->deriv);
    layer->delta->freeMem(layer->delta);
    free(layer);
}

void makeWeights( Matrix* matrix)
{
    srand(time(NULL));
    for(int i = 0; i < matrix->shape.y; i++)
    {
        for(int j = 0; j < matrix->shape.x; j++)
        {
            matrix->data[i*matrix->shape.x+j] = (((float)rand()/(float)(RAND_MAX)));
        }
    }
}

void forward( Layer *layer)
{
    for(int i=0;i<layer->out;i++)
    {
        for(int j=0;j<layer->in;j++)
        {
            layer->output->data[i] += layer->input->data[j] * layer->weights->data[i*layer->in+j];
        }
    }
    layer->actFunc(layer);
    layer->derivFunc(layer);

}

/*
void initLinear( Layer **layer, char activation[], int in, int out)
{
    
    (*layer)->deriv = createMatrix( out, 1);
    (*layer)->weights = createMatrix( in, out);
    (*layer)->output = createMatrix( out, 1);
    (*layer)->input = createMatrix( in, 1);

    (*layer)->in = in;
    (*layer)->out = out;
    makeWeights( (*layer)->weights);
    if(strcmp(activation, "relu") == 0)
    {
        (*layer)->actFunc = relu;
        (*layer)->derivFunc = relu_deriv;
    }else{
        (*layer)->actFunc = none;
        (*layer)->derivFunc = none;
    }
    //layer->actFunc = funcs->func;
    //layer->derivFunc = funcs->deriv;
    (*layer)->free_layer = freeLayer;
    (*layer)->forward_pass = forward;
}
*/
Layer* createLayer(char activation[], int in, int out)
{
    Layer *layer = (Layer*)malloc(sizeof(Layer));
    //initLinear(&layer, activation, in, out);
    layer->deriv = createMatrix( out, 1);
    layer->weights = createMatrix( in, out);
    layer->output = createMatrix( out, 1);
    //layer->input = createMatrix( in, 1);
    layer->input = NULL;
    layer->delta = createMatrix(out, 1);
    layer->nextDelta = NULL;
    layer->nextWeights = NULL;

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
    //layer->actFunc = funcs->func;
    //layer->derivFunc = funcs->deriv;
    layer->free_layer = freeLayer;
    layer->forward_pass = forward;
    layer->backward_pass = backward;
    return layer;
}

/*def backward(self, front):
if self.activation:
    delta = front*self._relu2deriv(self.out)
    out = delta.dot(self.weights.T)
else:
    delta = self.out - front
    out = delta.dot(self.weights.T)
self.weights -= alpha * self.input.T.dot(delta)
return out
*/
void backward(struct Layer* layer, float* front)
{
    if(layer->nextDelta==NULL)
    {   
        if(layer->out==1)
        {
            float tmpDelta = layer->output->data - front;
            memmove(layer->delta->data, &tmpDelta, sizeof(float));
        }else{
        for(int i=0;i<layer->out;i++) layer->delta->data[i] = layer->output->data[i] - front[i];
        }
    }else{
        int col = sizeof(layer->nextWeights->data) / sizeof(layer->nextWeights->data[0]) / layer->out;
        for(int i=0;i<layer->out;i++)
        {
            for(int j=0;j<col;j++)
            {
                layer->delta->data[i] += layer->nextDelta->data[j] * layer->nextWeights->data[i*col+j];
            }
        }
        for(int i=0;i<layer->out;i++) layer->delta->data[i] = layer->delta->data[i] * layer->deriv->data[i];
    }
    for(int i=0; i<layer->in;i++)
    {
        for(int j=0; j<layer->out;j++)
        {
            layer->weights->data[i*layer->out+j] -= ALPHA * layer->input->data[i] * layer->output->data[j]; 
        }
    }
}
