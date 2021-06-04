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

#define ALPHA .2


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
    memset(layer->output->data, 0, layer->out*sizeof(float));
    for(int i=0;i<layer->in;i++)
    {
        for(int j=0;j<layer->out;j++)
        {
            layer->output->data[j] += layer->input->data[i] * layer->weights->data[layer->out*i+j];
        }
    }
    layer->actFunc(layer);
    layer->derivFunc(layer);

}


Layer* createLayer(char activation[], int in, int out)
{
    Layer *layer = (Layer*)malloc(sizeof(Layer));

    layer->deriv = createMatrix( out, 1);
    layer->weights = createMatrix( in, out);
    layer->output = createMatrix( out, 1);
    layer->delta = createMatrix(out, 1);


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
    layer->backward_pass = backward;
    layer->nextDelta = NULL;
    layer->nextWeights = NULL;
    layer->nextOut;
    return layer;
}


void backward(struct Layer* layer, float* front)
{
    float alpha = .1;
    if(layer->nextDelta==NULL)
    {   
        if(layer->out==1)
        {
            float tmpDelta = *(layer->output->data) - *front;
            //memmove(layer->delta->data, &tmpDelta, sizeof(float));
            *(layer->delta->data) = tmpDelta;
        }else{
            for(int i=0;i<layer->out;i++) layer->delta->data[i] = layer->output->data[i] - front[i];
        }
    }else{
        //int col = sizeof(layer->nextWeights->data) / sizeof(layer->nextWeights->data[0]) / layer->out;
        for(int i=0;i<layer->out;i++)
        {
            for(int j=0;j<layer->nextOut;j++)
            {
                layer->delta->data[i] += layer->nextDelta->data[j] * layer->nextWeights->data[i*layer->nextOut+j];
            }
        }
        for(int i=0;i<layer->out;i++) layer->delta->data[i] = layer->delta->data[i] * layer->deriv->data[i];
    }
    for(int i=0; i<layer->in;i++)
    {
        for(int j=0; j<layer->out;j++)
        {
            layer->weights->data[i*layer->out+j] -= alpha * layer->input->data[i] * layer->delta->data[j]; 
        }
    }
}
