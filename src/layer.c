//
// Created by alex on 3/21/21.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "../include/activation_functions.h"
#include "../include/layer.h"
//#include "../include/nn.h"


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
    layer->input->freeMem(layer->input);
    layer->deriv->freeMem(layer->deriv);
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
    for(int i=0;i<layer->in;i++)
    {
        for(int j=0;j<layer->out;j++)
        {
            layer->output->data[j] += layer->input->data[i] * layer->weights->data[i*layer->out+j];
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
    layer->input = createMatrix( in, 1);

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
void backward(struct Layer* layer, float* front)
{
    float* delta;
    delta = malloc(sizeof(float)*layer->out);

    if(layer->actFunc != NULL)
    {
        relu_deriv(layer);
        matrixVector( layer->output, front, delta, layer->in, layer->out)
        realloc(front, sizeof(float)*layer->in);
        float* transpose_weights;
        transpose_weights = malloc(sizeof(float)*layer->out*layer->in);
        transpose( layer->weights, transpose_weights, layer->in, layer->out);
        matrixVector( delta, transpose_weights, front);
        free(transpose_weights);
    } else{
        *delta = layer->output - *front;
        float* transpose_weights;
        transpose_weights = malloc(sizeof(float)*layer->out*layer->in);
        transpose( layer->weights, transpose_weights, layer->in, layer->out);
        matrixVector( delta, transpose_weights, front);
        free(transpose_weights);

    }

}*/