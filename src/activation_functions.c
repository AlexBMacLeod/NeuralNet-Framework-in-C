#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>

#include "../include/activation_functions.h"

void relu(LinearLayer* layer)
{
    for(int i=0; i<layer->batch_size; i++)
    {
        for(int j=0; j<layer->out; j++)
        {
            if(layer->output->data[i*layer->out+j]<0) layer->output->data[i*layer->out+j] = 0.0f;
        }
    }
}

void relu_deriv(LinearLayer* layer)
{
    for(int i=0; i<layer->batch_size; i++)
    {
        for(int j=0; j<layer->out; j++)
        {
            if(layer->output->data[i*layer->out+j]>0)
            {
                layer->deriv->data[i*layer->out+j] = 1.0f;
            }else{
                layer->deriv->data[i*layer->out+j] = 0.0f;
            }
        }
    }
}
void none(LinearLayer* layer)
{
    return;
}/*
def tanh(x):
    return np.tanh(x)

def tanh2deriv(output):
    return 1 - (output ** 2)

def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)
    */
void tanhAct(LinearLayer* layer)
{
    for(int i=0; i<layer->batch_size; i++)
    {
        for(int j=0; j<layer->out; j++)
        {
            layer->output->data[i*layer->out+j] = tanh(layer->output->data[i*layer->out+j]);
        }
    }
}

void tanhAct_deriv(LinearLayer* layer)
{
    for(int i=0; i<layer->batch_size; i++)
    {
        for(int j=0; j<layer->out; j++)
        {
            layer->deriv->data[i*layer->out+j] = 1-pow(layer->output->data[i*layer->out+j],2);
        }
    }
}

void softMax(LinearLayer* layer)
{
    float *tmp = calloc(layer->batch_size, sizeof(float));
    for(int i=0; i<layer->batch_size; i++)
    {
        for(int j=0; j<layer->out; j++)
        {
            layer->output->data[i*layer->out+j] = exp(layer->output->data[i*layer->out+j]);
            tmp[i] += layer->output->data[i*layer->out+j];
        }
    }
    for(int i=0; i<layer->batch_size; i++)
    {
        for(int j=0; j<layer->out; j++)
        {
            layer->output->data[i*layer->out+j] = layer->output->data[i*layer->out+j]/tmp[i];
        }
    }
    free(tmp);
}

