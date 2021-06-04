#include <stdio.h>
#include <unistd.h>

#include "../include/activation_functions.h"
/*
void relu(Layer* layer)
{
    int col = layer->output->shape.x;
    int row = layer->output->shape.y;

    for(int i=0; i<row; i++)
    {
        for(int j=0; j<col; j++)
        {
            if(layer->output->data[i*col+j]<0)
            {
                layer->output->data[i*col+j] = 0;
            }
        }
    }
}*/

void relu(Layer* layer)
{
    for(int i=0; i<layer->out; i++)
    {
        if(layer->output->data[i]<0)
        {
            layer->output->data[i] = 0;
        }
    }
}

void relu_deriv(Layer* layer)
{
    for(int i=0; i<layer->out; i++)
    {
        if(layer->output->data[i]>0)
        {
            layer->deriv->data[i] = 1;
        }else{
            layer->deriv->data[i] = 0;
        }
    }
}
void none(Layer* layer)
{
    sleep(0);
}