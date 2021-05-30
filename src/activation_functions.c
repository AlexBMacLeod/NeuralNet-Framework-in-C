#include <stdio.h>

#include "../include/activation_functions.h"

void relu(linearLayer* layer)
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
}

void relu_deriv(linearLayer* layer)
{
    int col = layer->output->shape.x;
    int row = layer->output->shape.y;

    for(int i=0; i<row; i++)
    {
        for(int j=0; j<col; j++)
        {
            if(layer->output->data[i*col+j]>0)
            {
                layer->deriv->data[i*col+j] = 1;
            }else{
                layer->deriv->data[i*col+j] = 0;
            }
        }
    }
}