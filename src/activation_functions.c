#include <stdio.h>
#include <unistd.h>

#include "../include/activation_functions.h"

void relu(LinearLayer* layer)
{
    for(int i=0; i<layer->out; i++)
    {
        if(layer->output->data[i]<0)
        {
            layer->output->data[i] = 0.0f;
        }
    }
}

void relu_deriv(LinearLayer* layer)
{
    for(int i=0; i<layer->out; i++)
    {
        if(layer->output->data[i]>0)
        {
            layer->deriv->data[i] = 1.0f;
        }else{
            layer->deriv->data[i] = 0.0f;
        }
    }
}
void none(LinearLayer* layer)
{
    sleep(0);
}