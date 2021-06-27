#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "../include/conv2d.h"
#include "../include/activation_functions.h"
#include "../include/mmath.h"
//#include "../include/shape.h"

conv2DLayer* createConv2DLayer(char activation[], struct Shape in, int stride, int in_channels, int out_channels, int out, int batch_size, int kernel_size, bool padding)
{
    conv2DLayer *layer = malloc(sizeof(conv2DLayer));

    if(padding){
        layer->output = createMatrix( batch_size, in.x, in.y, out_channels);
        layer->deriv = createMatrix( batch_size, in.x, in.y, out_channels);
    }else{
        assert((in.x-kernel_size+1)>0);
        assert((in.y-kernel_size+1)>0);
        layer->output = createMatrix( batch_size, (in.x-kernel_size+1), (in.y-kernel_size+1), out_channels);
        layer->deriv = createMatrix( batch_size, (in.x-kernel_size+1), (in.y-kernel_size+1), out_channels);
    }
    layer->delta = createMatrix( batch_size, 1, out, 1);
    layer->kernels = createMatrix(1, in_channels, kernel_size*kernel_size, out_channels);

    assert(floor(kernel_size/2)!=ceil(kernel_size/2));
    layer->kernel_size = kernel_size;
    layer->padding = padding;
    layer->batch_size = batch_size;
    layer->in = in;
    layer->stride = stride;
    if(layer->padding) {layer->out.x = layer->in.x; layer->out.y=layer->in.y;
    }else{ layer->out.x = layer->in.x - (layer->kernel_size - 1);
        layer->out.y = layer->in.y - (layer->kernel_size - 1);
    }

    makeWeights( layer->kernels);
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

void makeWeights( Matrix* matrix)
{
    srand(time(NULL));
    for(int i = 0; i < matrix->shape.x; i++)
    {
        for(int j = 0; j < matrix->shape.y; j++)
        {
            for(int k=0; k<matrix->shape.z; k++)
            {
                matrix->data[(i*matrix->shape.y+j)*matrix->shape.z+k] = .2f*(((float)rand()/(float)(RAND_MAX)))-.1f;
            }
        }
    }
}


void addPadding(Matrix *img, int kernel_size)
{
    int pad = (kernel_size-1)/2;
    for(int l=0; l<img->shape.n; l++)
        for(int i = 0; i < img->shape.x+pad*2; i++)
        {
            for(int j = 0; j < img->shape.y; j++)
            {
                for(int k=0; k<img->shape.z; k++)
                {
                    if(i<pad || i>img->shape.x)
                        img->data[((l*(img->shape.x+2*pad)+i)*(img->shape.y+2*pad)+j)*img->shape.z+k] = 0;
                }
            }
        }
}

void forwardConv2DZeroPadding( conv2DLayer* layer)
{
    nonpaddedConvolutionalKernel(layer->input, layer->kernels, layer->output,layer->stride);
    layer->actFunc(layer);
    layer->derivFunc(layer);
}