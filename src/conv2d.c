#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "../include/conv2d.h"
#include "../include/activation_functions.h"
#include "../include/mmath.h"
//#include "../include/shape.h"

conv2DLayer* createConv2DLayer(char activation[], struct Shape in, int stride, int in_channels, int out_channels, int kernel_size, bool padding)
{
    conv2DLayer *layer = malloc(sizeof(conv2DLayer));
    int batch_size = in.n;
    if(padding){
        layer->output = createMatrix( batch_size, in.x, in.y, out_channels);
        layer->deriv = createMatrix( batch_size, in.x, in.y, out_channels);
    }else{
        assert((in.x-kernel_size+1)>0);
        assert((in.y-kernel_size+1)>0);
        layer->output = createMatrix( batch_size, (in.x-kernel_size+1), (in.y-kernel_size+1), out_channels);
        layer->deriv = createMatrix( batch_size, (in.x-kernel_size+1), (in.y-kernel_size+1), out_channels);
    }
    //layer->delta = createMatrix( batch_size, , out, 1);
    layer->kernels = createMatrix(1, in_channels, kernel_size*kernel_size, out_channels);

    assert(floor(kernel_size/2)!=ceil(kernel_size/2));
    layer->kernel_size = kernel_size;
    layer->padding = padding;
    layer->batch_size = batch_size;
    layer->in = in;
    layer->stride = stride;
    if(layer->padding) {layer->out.n=in.n;layer->out.x=layer->in.x;layer->out.y=layer->in.y;layer->out.z=out_channels;
    }else{ layer->out.x = (layer->in.x - (layer->kernel_size - 1))/layer->stride;
        layer->out.y = (layer->in.y - (layer->kernel_size - 1))/layer->stride;
    }

    makeWeights( layer->kernels);
    if(strcmp(activation, "relu") == 0)
    {
        layer->actFunc = relu2C;
        layer->derivFunc = relu_deriv2C;
    }else if(strcmp(activation, "tanh")==0){
        layer->actFunc = tanhAct2C;
        layer->derivFunc = tanhAct_deriv2C;
    }else{
        layer->actFunc = none2C;
        layer->derivFunc = none2C;
    }

    layer->free_layer = freeLayer;
    layer->forward_pass = forward;
    layer->backward_weights = backward;
    layer->backward_delta = delta;
    layer->nextDelta = NULL;
    layer->nextWeights = NULL;
    return layer;
}

void makeKernelWeights( Matrix* matrix)
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