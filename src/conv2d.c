#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <string.h>

#include "../include/conv2d.h"
#include "../include/activation_functions.h"
#include "../include/mmath.h"
//#include "../include/shape.h"

conv2DLayer* createConv2DLayer(char activation[], struct Shape in, int stride, int in_channels, int out_channels, int kernel_size, bool padding)
{
    conv2DLayer *layer = malloc(sizeof(conv2DLayer));
    assert(floor((float)kernel_size/2.0f)!=ceil((float)kernel_size/2.0f));
    int batch_size = in.n;
    if(padding){
        layer->output = createMatrix( batch_size, floor(in.x/stride), floor(in.y/stride), out_channels);
        layer->deriv = createMatrix( batch_size, floor(in.x/stride), floor(in.y/stride), out_channels);
    }else{
        assert((in.x-kernel_size+1)>0);
        assert((in.y-kernel_size+1)>0);
        layer->output = createMatrix( batch_size, floor((in.x-kernel_size+1)/stride), floor((in.y-kernel_size+1)/stride), out_channels);
        layer->deriv = createMatrix( batch_size, floor((in.x-kernel_size+1)/stride), floor((in.y-kernel_size+1)/stride), out_channels);
    }
    //layer->delta = createMatrix( batch_size, , out, 1);
    layer->kernels = createMatrix(in_channels, kernel_size, kernel_size, out_channels);
    layer->dK = createMatrix(in_channels, kernel_size, kernel_size, out_channels);

    
    layer->kernel_size = kernel_size;
    layer->padding = padding;
    layer->batch_size = batch_size;
    layer->in = in;
    layer->stride = stride;
    if(layer->padding) {layer->out.n=in.n;layer->out.x=layer->in.x;layer->out.y=layer->in.y;layer->out.z=out_channels;
    }else{ layer->out.x = (layer->in.x - (layer->kernel_size - 1))/layer->stride;
        layer->out.y = (layer->in.y - (layer->kernel_size - 1))/layer->stride;
    }

    makeKernelWeights( layer->kernels);
    if(strncmp(activation, "relu", 5) == 0)
    {
        layer->actFunc = relu2C;
        layer->derivFunc = relu_deriv2C;
    }else if(strncmp(activation, "tanh", 5)==0){
        layer->actFunc = tanhAct2C;
        layer->derivFunc = tanhAct_deriv2C;
    }else{
        layer->actFunc = none2C;
        layer->derivFunc = none2C;
    }

    layer->free_layer = freeConv;
    layer->forward_pass = forwardConv2D;
    layer->backward_weights = weightUpdate;
    layer->backward_delta = backwardConv2D;
    layer->input=NULL;
    layer->nextDelta = NULL;
    layer->nextWeights = NULL;
    return layer;
}

void makeKernelWeights( Matrix* matrix)
{
    srand(time(NULL));
    int len = matrix->shape.n*matrix->shape.x*matrix->shape.y*matrix->shape.z;
    for(int i=0; i<len; i++){
        matrix->data[i] = .2f*(((float)rand()/(float)(RAND_MAX)))-.1f;
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

void forwardConv2D( conv2DLayer* layer)
{
    if(layer->padding){ 
        paddedConvolutionalKernel(layer->input, layer->kernels, layer->output,layer->stride);
    }
    else nonpaddedConvolutionalKernel(layer->input, layer->kernels, layer->output,layer->stride);
    layer->actFunc(layer);
    layer->derivFunc(layer);
}

void weightUpdate( conv2DLayer* layer)
{
    int len = layer->kernels->shape.n*layer->kernels->shape.y*layer->kernels->shape.x*layer->kernels->shape.z;
    for(int i=0;i<len;i++) layer->kernels->data[i] += layer->dK->data[i] * layer->lr;
}

void backwardConv2D( conv2DLayer* layer, float *y)
{
    memset(layer->delta->data, 0, layer->out.n*layer->out.x*layer->out.y*layer->out.z*sizeof(float));
    elemMatrixMultInPlace(layer->deriv, layer->kernels);
    int pad = (layer->kernels->shape.x-1)/2;
    for(int i=0; i<layer->out.n; i++)
    {
        for(int j=0;j<layer->out.x;j++)
        {
            for(int k=0;k<layer->out.y;k++)
            {
                for(int m=0;m<layer->kernels->shape.x;m++)
                {
                    for(int n=0;n<layer->kernels->shape.y;n++)
                    {
                        for(int l=0;l<layer->in.z;l++)
                        {
                            for(int p=0;p<layer->out.z;p++)
                            {
                                if((j-pad)>=0 && (j+pad)<layer->out.x && (k-pad)>=0 && (k+pad)<layer->out.y){
                                layer->delta->data[((i*layer->out.x+(j-pad+m))*layer->out.y+(k-pad+n))*layer->in.z+l]
                                += layer->kernels->data[((l*layer->kernels->shape.x+m)*layer->kernels->shape.y+n)*layer->out.z+p]
                                * layer->nextDelta->data[((i*layer->out.x+j)*layer->out.y+k)*layer->out.z+p];

                                layer->dK->data[((l*layer->kernels->shape.x+m)*layer->kernels->shape.y+n)*layer->out.z+p]
                                += layer->nextDelta->data[((i*layer->out.x+j)*layer->out.y+k)*layer->in.z+l]
                                * layer->input->data[((i*layer->out.x+(j-pad+m))*layer->out.y+(k-pad+n))*layer->in.z+l];}
                            }
                        }
                    }
                }
            }
        }
    }
}


void freeConv(conv2DLayer* layer)
{
    layer->kernels->freeMem(layer->kernels);
    layer->output->freeMem(layer->output);
    layer->deriv->freeMem(layer->deriv);
    layer->delta->freeMem(layer->delta);
    layer->dK->freeMem(layer->dK);
    free(layer);
}