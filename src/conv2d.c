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
    //Idea change delta to reflect channels of in layer to enable element multiplication
    if(padding){
        layer->output = createMatrix( batch_size, floor(in.x/stride), floor(in.y/stride), out_channels);
        layer->deriv = createMatrix( batch_size, floor(in.x/stride), floor(in.y/stride), out_channels);
        layer->delta = createMatrix( batch_size, floor(in.x/stride), floor(in.y/stride), in_channels);
    }else{
        assert((in.x-kernel_size+1)>0);
        assert((in.y-kernel_size+1)>0);
        layer->output = createMatrix( batch_size, floor((in.x-kernel_size+1)/stride), floor((in.y-kernel_size+1)/stride), out_channels);
        layer->deriv = createMatrix( batch_size, floor((in.x-kernel_size+1)/stride), floor((in.y-kernel_size+1)/stride), out_channels);
        layer->delta = createMatrix( batch_size, floor((in.x-kernel_size+1)/stride), floor((in.y-kernel_size+1)/stride), out_channels);
    }
    //layer->delta = createMatrix( batch_size, , out, 1);
    layer->kernels = createMatrix(in_channels, kernel_size, kernel_size, out_channels);
    layer->dK = createMatrix(in_channels, kernel_size, kernel_size, out_channels);

    
    layer->kernel_size = kernel_size;
    layer->padding = padding;
    layer->batch_size = batch_size;
    layer->in = in;
    layer->stride = stride;
    if(layer->padding){
        layer->out.n=layer->in.n;
        layer->out.x=layer->in.x;
        layer->out.y=layer->in.y;
        layer->out.z=out_channels;
    }
    else{ 
        layer->out.x = (layer->in.x - (layer->kernel_size - 1))/layer->stride;
        layer->out.y = (layer->in.y - (layer->kernel_size - 1))/layer->stride;
        layer->out.n = batch_size;
        layer->out.z = out_channels;
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
    if(layer->padding)
        padDevolved2d(layer->input, layer->kernels, layer->output,layer->stride);
    else 
        nonpaddedConvolutionalKernel(layer->input, layer->kernels, layer->output,layer->stride);
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
    layer->delta->zero(layer->delta);
    layer->dK->zero(layer->dK);
    elemMatrixMultInPlace(layer->nextDelta, layer->deriv);

    int pad = (layer->kernels->shape.x-1)/2;
    int o_M, o_X, o_Y, i_C, o_C, k_X, k_Y, l, i, j, k, m, n, p;
    i_C=layer->input->shape.z;
    o_M=layer->nextDelta->shape.n; o_X=layer->nextDelta->shape.x; o_Y=layer->nextDelta->shape.y; o_C=layer->nextDelta->shape.z;
    k_X=layer->kernels->shape.x; k_Y=layer->kernels->shape.y;
    #pragma omp parallel shared(layer) private(l, i, j, k, m, n, p)
    {
        #pragma omp for schedule(static)
        for(l=0;l<i_C;l++)
        {
            for(i=0; i<o_M; i++)
            {
                for(j=0;j<o_X;j++)
                {
                    for(k=0;k<o_Y;k++)
                    {
                        for(m=0;m<k_X;m++)
                        {
                            for(n=0;n<k_Y;n++)
                            {
                                for(p=0;p<o_C;p++)
                                    {
                                    if((j-pad)>=0 && (j+pad)<layer->out.x && (k-pad)>=0 && (k+pad)<layer->out.y)
                                    {
                                        layer->delta->data[i*o_X*o_Y*i_C+(j-pad+m)*o_Y*i_C+(k-pad+n)*i_C+l]
                                        += layer->kernels->data[l*k_X*k_Y*o_C+m*k_Y*o_C+n*o_C+p]
                                        * layer->nextDelta->data[i*o_X*o_Y*o_C+j*o_Y*o_C+k*o_C+p];


                                        layer->dK->data[l*k_X*k_Y*o_C+m*k_Y*o_C+n*o_C+p]
                                        += layer->nextDelta->data[i*o_X*o_Y*o_C+j*o_Y*o_C+k*o_C+p]
                                        * layer->input->data[i*o_X*o_Y*i_C+(j-pad+m)*o_Y*i_C+(k-pad+n)*i_C+l];
                                    }
                                }
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