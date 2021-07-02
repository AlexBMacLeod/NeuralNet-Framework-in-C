#include <stdio.h>
#include <stdlib.h>


#include "../include/input.h"

void freeInput(inputLayer* layer)
{
    layer->output->freeMem(layer->output);
    free(layer);
}

static void flatten_out(inputLayer **layer)
{
    (*layer)->out.y = (*layer)->out.x*(*layer)->out.y*(*layer)->out.z;
    (*layer)->out.x = 1; (*layer)->out.x=1;
}

static void flatten(inputLayer *layer)
{
    layer->output->flatten(layer->output);
    flatten_out(&layer);
}

inputLayer* createInput(struct Shape out, float lr)
{
        inputLayer *layer = malloc(sizeof(inputLayer));
        layer->lr = lr;
        layer->out = out;
        layer->flat = out.x * out.y * out.z;
        layer->output = createMatrix( out.n, out.x, out.y, out.z);
        layer->flatten = flatten;
        layer->free_layer = freeInput;
        return layer;
}
