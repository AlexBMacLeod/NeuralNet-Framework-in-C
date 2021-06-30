#include <stdio.h>
#include <stdlib.h>


#include "../include/input.h"

void freeInput(inputLayer* layer)
{
    layer->output->freeMem(layer->output);
    free(layer);
}

static void flatten(inputLayer *layer)
{
    layer->output->flatten(layer->output);
    layer->out.y = layer->out.x*layer->out.y*layer->out.z;
    layer->out.x = 1; layer->out.z=1;
}

inputLayer* createInput(struct Shape out, int lr)
{
        inputLayer *layer = malloc(sizeof(inputLayer));
        layer->lr = lr;
        layer->out = out;
        layer->output = createMatrix( out.n, out.x, out.y, out.z);
        layer->flatten = flatten;
        layer->free_layer = freeInput;
        return layer;
}
