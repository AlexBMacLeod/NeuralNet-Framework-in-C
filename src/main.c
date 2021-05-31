#include <stdio.h>
#include "../include/nn.h"


int main(void)
{
    float in[5] = {0, 1, 2, 3, 4};
    net_add_layer("relu", 5, 5);
    net_add_layer("relu", 5, 5);
    //insertLayerFirst("relu", 5, 5);
    //insertLayerFirst("relu", 5, 5);
    net_forward(in, in);
    net_delete();
    return 0;
}