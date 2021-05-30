#include <stdio.h>
#include "../include/nn.h"


int main(void)
{
    insertLayerFirst("relu", 5, 5);
    insertLayerFirst("relu", 5, 5);
    deleteList();
    return 0;
}