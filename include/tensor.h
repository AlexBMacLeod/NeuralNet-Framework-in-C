
#ifndef _TENSOR_H
#define _TENSOR_H

#include "shape.h"

typedef struct Matrix{
    struct Shape shape;
    float *data;
    void (*flatten)(struct Matrix*);
    void (*inputData)( struct Matrix*, float*);
    void (*giveMem)( struct Matrix*, struct Shape);
    void (*freeMem)( struct Matrix*);
    void (*zero)( struct Matrix*);
}Matrix;

Matrix* createMatrix( int, int x, int y, int z);


#endif //TENSOR_H
