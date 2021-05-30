
#ifndef _TENSOR_H
#define _TENSOR_H

#include "shape.h"

typedef struct Matrix{
    struct Shape shape;
    float *data;
    void (*inputData)( struct Matrix*, float*);
    void (*giveMem)( struct Matrix*, struct Shape);
    void (*freeMem)( struct Matrix*);
}Matrix;

Matrix* createMatrix( int x, int y);
static void input(Matrix*, float*);
static void reallocateMem( Matrix*, struct Shape);
static void freeMatrix( Matrix*);

#endif //TENSOR_H
