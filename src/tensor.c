#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/tensor.h"

static void input(Matrix*, float*);
static void reallocateMem( Matrix*, struct Shape);
static void freeMatrix( Matrix*);

static void reallocateMem(Matrix *matrix, struct Shape shape)
{
    //matrix->shape.x = x;
    //matrix->shape.y = y;
    matrix->data = realloc(matrix->data, sizeof(float)*matrix->shape.x*matrix->shape.y);
}

static void freeMatrix(Matrix *matrix)
{
    free(matrix->data);
    free(matrix);
}

static void flatten(Matrix *matrix)
{
    matrix->shape.y = matrix->shape.x*matrix->shape.y*matrix->shape.z;
    matrix->shape.x = 1;
    matrix->shape.z = 1;
}

static void input(Matrix* matrix, float* inMatrix)
{
    memcpy( matrix->data, inMatrix, sizeof(float)*matrix->shape.x*matrix->shape.y);
}

static void zero(Matrix* matrix)
{
    long data_size = matrix->shape.n*matrix->shape.x*matrix->shape.y*matrix->shape.z * sizeof(float);
    memset(matrix->data, 0, data_size);
}

Matrix* createMatrix( int n, int x, int y, int z)
{
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->shape.n = n;
    matrix->shape.x = x;
    matrix->shape.y = y;
    matrix->shape.z = z;
    matrix->giveMem = reallocateMem;
    matrix->flatten = flatten;
    matrix->freeMem = freeMatrix;
    matrix->inputData = input;
    matrix->zero = zero;
    matrix->data = calloc(matrix->shape.n*matrix->shape.x*matrix->shape.y*matrix->shape.z, sizeof(float));
    return matrix;
}

