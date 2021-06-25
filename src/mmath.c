#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "../include/mmath.h"

Matrix* createInverse(Matrix* matrix)
{
    Matrix* newMatrix = createMatrix(matrix->shape.y, matrix->shape.x);
    for(int i=0; i<matrix->shape.x; i++)
    {
        for(int j=0; j<matrix->shape.y; j++)
        {
            newMatrix->data[j*matrix->shape.x+i] = matrix->data[matrix->shape.y*i+j];
        }
    }
    return newMatrix;
}

void matrixMultiplication(Matrix* A, Matrix* B, Matrix* C)
{
    float sum;
    int i, j, k;
    assert(A->shape.y == B->shape.x);
    assert(A->shape.x == C->shape.x);
    assert(B->shape.y == C->shape.y);
    //Here we create our parallel loops, we specify that the variables A, B and C are shared between all threads where as
    //i, j and k are private to each thread. This is fairly similar to basic matrix multiplication in
    //CUDA where the matrices would be global and i, j, k local. The performance constraint there as well as here is how the
    //matrices are accessed since C stores arrays in row major form, coalescing memory accesses but that will be done
    //in the CUDA code, not here. Also CUDA allows the use of Shared memory and aggressive caching but these are also beyond
    //the scope of the basic C code
    #pragma omp parallel shared(A,B,C) private(i, sum, j,k)
    {
        #pragma omp for schedule(static)
        for (i = 0; i < A->shape.x; i++) {
            for (j = 0; j < B->shape.y; j++) {
                sum = 0;
                for (k = 0; k < A->shape.y; k++)
                    sum = sum + A->data[i * A->shape.y + k] * B->data[k * B->shape.y + j];
                C->data[i * B->shape.y + j] = sum;
            }
        }
    }
}

void matrixVecMultiplication(Matrix* A, Matrix* b, Matrix* c)
{
    assert(A->shape.y==b->shape.x);
    float sum;
    int i, k;
    #pragma omp parallel shared(A,b,c) private(i,k)
    {
        #pragma omp for schedule(static)
        for (i = 0; i < A->shape.x; i++) 
        {
            sum = 0;
            for (k = 0; k < A->shape.y; k++)
                sum += A->data[i * A->shape.y + k] * b->data[k];
            c->data[i] = sum;
        }
    }
}

void vecMatrixMultiplication(Matrix* a, Matrix* B, Matrix* c)
{
    assert(a->shape.y==B->shape.x);
    float sum;
    int j, k;
    #pragma omp parallel shared(a,B,c) private(j,k)
    {
        #pragma omp for schedule(static)
        for(j=0; j<B->shape.y; j++)
        {
            sum = 0;
            for(k=0; k<a->shape.y; k++)
            {
                sum += a->data[k] * B->data[k* B->shape.y + j];
            }
            c->data[j] = sum;
        }
    }
}

void vecVecMultiplication(Matrix* a, Matrix* b, Matrix* C)
{
    assert(a->shape.y == b->shape.x);
    for(int i=0; i<a->shape.x; i++)
    {
        for(int j=0; j<b->shape.y; j++)
        {
            C->data[i*b->shape.y + j] = a->data[i] * b->data[j];
        }
    }
}

void matrixSubtraction(Matrix* A, Matrix* B)
{
    assert(A->shape.x == B->shape.x);
    assert(A->shape.y == B->shape.y);
    for(int i=0; i<A->shape.x; i++)
    {
        for(int j=0; j<A->shape.y; j++)
        {
            A->data[i*A->shape.y + j] -= B->data[i*A->shape.y + j];
        }
    }
}

void vecElemMultiplication(Matrix* a, Matrix* b)
{
    if(a->shape.x>1)
    {
        assert(a->shape.x == b->shape.x);
        for(int i=0;i<a->shape.x;i++) a->data[i] = a->data[i] * b->data[i];
    }
    else if (a->shape.y>1)
    {
        assert(a->shape.y == b->shape.y);
        for(int i=0;i<a->shape.y;i++) a->data[i] = a->data[i] * b->data[i];
    }
    else
    {
        assert(a->shape.x == b->shape.x);
        for(int i=0;i<a->shape.x;i++) a->data[i] = a->data[i] * b->data[i];
    }
    
}

void matrixScalarMultiplication(Matrix* A, float sc)
{
    for(int i=0; i<A->shape.x; i++)
    {
        for(int j=0; j<A->shape.y; j++)
        {
            A->data[i*A->shape.y+j] = A->data[i*A->shape.y+j] * sc;
        }
    }
}

void elemMatrixMultInPlace(Matrix* A, Matrix* B)
{
    int i, j;
    assert(A->shape.x == B->shape.x);
    assert(A->shape.y == B->shape.y);

    #pragma omp parallel shared(A,B) private(i,j)
    {
        #pragma omp for schedule(static)
        for (i = 0; i < A->shape.x; i++) {
            for (j = 0; j < B->shape.y; j++) {
                     A->data[i*B->shape.y+j] = A->data[i*B->shape.y+j] * B->data[i*B->shape.y+j];
            }
        }
    }
}