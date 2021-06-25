#ifndef _MMATH_H
#define _MMATH_H

#include "tensor.h"
/*
Essentially this contains basic linear algebra operations
Most of the code is fairly straight forward, the only liberties I've taken are that
I didn't include multi-dimensional arrays, so all tensors are written and accessed in row
major 1D order. Linearizing tensors is an important concept for CUDA.
I've also added light multi-threading in the form of omp for loops because
single threaded it runs quite slow, and basic omp threads for loops
don't impact readability at all. Due to concerns about overhead multi-threading was 
only added to level 2 and 3 operations. From a performance perspective this is a far cry
from the massively parallel operations in CUDA but one should expect a decent performance increase
especially with 4 or so cores available in the CPU.
*/

Matrix* createInverse(Matrix*);

void matrixMultiplication(Matrix*, Matrix*, Matrix*);

void elemMatrixMultInPlace(Matrix*, Matrix*);

void matrixVecMultiplication(Matrix*, Matrix*, Matrix*);

void vecMatrixMultiplication(Matrix*, Matrix*, Matrix*);

void matrixSubtraction(Matrix*, Matrix*);

void vecVecMultiplication(Matrix*, Matrix*, Matrix*);

void vecElemMultiplication(Matrix*, Matrix*);

void matrixScalarMultiplication(Matrix*, float);

#endif