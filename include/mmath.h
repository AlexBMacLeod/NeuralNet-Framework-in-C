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

void nonpaddedConvolutionalKernel(Matrix*, Matrix*, Matrix*,int);
/*
So this is a bit of a beast but the basic idea is the same, we're sliding a kernel, or a number
accross the image and storing the results into the output. The main complexity is that all
the arrays are linearized which leads to nasty looking equations. The secondary complexity
is correctly slicing the arrays, especially as they're indexed differently. Also I've inserted
light threading to speed things up here, with a basic omp for loop.
*/

void paddedConvolutionalKernel(Matrix*, Matrix*, Matrix*,int);
//I could mix the padded and non padded kernels together, but from a readability stand point, I'm not sure if it makes sense.

#endif