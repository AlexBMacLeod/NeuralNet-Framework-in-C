#ifndef _MMATH_H
#define _MMATH_H

#include "tensor.h"

Matrix* createInverse(Matrix*);

void matrixMultiplication(Matrix*, Matrix*, Matrix*);

void matrixVecMultiplication(Matrix*, Matrix*, Matrix*);

void vecMatrixMultiplication(Matrix*, Matrix*, Matrix*);

void matrixSubtraction(Matrix*, Matrix*);

void vecVecMultiplication(Matrix*, Matrix*, Matrix*);

void vecElemMultiplication(Matrix*, Matrix*);

void matrixScalarMultiplication(Matrix*, float);

#endif