
#ifndef _ACTIVATION_FUNCTIONS_H
#define _ACTIVATION_FUNCTIONS_H

#include "linear.h"
#include "conv2d.h"

void relu(LinearLayer*);

void relu_deriv(LinearLayer*);

void none(LinearLayer*);

void tanhAct(LinearLayer*);

void tanhAct_deriv(LinearLayer*);

void softMax(LinearLayer*);

void relu2C(conv2DLayer*);

void relu_deriv2C(conv2DLayer*);

void none2C(conv2DLayer*);

void tanhAct2C(conv2DLayer*);

void tanhAct_deriv2C(conv2DLayer*);


#endif