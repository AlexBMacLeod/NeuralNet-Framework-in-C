
#ifndef _ACTIVATION_FUNCTIONS_H
#define _ACTIVATION_FUNCTIONS_H

#include "linear.h"

void relu(LinearLayer*);

void relu_deriv(LinearLayer*);

void none(LinearLayer*);

void tanhAct(LinearLayer*);

void tanhAct_deriv(LinearLayer*);

void softMax(LinearLayer*);


#endif