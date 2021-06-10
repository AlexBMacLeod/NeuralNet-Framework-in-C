
#ifndef _ACTIVATION_FUNCTIONS_H
#define _ACTIVATION_FUNCTIONS_H

#include "linear.h"

void relu(LinearLayer*);

void relu_deriv(LinearLayer*);

void none(LinearLayer*);

#endif