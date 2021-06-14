#ifndef _COMMON_H
#define _COMMON_H

void load_data(char[], float *);

void checkLen(char[], int*);

void one_hot_encoder(int*, float*, int);

void splitLabels(float*, float*, int*, int);

#endif