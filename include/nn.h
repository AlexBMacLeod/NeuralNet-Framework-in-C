
#ifndef _NN_H
#define _NN_H

#include <stdbool.h>
#include "layer.h"

struct node {
    struct node* next; // Pointer to next node in DLL
    struct node* prev; // Pointer to previous node in DLL
    Layer *layer;
};

extern struct node *head;
extern struct node *last;
extern struct node *current;

bool isEmpty();

int length();

void displayForward();

void displayBackward();

void nn_linear(char[], int, int);

void net_add_layer(char[], int, int);

void nn_forward(float*, float*);

void nn_backward(float*);

struct node* deleteFirst();

struct node* deleteLast();

void nn_delete();

#endif //_NN_H