
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

void insertLayerFirst(char[], int, int);

void insertLayerLast(char[], int, int);

float* network_forward(float*);

float* backward(float*);

struct node* deleteFirst();

struct node* deleteLast();

void deleteNetwork();

#endif //_NN_H