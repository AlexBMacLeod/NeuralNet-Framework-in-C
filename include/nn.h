
#ifndef _NN_H
#define _NN_H

#include "layer.h"

struct node {
    struct node* next; // Pointer to next node in DLL
    struct node* prev; // Pointer to previous node in DLL
    Layer *layer;
};

extern struct node *head = NULL;

extern struct node *last = NULL;

extern struct node *current = NULL;

bool isEmpty();

int length();

void displayForward();

void displayBackward();

void insertFirst(char *, int, int);

void insertLast(float *);

struct node* deleteFirst();

struct node* deleteLast();

void deleteList();

#endif //_NN_H