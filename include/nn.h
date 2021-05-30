
#ifndef _NN_H
#define _NN_H

struct node {
    struct node* next; // Pointer to next node in DLL
    struct node* prev; // Pointer to previous node in DLL
    float row[30];
};

extern struct node *head = NULL;

extern struct node *last = NULL;

extern struct node *current = NULL;

bool isEmpty();

int length();

void displayForward();

void displayBackward();

void insertFirst(float *data);

void insertLast(float data[]);

struct node* deleteFirst();

struct node* deleteLast();

void deleteList();

#endif //_NN_H