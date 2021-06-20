#ifndef _NN_H
#define _NN_H

#include "linear.h"

struct Node  {
	struct Node* next;
	struct Node* prev;
    LinearLayer *layer;
};

typedef struct NeuralNet{
	void (*add_linear_layer)(char[], int);
	void (*forward_pass)(float*, float*);
	void (*backward_pass)(float*);
	void (*clean_up)();
}NeuralNet;

extern struct Node* head; // global variable - pointer to head node.

//Creates a new Node and returns pointer to it. 
struct Node* GetNewNode(char[], int, int);

struct Node* GetFirstNode(float lr, int out);

//Inserts a Node at head of doubly linked list
void InsertAtHead(char[], int);

void InsertFirst(float lr, int out);

//Inserts a Node at tail of Doubly linked list
void InsertAtTail(char[], int, int);

NeuralNet createNetwork(float, int);

void Forward(float*, float*);

void Delete();

void Backward(float*);

int validation_run(float*, float*, int, NeuralNet);

#endif //_NN_H