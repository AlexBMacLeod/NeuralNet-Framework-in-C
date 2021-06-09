#ifndef _NN_H
#define _NN_H

#include "layer.h"

struct Node  {
	struct Node* next;
	struct Node* prev;
    Layer *layer;
};

typedef struct NeuralNet{
	void (*add_linear_layer)(char[], int, int);
	void (*forward_pass)(float*, float*);
	void (*backward_pass)(float*);
	void (*clean_up)();
}NeuralNet;

extern struct Node* head; // global variable - pointer to head node.

//Creates a new Node and returns pointer to it. 
struct Node* GetNewNode(char[], int, int);

//Inserts a Node at head of doubly linked list
void InsertAtHead(char[], int, int);

//Inserts a Node at tail of Doubly linked list
void InsertAtTail(char[], int, int);

NeuralNet createNetwork();

void Forward(float*, float*);

void Delete();

void Backward(float*);

#endif //_NN_H