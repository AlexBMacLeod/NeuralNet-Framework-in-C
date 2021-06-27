#ifndef _NN_H
#define _NN_H
#include <stdbool.h>

#include "linear.h"
#include "conv2d.h"

struct Node  {
	struct Node* next;
	struct Node* prev;
	char layerType[20];
    LinearLayer *layer;
	conv2DLayer *convLayer;
};

typedef struct NeuralNet{
	void (*add_linear_layer)(char[], int);
	void (*add_convolutional_layer)(char[], int, int, int);
	void (*forward_pass)(float*, float*);
	void (*backward_pass)(float*);
	void (*clean_up)();
}NeuralNet;

extern struct Node* head; // global variable - pointer to head node.

//Creates a new Node and returns pointer to it. 
struct Node* GetNewNode(char[], int, int, int);

struct Node* GetFirstNode(float, struct Shape);

//Inserts a Node at head of doubly linked list
void InsertAtHead(char[], int);

//Inserts a node 
void InsertC2DAtHead(char activation[], int, int, int, int, bool);

void InsertFirst(float, struct Shape);

//Inserts a Node at tail of Doubly linked list
void InsertAtTail(char[], int, int, int);

NeuralNet createNetwork(float, int, int, int, int);

void Forward(float*, float*);

void Delete();

void Backward(float*);

int validation_run(float*, float*, int, int, NeuralNet);

#endif //_NN_H