/* Doubly Linked List implementation */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/nn.h"

struct Node* head; // global variable - pointer to head node.

//Creates a new Node and returns pointer to it. 
struct Node* GetNewNode(char activation[], int in, int out) {
	struct Node* newNode
		= (struct Node*)malloc(sizeof(struct Node));
	newNode->layer = createLayer(activation, in, out);
	newNode->prev = NULL;
	newNode->next = NULL;
	return newNode;
}
//So then head is the first
//Inserts a Node at head of doubly linked list
void InsertAtHead(char activation[], int in, int out) {
	struct Node* newNode = GetNewNode(activation, in, out);
	if(head == NULL) {
        newNode->layer->input = createMatrix(in, 1);
		head = newNode;
		return;
	}
	head->prev = newNode;
	newNode->next = head; 
    newNode->layer->input = head->layer->output;
    head->layer->nextDelta = newNode->layer->delta;
    head->layer->nextWeights = newNode->layer->weights;
	head->layer->nextOut = newNode->layer->out;
	head = newNode;
}

//Inserts a Node at tail of Doubly linked list
void InsertAtTail(char activation[], int in, int out) {
	struct Node* temp = head;
	struct Node* newNode = GetNewNode(activation, in, out);
	if(head == NULL) {
		head = newNode;
		return;
	}
	while(temp->next != NULL) temp = temp->next; // Go To last Node
	temp->next = newNode;
	newNode->prev = temp;
}

//Prints all the elements in linked list in forward traversal order
void Backward(float *y) {
	struct Node* temp = head;
	while(temp != NULL) {
		temp->layer->backward_pass(temp->layer, y);
		temp = temp->next;
	}
}



void Forward(float *input, float *output) {
	struct Node* temp = head;
	if(temp == NULL) return; // empty list, exit
	// Going to last Node
	while(temp->next != NULL) {
		temp = temp->next;
	}
	// Traversing backward using prev pointer
	//memmove(temp->layer->input, input, sizeof(float)*temp->layer->in);
	for(int i=0; i<temp->layer->in; i++) temp->layer->input->data[i] = input[i];
	while(temp != NULL) {
        temp->layer->forward_pass(temp->layer);
        if(temp->prev==NULL) *output = *(temp->layer->output->data);
		temp = temp->prev;
	}
}

void Delete() {
    	struct Node* temp = head;
        struct Node* prev = temp;
	if(temp == NULL) return; // empty list, exit
	// Going to last Node
	while(temp->next != NULL) {
		temp = temp->next;
	}
	temp->layer->input->freeMem(temp->layer->input);
	while(temp != NULL) {
		prev = temp->prev;
        //if(prev==NULL) temp->layer->input->freeMem(temp->layer->input);
        temp->layer->free_layer(temp->layer);
        free(temp);
        temp = prev;
	}
}