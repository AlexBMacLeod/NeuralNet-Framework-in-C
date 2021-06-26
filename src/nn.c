/* Doubly Linked List implementation */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../include/nn.h"
#include "../include/common.h"

struct Node* head; // global variable - pointer to head node.

//Creates a new Node and returns pointer to it. 
struct Node* GetNewNode(char activation[], int in, int out, int batch_size) {
	struct Node* newNode
		= (struct Node*)malloc(sizeof(struct Node));
	newNode->layer = createLayer(activation, in, out, batch_size);
	newNode->prev = NULL;
	newNode->next = NULL;
	return newNode;
}

//With this one we set up an input layer, we define the learning rate
//and then set up a vector for the input to be placed into
struct Node* GetFirstNode(float lr, int out, int batch_size) {
	struct Node* newNode
		= (struct Node*)malloc(sizeof(struct Node));
	newNode->layer = createLayer("none", 1, out, batch_size);
	newNode->prev = NULL;
	newNode->next = NULL;
	newNode->layer->lr = lr;
	return newNode;
}

//This creates the actual input layer
//The main goal with differentiating this is two fold,
//by defining the input layer like this I can use every previous layers output as the dimension
//for the new layers input. Meaning that looking forward a little bit convolutional layers can be flattened
//and have their flattened dimensions used as input for the linear layers
//The other idea is to use this to introduce other things such as optimizers
void InsertFirst(float lr, int out, int batch_size) {
	struct Node* newNode = GetFirstNode(lr, out, batch_size);
	if(head == NULL) {
		head = newNode;
		return;
	}
}
//So then head is the first
//Inserts a Node at head of doubly linked list
void InsertAtHead(char activation[], int out) {
	struct Node* newNode = GetNewNode(activation, head->layer->out, out, head->layer->batch_size);
	head->prev = newNode;
	newNode->next = head;
	newNode->layer->lr = head->layer->lr; 
    newNode->layer->input = head->layer->output;
    head->layer->nextDelta = newNode->layer->delta;
    head->layer->nextWeights = newNode->layer->weights;
	head = newNode;
}


//Inserts a Node at tail of Doubly linked list
void InsertAtTail(char activation[], int in, int out, int batch_size) {
	struct Node* temp = head;
	struct Node* newNode = GetNewNode(activation, in, out, batch_size);
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
	while(temp->next != NULL) {
		temp->layer->backward_delta(temp->layer, y);
		temp = temp->next;
	}
	temp = head;
	while(temp->next != NULL) {
		temp->layer->backward_weights(temp->layer);
		temp = temp->next;
	}
}


void Forward(float *input, float *output) {
	struct Node* temp = head;
	if(temp == NULL) {return; printf("error in forward pass");} 
	// Going to last Node
	while(temp->next != NULL) {
		temp = temp->next;
	}
	// Traversing backward using prev pointer
	memmove(temp->layer->output->data, input, sizeof(float)*temp->layer->out*temp->layer->batch_size);

	temp = temp->prev;
	while(temp != NULL) {
        temp->layer->forward_pass(temp->layer);
        if(temp->prev==NULL){ 
			memmove(output, temp->layer->output->data, sizeof(float)*temp->layer->out*temp->layer->batch_size);
		}
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
	while(temp != NULL) {
		prev = temp->prev;
        //if(prev==NULL) temp->layer->input->freeMem(temp->layer->input);
        temp->layer->free_layer(temp->layer);
        free(temp);
        temp = prev;
	}
}


NeuralNet createNetwork(float lr, int in, int batch_size)
{
	NeuralNet nn;
	InsertFirst(lr, in, batch_size);
	nn.add_linear_layer = InsertAtHead;
	nn.backward_pass = Backward;
	nn.forward_pass = Forward;
	nn.clean_up = Delete;
	return nn;
}

int validation_run(float *train, float *train_labels, int len, int batch_size, NeuralNet nn)
{
	int correct_cnt = 0;

	float *in = calloc(784*batch_size, sizeof(float));
	float *y_hat = calloc(10*batch_size, sizeof(float));
	float *y = calloc(10*batch_size, sizeof(float));
	for(int i=0; i<floor(len/batch_size); i++)
	{
		memmove(in, (train+(i*784*batch_size)), sizeof(float)*784*batch_size); 
		memmove(y, (train_labels+(i*10*batch_size)), sizeof(float)*10*batch_size);
        nn.forward_pass(in, y_hat);
        correct_cnt += argmax_batch(y_hat, y, 10, batch_size);
	}
	free_all(y, in, y_hat);
	return correct_cnt;
}