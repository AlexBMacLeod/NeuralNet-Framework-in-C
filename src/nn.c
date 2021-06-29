/* Doubly Linked List implementation */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


#include "../include/nn.h"
#include "../include/common.h"
#include "../include/shape.h"

struct Node* head; // global variable - pointer to head node.

//Creates a new Node and returns pointer to it. 
struct Node* GetNewNode(char activation[], int in, int out, int batch_size) {
	struct Node* newNode
		= (struct Node*)malloc(sizeof(struct Node));
	struct Shape input={.n=1, .x=in, .y=1, .z=1};
	newNode->layer = createLayer(activation, input, out, batch_size);
	strncpy(newNode->layerType, "linear", 20);
	newNode->prev = NULL;
	newNode->next = NULL;
	return newNode;
}

//Convolutional node
struct Node* GetConvNode(char activation[], struct Shape in, int in_channels, int out_channels, int stride, int kernel_size, bool padding) {
	struct Node* newNode
		= (struct Node*)malloc(sizeof(struct Node));
	newNode->convLayer = createConv2DLayer(activation, in, stride, in_channels, out_channels, kernel_size, padding);
	strncpy(newNode->layerType, "conv2d", 20);
	newNode->prev = NULL;
	newNode->next = NULL;
	return newNode;
}

//With this one we set up an input layer, we define the learning rate
//and then set up a vector for the input to be placed into
struct Node* GetFirstNode(float lr, struct Shape output) {
	struct Node* newNode
		= (struct Node*)malloc(sizeof(struct Node));
	struct Shape input={.n=1, .x=1, .y=1, .z=1};
	int out = output.x * output.y * output.z;
	int batch_size = output.n;
	newNode->layer = createLayer("none", input, out, batch_size);
	newNode->layer->output->shape =output;
	strncpy(newNode->layerType, "linear", 20);
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
void InsertFirst(float lr, struct Shape in) {
	struct Node* newNode = GetFirstNode(lr, in);
	if(head == NULL) {
		head = newNode;
		return;
	}
}
//So then head is the first
//Inserts a Node at head of doubly linked list
void InsertAtHead(char activation[], int out) {
	struct Node* newNode;
	if(strncmp(head->layerType, "linear", 7)==0){
		newNode = GetNewNode(activation, head->layer->out, out, head->layer->batch_size);
	}else if(strncmp(head->layerType, "conv2d", 7)==0){
		int flatten = head->convLayer->out.x*head->convLayer->out.y*head->convLayer->out.z;
		newNode = GetNewNode(activation, flatten, out, head->convLayer->batch_size);
	}
	head->prev = newNode;
	newNode->next = head;
	if(strncmp(head->layerType, "linear", 7)==0){
		newNode->layer->lr = head->layer->lr; 
		newNode->layer->input = head->layer->output;
		head->layer->nextDelta = newNode->layer->delta;
		head->layer->nextWeights = newNode->layer->weights;
	}else if(strncmp(head->layerType, "conv2d", 7)==0){
		newNode->layer->lr = head->convLayer->lr;
		newNode->layer->input = head->convLayer->output;
		head->convLayer->nextDelta = newNode->layer->delta;
		head->convLayer->nextWeights = newNode->layer->weights;
	}
	head = newNode;
}

void InsertC2DAtHead(char activation[], int in_channels, int out_channels, int stride, int kernel_size, bool padding)
{
	struct Node* newNode;
	if(strncmp(head->layerType, "linear", 7)==0){
		newNode = GetConvNode(activation, head->layer->output->shape, in_channels, out_channels, stride, kernel_size, padding);
	}else if(strncmp(head->layerType, "conv2d", 7)==0){
		newNode = GetConvNode(activation, head->convLayer->out, in_channels, out_channels, stride, kernel_size, padding);
	}
	head->prev = newNode;
	newNode->next = head;
		if(strncmp(head->layerType, "linear", 7)==0){
		newNode->convLayer->lr = head->layer->lr; 
		newNode->convLayer->input = head->layer->output;
		head->layer->nextDelta = newNode->convLayer->delta;
		head->layer->nextWeights = newNode->convLayer->kernels;
	}else if(strncmp(head->layerType, "conv2d", 7)==0){
		newNode->convLayer->lr = head->convLayer->lr;
		newNode->convLayer->input = head->convLayer->output;
		head->convLayer->nextDelta = newNode->convLayer->delta;
		head->convLayer->nextWeights = newNode->convLayer->kernels;
	}
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



void Backward(float *y) {
	struct Node* temp = head;
	while(temp->next != NULL) {
		if(strncmp(temp->layerType, "linear", 7)==0){
        	temp->layer->backward_delta(temp->layer, y);
		}else if(strncmp(temp->layerType, "conv2d", 7)==0){
			temp->convLayer->backward_delta(temp->convLayer, y);
		}
		temp = temp->next;
	}
	temp = head;
	while(temp->next != NULL) {
		if(strncmp(temp->layerType, "linear", 7)==0){
        	temp->layer->backward_weights(temp->layer);
		}else if(strncmp(temp->layerType, "conv2d", 7)==0){
			temp->convLayer->backward_weights(temp->convLayer);
		}
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
		if(strncmp(temp->layerType, "linear", 7)==0){
        	temp->layer->forward_pass(temp->layer);
		}else if(strncmp(temp->layerType, "conv2d", 7)==0){
			temp->convLayer->forward_pass(temp->convLayer);
		}
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
		if(strncmp(temp->layerType, "linear", 7)==0){
        	temp->layer->free_layer(temp->layer);
		}else if(strncmp(temp->layerType, "conv2d", 7)==0){
			temp->convLayer->free_layer(temp->convLayer);
		}
        free(temp);
        temp = prev;
	}
}


NeuralNet createNetwork(float lr, int batch_size, int x, int y, int channels)
{
	NeuralNet nn;
	struct Shape in = {.n=batch_size, .x=x, .y=y, .z=channels};
	InsertFirst(lr, in);
	nn.add_linear_layer = InsertAtHead;
	nn.add_convolutional_layer = InsertC2DAtHead;
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