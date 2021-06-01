#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>



#include "../include/nn.h"

struct node *head = NULL;
struct node *last = NULL;
struct node *current = NULL;


void nn_delete()
{
   /* deref head_ref to get the real head */
   //struct node* current;
   struct node* next;

   current = head;

   //last->layer->input->freeMem(current->layer->input);

   while (current != NULL) 
   {
       next = current->next;
       if(next==NULL) current->layer->input->freeMem(current->layer->input);
       current->layer->free_layer(current->layer);
       free(current);
       current = next;
   }
   
   /* deref head_ref to affect the real head back
      in the caller. */
   head = NULL;
}

bool isEmpty() {
   return head == NULL;
}

int length() {
   int length = 0;
   struct node *current;
	
   for(current = head; current != NULL; current = current->next){
      length++;
   }
	
   return length;
}
/*
//display the list in from first to last
void displayForward() {

   //start from the beginning
   struct node *ptr = head;
	
   //navigate till the end of the list
   printf("\n[ ");
	
   while(ptr != NULL) 
   {
        for(int i=0;i<COLS;i++)
        {         
            printf("(%d,%d) ",ptr->key,ptr->data);
        }
        ptr = ptr->next;
   }
	
   printf(" ]");
}

//display the list from last to first
void displayBackward() {

   //start from the last
   struct node *ptr = last;
	
   //navigate till the start of the list
   printf("\n[ ");
	
   while(ptr != NULL) {    
	
      //print data
        for(int i=0;i<COLS;i++)  
        {         
            printf("(%d,%d) ",ptr->key,ptr->data);
        }
		
        //move to next item
        ptr = ptr ->prev;
      
   }
	
}
*/
//insert link at the first location
void nn_linear(char activation[], int in, int out) {

   //create a link
   struct node *link = (struct node*) malloc(sizeof(struct node));

   link->layer = createLayer(activation, in, out);
	
   if(isEmpty()) {
      //make it the last link
      link->layer->input = createMatrix(in, 1);
      last = link;
   } else {
      //update first prev link

      link->layer->input = head->layer->output;
      head->layer->nextDelta = link->layer->delta;
      head->layer->nextWeights = link->layer->weights;
      head->prev = link;
   }

   //point it to old first link
   link->next = head;
	
   //point first to new first link
   head = link;
}

//insert link at the last location
void net_add_layer(char activation[], int in, int out) {

   //create a link
   struct node *link = (struct node*) malloc(sizeof(struct node));

   link->layer = createLayer(activation, in, out);
	
   if(isEmpty()) {
      //make it the last link
      //link->layer->input = createMatrix(in, 1);
      head = link;
      //last = link;
   } else {
      //make link a new last link
      //last->layer->nextDelta = link->layer->delta;
      //last->layer->nextWeights = link->layer->weights;
      //link->layer->input = last->layer->output;
      last->next = link;     
      
      //mark old last node as prev of new link
      link->prev = last;
   }

   //point last to new last node
   last = link;
}


void nn_forward(float* in, float *out)
{
   //struct node* next;

   current = head;
   memmove(current->layer->input->data, in, sizeof(float)*5);
   //current->layer->input->data = in;
   while (current != NULL) 
   {
      current->layer->forward_pass(current->layer);
      //if(current->next==NULL) memmove(out, current->layer->output->data, sizeof(float)*5);
      current = current->next;
   }
   
}


void nn_backward(float* in)
{
   sleep(0);
}