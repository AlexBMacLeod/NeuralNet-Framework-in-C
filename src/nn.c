#include <stdio.h>
#include <string.h>
#include <stdlib.h>




#include "../include/nn.h"

struct node *head = NULL;
struct node *last = NULL;
struct node *current = NULL;


void deleteNetwork()
{
   /* deref head_ref to get the real head */
   //struct node* current;
   struct node* next;

   current = head;
   while (current != NULL) 
   {
       next = current->next;
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
void insertLayerFirst(char activation[], int in, int out) {

   //create a link
   struct node *link = (struct node*) malloc(sizeof(struct node));

   link->layer = createLayer(activation, in, out);
	
   if(isEmpty()) {
      //make it the last link
      last = link;
   } else {
      //update first prev link
      head->prev = link;
   }

   //point it to old first link
   link->next = head;
	
   //point first to new first link
   head = link;
}

//insert link at the last location
void insertLayerLast(char activation[], int in, int out) {

   //create a link
   struct node *link = (struct node*) malloc(sizeof(struct node));

   link->layer = createLayer(activation, in, out);
	
   if(isEmpty()) {
      //make it the last link
      last = link;
   } else {
      //make link a new last link
      last->layer->nextDelta = link->layer->delta;
      last->layer->nextWeights = link->layer->weights;
      last->next = link;     
      
      //mark old last node as prev of new link
      link->prev = last;
   }

   //point last to new last node
   last = link;
}


float* network_forward(float* in)
{
   struct node* next;

   current = head;
   memcpy(current->layer->input->data, in, current->layer->in * sizeof(float));
   while (current != NULL) 
   {
       next = current->next;
       current->layer->forward_pass(current->layer);
   }
   

   return last->layer->output->data;
}


float* backward(float*)