#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>



#include "../include/nn.h"

void deleteList()
{
   /* deref head_ref to get the real head */
   struct node* current;
   struct node* next;

   current = head;
   while (current != NULL) 
   {
       next = current->next;
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

//insert link at the first location
void insertFirst(float data[COLS]) {

   //create a link
   struct node *link = (struct node*) malloc(sizeof(struct node));
   memcpy(link->row, data, sizeof(float)*30);
	
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
void insertLast(float data[COLS]) {

   //create a link
   struct node *link = (struct node*) malloc(sizeof(struct node));
   memcpy(link->row, data, sizeof(data));
	
   if(isEmpty()) {
      //make it the last link
      last = link;
   } else {
      //make link a new last link
      last->next = link;     
      
      //mark old last node as prev of new link
      link->prev = last;
   }

   //point last to new last node
   last = link;
}

//delete first item
struct node* deleteFirst() {

   //save reference to first link
   struct node *tempLink = head;
	
   //if only one link
   if(head->next == NULL){
      last = NULL;
   } else {
      head->next->prev = NULL;
   }
	
   head = head->next;
   //return the deleted link
   return tempLink;
}

//delete link at the last location

struct node* deleteLast() {
   //save reference to last link
   struct node *tempLink = last;
	
   //if only one link
   if(head->next == NULL) {
      head = NULL;
   } else {
      last->prev->next = NULL;
   }
	
   last = last->prev;
	
   //return the deleted link
   return tempLink;
}

