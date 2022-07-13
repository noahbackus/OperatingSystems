#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dlist.nbackus.h"

int insertState(StateDListNode **list, char *name, int population);
int findState(StateDListNode *list, char *name, int *population);
int deleteState(StateDListNode **list, char *name);
int printList(StateDListNode *list);

int insertState(StateDListNode **list, char *name, int population) {
  // Check name length
  if (strlen(name) > MAX_STATE_NAME_LENGTH) {
    return 1;
  }

  // Check to see if node exists already
  int dummyPop = 0;
  if (findState(*list, name, &dummyPop) == 0) {
    return 1;
  }

  // Create struct and malloc it
  StateDListNode *newNode;
  newNode = (StateDListNode *) malloc(sizeof(StateDListNode));

  // Add data to struct
  strcpy(newNode->name, name);
  newNode->population = population;
  newNode->next = NULL;
  newNode->prev = NULL;

  // Check it see if list is empty
  if (*list == NULL) {
    *list = newNode;
    return 0;
  }

  // Create current node
  StateDListNode *current = *list;

  // Check if smaller than first item
  if (newNode->population < current->population) {
    newNode->next = current;
    current->prev = newNode;
    *list = newNode;
    return 0;
  }

  // Loop through list to find correct spot for population
  while (current->next != NULL) {

    // Check population of node infront of current
    if (!(current->next->population < population)) {

      // Insert new node, population is smaller than the next item in the list
      // Set newNode next to current next
      newNode->next = current->next;
      newNode->prev = current;

      // Set current next to newNode
      current->next = newNode;

      // Return
      return 0;
    }

    // Set current equal to next value
    current = current->next;
  }

  // We have hit the end of the list, newNode is the largest
  current->next = newNode;
  newNode->prev = current;
  return 0;

}

int deleteState(StateDListNode **list, char *name) {

  // Check if list is empty
  if (*list == NULL) {
    return 1;
  }

  // Create current and prev node
  StateDListNode *current = *list;

  // Loop through list
  while (current != NULL) {

    // Check name
    if (strcmp(current->name, name) == 0) {

      // Check if prev is NULL
      if (current->prev == NULL) {
        // Check if there is a next
        if (current->next == NULL) {
            *list = NULL;
            free(current);
            return 0;
        }
        current->next->prev = NULL;
        *list = current->next;
        free(current);
        return 0;
      } else if (current->next == NULL) {
        current->prev->next = NULL;
        free(current);
        return 0;
      } else {
        // Set prev next to current next and free current
        //prev->next = current->next;
        current->prev->next = current->next;
        current->next->prev = current->prev;
        free(current);
        return 0;
      }
    }

    // Set current
    current = current->next;

  }

  // Not found
  return 1;

}

int printList(StateDListNode *list) {

  // Check if list is empty
  if(list == NULL) {
    printf("List is empty\n");
  }

  // Create current
  StateDListNode *current = list;

  // Loop through list
  while (current != NULL) {

    // Print data
    printf("|%s| %d\n", current->name, current->population);

    // Go to next node
    current = current->next;
  }

  return 0;

}

int findState(StateDListNode *list, char *name, int *population) {

  // Check if list is empty
  if (list == NULL) {
    return 1;
  }

  // Create current node
  StateDListNode *current = list;

  // Loop through list 
  while (current != NULL) {
    // Check to see if names are the same
    if(strcmp(name, current->name) == 0) {
      *population = current->population;
      return 0;
    }

    // Set current to next node
    current = current->next;
  }

  // Nothing matched
  return 1;

}


// int main() {
//   StateDListNode *theList = NULL;
//   char name[MAX_STATE_NAME_LENGTH];
//   int population;
//   int rc;
 
//   strcpy(name, "Kentucky"); 
//   rc = insertState(&theList, name, 4468402);
//   if (rc == 0)
//     printf("inserted %s\n", name);
//   else
//     printf("failed to insert %s\n", name);

//   strcpy(name, "Vermont"); 
//   rc = insertState(&theList, name, 626299);
//   if (rc == 0)
//     printf("inserted %s\n", name);
//   else
//     printf("failed to insert %s\n", name);
 
//   strcpy(name, "Iowa"); 
//   rc = insertState(&theList, name, 3156145);
//   if (rc == 0)
//     printf("inserted %s\n", name);
//   else
//     printf("failed to insert %s\n", name);
 
//   strcpy(name, "New Mexico"); 
//   rc = insertState(&theList, name, 2095428);
//   if (rc == 0)
//     printf("inserted %s\n", name);
//   else
//     printf("failed to insert %s\n", name);
 
//   strcpy(name, "Delaware"); 
//   rc = insertState(&theList, name, 967171);
//   if (rc == 0)
//     printf("inserted %s\n", name);
//   else
//     printf("failed to insert %s\n", name);
 
//   strcpy(name, "Vermont"); 
//   rc = insertState(&theList, name, 626299);
//   if (rc == 0)
//     printf("inserted %s\n", name);
//   else
//     printf("failed to insert %s\n", name);
 
//   strcpy(name, "Utah"); 
//   rc = insertState(&theList, name, 3161105);
//   if (rc == 0)
//     printf("inserted %s\n", name);
//   else
//     printf("failed to insert %s\n", name);

//   printList(theList);

//   //---------------------------------------------------------------

//   strcpy(name, "Utah");
//   rc = findState(theList, name, &population);
//   if (rc == 0)
//     printf("found %s: population = %d\n", name, population);
//   else
//     printf("did not find state '%s'\n", name);

//   strcpy(name, "Kansas");
//   rc = findState(theList, name, &population);
//   if (rc == 0)
//     printf("found %s: population = %d\n", name, population);
//   else
//     printf("did not find state '%s'\n", name);

//   strcpy(name, "Vermont");
//   rc = findState(theList, name, &population);
//   if (rc == 0)
//     printf("found %s: population = %d\n", name, population);
//   else
//     printf("did not find state '%s'\n", name);

//   strcpy(name, "Alabama");
//   rc = findState(theList, name, &population);
//   if (rc == 0)
//     printf("found %s: population = %d\n", name, population);
//   else
//     printf("did not find state '%s'\n", name);

//   strcpy(name, "New Mexico");
//   rc = findState(theList, name, &population);
//   if (rc == 0)
//     printf("found %s: population = %d\n", name, population);
//   else
//     printf("did not find state '%s'\n", name);

//   printList(theList);

//   //---------------------------------------------------------------

//   strcpy(name, "Utah");
//   rc = deleteState(&theList, name);
//   if (rc == 0)
//     printf("deleted '%s'\n", name);
//   else
//     printf("failed to delete '%s'\n", name);

//   strcpy(name, "Vermont");
//   rc = deleteState(&theList, name);
//   if (rc == 0)
//     printf("deleted '%s'\n", name);
//   else
//     printf("failed to delete '%s'\n", name);

//   strcpy(name, "Nevada");
//   rc = deleteState(&theList, name);
//   if (rc == 0)
//     printf("deleted '%s'\n", name);
//   else
//     printf("failed to delete '%s'\n", name);

//   strcpy(name, "Hawaii"); 
//   rc = insertState(&theList, name, 1420491);
//   if (rc == 0)
//     printf("inserted %s\n", name);
//   else
//     printf("failed to insert %s\n", name);

//   strcpy(name, "Utah");
//   rc = deleteState(&theList, name);
//   if (rc == 0)
//     printf("deleted '%s'\n", name);
//   else
//     printf("failed to delete '%s'\n", name);

//   strcpy(name, "Iowa");
//   rc = findState(theList, name, &population);
//   if (rc == 0)
//     printf("found %s: population = %d\n", name, population);
//   else
//     printf("did not find state '%s'\n", name);

//   strcpy(name, "New Mexico");
//   rc = deleteState(&theList, name);
//   if (rc == 0)
//     printf("deleted '%s'\n", name);
//   else
//     printf("failed to delete '%s'\n", name);

//   strcpy(name, "Delaware");
//   rc = deleteState(&theList, name);
//   if (rc == 0)
//     printf("deleted '%s'\n", name);
//   else
//     printf("failed to delete '%s'\n", name);


//   printList(theList);

//   strcpy(name, "Kentucky");
//   rc = deleteState(&theList, name);
//   if (rc == 0)
//     printf("deleted '%s'\n", name);
//   else
//     printf("failed to delete '%s'\n", name);

//   printList(theList);

//   strcpy(name, "Hawaii");
//   rc = deleteState(&theList, name);
//   if (rc == 0)
//     printf("deleted '%s'\n", name);
//   else
//     printf("failed to delete '%s'\n", name);

//   strcpy(name, "Iowa");
//   rc = deleteState(&theList, name);
//   if (rc == 0)
//     printf("deleted '%s'\n", name);
//   else
//     printf("failed to delete '%s'\n", name);

//   printList(theList);

//   strcpy(name, "Vermont");
//   rc = findState(theList, name, &population);
//   if (rc == 0)
//     printf("found %s: population = %d\n", name, population);
//   else
//     printf("did not find state '%s'\n", name);

//   return 0;
// }
