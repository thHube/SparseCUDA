/**
 * @file   stack_management.c
 * @author Alberto Franco
 * @date   12/05/2010
 * 
 * Contiene l'implementazione delle funzionalità di manipolazione
 * dello stack delle matrici di aggiornamento.
 */
#include "gpu_operations.h"
 
/**
 * Alloca un nuovo stack per le matrici di aggiornamento.
 * @param nodeCount Il numero di nodi della foresta di eliminazione
 */
UpdateMatrixStack_t allocate(int nodeCount)
{
   // Utilizzo un puntatore di supporto
   UpdateMatrixStack_t new_allocated_stack;
   
   // Richiedo memoria sullo heap
   new_allocated_stack = (UpdateMatrixStack_t)malloc(sizeof(UpdateMatrixStack));
   
   // Assegno il valore della dimensione
   new_allocated_stack->size = nodeCount;
   
   // Richiedo memoria per salvare le matrici
   new_allocated_stack->stack = (Matrix_t*)malloc(sizeof(Matrix_t) * nodeCount);
   
   // Ritorno il nuovo stack allocato
   return new_allocated_stack;
}

/**
 * Libera tutta la memoria allocata per lo stack di matrici di 
 * aggiornamento e lo stack stesso.
 * @param stack Lo stack da liberare
 */
void freeStack(UpdateMatrixStack_t stack)
{
   // Scorro tutto lo stack per vedere se posso deallocare
   for(int i = 0; i < stack->size; i++)
   {
      // Controllo che la matrice sia allocata
      if(stack->stack[i] != NULL)
      {
         // Se la matrice è allocata libero la memoria
         free(stack->stack[i]->data);
         free(stack->stack[i]->indexSubset);
         free(stack->stack[i]);
         
         // Assegno a null il puntatore per coerenza
         stack->stack[i] = NULL;
      }
   }
   
   // Infine dealloco lo stack
   free(stack);
}
