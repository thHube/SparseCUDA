/**
 * @file   extend_add.cpp
 * @author Alberto Franco
 * @date   14/05/2010
 * 
 * Implementazione della funzionalità di somma estesa
 * delle matrici utilizzando la CPU.
 */
#include "gpu_operations.h"

/**
 * Calcola il merge dei sottoinsiemi di indici delle due matrici passate come
 * parametro.
 */
inline int calculateNewSubset(Matrix_t first, Matrix_t second, int* newSubset)
{
   // Alcune variabili di supporto
   int newLength = 0;
   int i = 0; 
   int j = 0;
   bool a_finish = false;
   bool b_finish = false;;
   
   while(i < first->subsetLength || j < second->subsetLength)
   {
      // Se l'i-esimo valore del primo è minore o uguale al j-esimo
      // valore del secondo:
      if( b_finish || 
         !a_finish &&  first->indexSubset[i] <= second->indexSubset[j])
      {
         // Inserisco il valore all'interno del nuovo insieme
         newSubset[newLength] = first->indexSubset[i];
         // Incremento i valori
         i++; newLength++;
         
         // Se il valore è uguale 
         if(first->indexSubset[i - 1] == second->indexSubset[j])
         {
            // Incremento anche il valore di j
            j++;
         }
      }
      // Altrimenti è l'altro valore quello da inserire.
      else 
      {
         // Inserisco il valore nel nuovo insieme
         newSubset[newLength] = second->indexSubset[j];
         // Incremento gli indici
         j++; newLength++;
      }
      
      // Conrtollo che abbiamo terminato di controllare i valori 
      b_finish = (j == second->subsetLength)? true: false;
      a_finish = (i == first->subsetLength)? true: false;
   }
   
   return newLength;
}

/**
 * Somma estende le due matrici passate come parametro
 * @param first la prima matrice da somma estendere 
 * @param second la seconda matrice da somma estendere 
 * @return la matrice somma estesa
 */
Matrix_t extendAdd_cpu(Matrix_t first, Matrix_t second)
{
   // Alloco la memoria per la matrice
   Matrix_t extended = Allocator<Matrix>::get().allocate();
   
   // Calcolo la dimensione di lato massima che la matrice può avere
   int max_size = first->subsetLength + second->subsetLength;
   
   // Alloco il subset di indici corrispondente.
   extended->indexSubset = Allocator<int>::get().allocate(max_size);
   
   // L'allocatore si occupa di mettere a zero tutta la memoria
   // posso andare a scrivere. Calcolo il nuovo insieme di indici
   extended->order = calculateNewSubset(first, second, extended->indexSubset);
   
   // Alloco la matrice che mi serve
   extended->data = Allocator<double>::get().allocate(extended->order * 
                                                      extended->order);
   
   // vado a sommare sulla memoria appena allocata
   for(int i = 0, j = 0; i < extended->subsetLength; i++)
   {
      // Se l'elemento è presente nel sotto insieme di first
      if(first->indexSubset[j] == extended->indexSubset[i])
      {
         // Sommo tutti gli elementi della riga 
         for(int k = 0; k < first->subsetLength; k++)
         {
            
         }
         // Incremento j per il prossimo elemento
         j += 1;
      }
   }
   
   return extended;   
}
