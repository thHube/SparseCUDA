/**
 * @file   gpu_operations.h
 * @author Alberto Franco
 * @date   07/05/2010
 * 
 * Contiene i prototipi delle funzioni che vanno ad 
 * agire sulla GPU e le strutture per agire con tali 
 * funzioni.
 */
#ifndef GPU_OPERATIONS__H__
#define GPU_OPERATIONS__H__

// Hack per la portabilità tra C e C++
#ifdef __cplusplus
extern "C" {
#endif

// Inclusioni principali
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// inclusione principale per CUDA
#include "cuda.h"



//////////////////////////////////////////////////////////////////////////
/// STRUTTURE DATI UTILI

/**
 * La struttura matrice rappresenta il modo più semplice di
 * interagire tra la GPU e la CPU. E' composta dal suo ordine,
 * i suoi dati e il sotto insieme degli indici.
 */
typedef struct
{
   
   // Ordine della matrice
   int order;
   
   // Un array con i valori della matrice
   double* data;
   
   // Il sotto insieme di indici contenuto nella matrice. 
   int* indexSubset;
   
   // La lunghezza dell'array contenente il sotto insieme degli indici.
   int subsetLength;
        
}Matrix, *Matrix_t;

/**
 * La struttura che rappresenta una matrice sparsa, per semplicità d'uso
 * verranno dichiarate alcune funzioni che permettono di manipolarla.
 */
typedef struct
{
   // L'ordine della matrice in questione
   int order;   
   
   // L'array degli offset di riga
   int* rowOffset;
   
   // L'array degli indici di colonna
   int* columnIndex;
   
   // I valori non nulli della matrice.
   double* values;
   
}Sparse, *Sparse_t;

/**
 * Lo stack delle matrici che verrà usato per effettuare le operazioni
 * di fattorizzazione sparsa.
 */
typedef struct
{
   // L'array che contiene tutte le matrici di aggiornamento
   Matrix_t* stack;
   
   // La dimensione dello stack
   int size;
   
}UpdateMatrixStack, *UpdateMatrixStack_t;



//////////////////////////////////////////////////////////////////////////
/// FUNZIONI DI MANIPOLAZIONE DELLE MATRICI

/**
 * Esegue la fattorizzazione frontale della matrice.
 * @param frontalMatrix La matrice da fattorizzare
 * @param sparseFactor Il fattore di Cholesky da aggiornare
 * @return La matrice di aggiornamento i-esima
 */
Matrix_t frontalFactor(Matrix_t frontalMatrix, Matrix_t sparseFactor, double* cudaMemory);

/**
 * ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
 * ! Questa versione è molto più efficente dell'altra      ! 
 * ! extendAddfirst probabilemente si gestisce meglio qlc. ! 
 * ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !  
 * Somma estende le due matrici passate come parametro
 * @param first la prima matrice da somma estendere 
 * @param second la seconda matrice da somma estendere 
 * @return la matrice somma estesa
 */
Matrix_t extendAdd_cuda(Matrix_t first, Matrix_t second);

/**
 * Somma estende le due matrici e salva il risultato 
 * nella prima.  
 * @param first la prima matrice da somma estendere in cui
 *        verrà salvato il risultato.
 * @param second la seconda matrice da somma estendere 
 */
void extendAddFirst(Matrix_t first, Matrix_t second);

/**
 * Moltiplicazione tra matrici.
 * @param first La prima matrice da moltiplicare
 * @param second La seconda matrice da moltiplicare
 * @return la matrice moltiplicata
 */
Matrix_t multipy(Matrix_t first, Matrix_t second);



//////////////////////////////////////////////////////////////////////////
/// MANIPOLAZIONE DELLO STACK

/**
 * Alloca un nuovo stack per le matrici di aggiornamento.
 * @param nodeCount Il numero di nodi della foresta di eliminazione
 */
UpdateMatrixStack_t allocate(int nodeCount);

/**
 * Inserisce una matrice all'interno dello stack in posizione
 * i-esima. 
 * @param stack Lo stack su cui inserire la matrice
 * @param updateMatrix La matrice di aggiornamento da inserire
 * @param childId La posizione a cui inserire la matrice
 */
inline int push(UpdateMatrixStack_t stack, Matrix_t updateMatrix, int childId)
{
   // Copio i dati dalla matrice allo stack
   stack->stack[childId] = updateMatrix;
   
   // Ritorno true come successo dell'operazione
   return true;
}

/**
 * Recupera la matrice di aggiornamento corrispondente
 * @param nodeId l'identificativo del nodo a cui è associata la matrice.
 */
inline Matrix_t pop(UpdateMatrixStack_t stack, int nodeId)
{
   // Ritorno il riferimento diretto allo stack in modo da evitare 
   // costose operazioni di memoria.
   if(nodeId < stack->size)
   {
      return stack->stack[nodeId];
   }
   else
   {
      // Se cerco di accedere a zone di memoria non allocate,
      // ritorno un puntatore nullo.
      return NULL;
   }
}
/**
 * Libera tutta la memoria allocata per lo stack di matrici di 
 * aggiornamento e lo stack stesso.
 * @param stack Lo stack da liberare
 */
void freeStack(UpdateMatrixStack_t stack);

//////////////////////////////////////////////////////////////////////////
/// FUNZIONI VARIE

/**
 * Alloca memoria sulla scheda e ritorno il puntatore a tale memoria.
 * @param matrixOrder L'ordine della matrice da fattorizzare.
 */
double* allocateMemoryCuda(int matrixOrder);

/**
 * Funzione di test della funzionalità di computazione
 * sulla GPU.
 */
int testGpuComputation(); 

#define extendAdd extendAdd_cuda

#ifdef __cplusplus
}

inline void printMatrix(Matrix_t matrix)
{
   for(int i = 0; i < matrix->order; i++)
   {
      printf("[");
      for(int j = 0 ; j < matrix->order; j++)
      {
         printf("%g,", matrix->data[matrix->order * i + j]);
      }
      printf("]\n");
   }
   printf("\n");
}

#endif

#endif // GPU_OPERATIONS__H__
