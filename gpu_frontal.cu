/**
 * @file   gpu_frontal.cu
 * @author Alberto Franco
 * @date   12/05/2010
 * 
 * Implementazione della funzionalità di fattorizzazione 
 * frontale sulla GPU.
 */
#include "gpu_operations.h"


// Kernel per la creazione della matrice di aggiornamento
__global__ void calcUpdateMatrix_kernel(double* frontal, 
                                        double* update , 
                                        double  inv_d,
                                        int     size,
                                        int     order  );

// Kernel per il calcolo del vettore dei valori di L
__global__ void calcCholeskyCoeff_kernel(double* frontal        ,
                                         double* cholesky_vector,
                                         double  sqrt_d,
                                         int     order          );

// Funzione per il salvataggio dei dati 
void saveCholeskyData(int*  indexSubset, int     subsetLength, 
                      Matrix_t cholesky, double* choleskyData);

/**
 * Esegue la fattorizzazione frontale della matrice.
 * @param frontalMatrix La matrice da fattorizzare
 * @param sparseFactor Il fattore di Cholesky da aggiornare
 * @return La matrice di aggiornamento i-esima
 */
Matrix_t frontalFactor(Matrix_t frontalMatrix, Matrix_t sparseFactor, double* cudaMemory)
{
   // Dimensione del blocco di thread
   int blockSize = 16;

   // Copio i dati in memoria video
   double* frontal_data;   
   double* update_data;
   double* cholesky_data;
   
   // Calcolo la dimensione della allocazione come il multiplo più 
   // vicino alla dimensione del blocco
   int size = frontalMatrix->order;
   int source_size = sizeof(double) * frontalMatrix->order * 
                                      frontalMatrix->order ;
   
   if(size % blockSize != 0)
   {
      // Il puiù vicino multiplo di blockSize
      size = blockSize * (size/blockSize + 1);
   }
   
   // Alloco la memoria per i dati della matrice frontale.
   cudaMalloc((void**)&frontal_data, source_size);
   // frontal_data = cudaMemory;
   
   // Alloco la memoria per i dati della matrice di aggiornamento che 
   // andrò a creare fattorizzando la matice 
   cudaMalloc((void**)&update_data, sizeof(double) * size * size);
   // update_data = frontal_data + (source_size / sizeof(double));
   
   // Alloco la memoria per i dati del fattore di cholesky
   cudaMalloc((void**)&cholesky_data, sizeof(double) * size);
   // cholesky_data = update_data + size * size;
   
   // Copio i dati della matrice in RAM video.
   cudaMemcpy(frontal_data, frontalMatrix->data, 
              source_size,  cudaMemcpyHostToDevice);
   
   // Calcolo prima i dati del fattore di cholesky
   int choleskyThreads = blockSize;
   int choleskyBlocks  = size / choleskyThreads;
   
   // Invoco il kernel per calcolare il fattore di cholesky
   calcCholeskyCoeff_kernel<<<choleskyBlocks, choleskyThreads>>> (
                                           frontal_data, 
                                           cholesky_data, 
                                           sqrt(frontalMatrix->data[0]),
                                           frontalMatrix->order);
                                           
   // Calcolo il numero di thread e blocchi da avviare
   dim3 updateThreads(blockSize, blockSize);
   dim3 updateBlocks(size / blockSize, size / blockSize);
   
   // Avvio il kernel e calcolo la matrice di aggiornamento
   calcUpdateMatrix_kernel<<<updateBlocks, updateThreads>>> (
                                           frontal_data,
                                           update_data,
                                           1 / frontalMatrix->data[0],
                                           size,
                                           frontalMatrix->order);
   
   // Recupero i dati dalla memoria, salvo i dati di cholesky  
   // nel fattore e poi restutiusco la matrice di aggiornamento
   {
      double* factor_data;
      int     factor_size = sizeof(double) * frontalMatrix->order;
      factor_data = (double*)malloc(factor_size);
      
      // Copio i dati dalla memoria video
      cudaMemcpy(factor_data, cholesky_data, factor_size, cudaMemcpyDeviceToHost);
      
      // Libero la memoria sulla scheda video
      cudaFree(cholesky_data);
      
      // Assegno il puntatore alla nuova area di memoria 
      cholesky_data = factor_data;
   }
   // Salvo i dati nel fattore
   saveCholeskyData(frontalMatrix->indexSubset, frontalMatrix->subsetLength, 
                    sparseFactor, cholesky_data);

   // Copio i dati della matrice di aggiornamento nella memoria centrale
   // prima alloco la matrice
   Matrix_t updateMatrix = (Matrix_t)malloc(sizeof(Matrix));
   
   // Modifico i valori della vecchia matrice
   updateMatrix->order = updateMatrix->subsetLength = frontalMatrix->order - 1;
   
   // Alloco gli array che mi servono per contenere i dati
   int allocSize = sizeof(double) * updateMatrix->order * updateMatrix->order;
   updateMatrix->data = (double*)malloc(allocSize);
   
   updateMatrix->indexSubset = (int*)malloc(updateMatrix->order * sizeof(int));
   
   // Copio la memoria dall'inseme degli indici di frontalMatrix
   memcpy(updateMatrix->indexSubset, frontalMatrix->indexSubset + 1,
          sizeof(int) * updateMatrix->subsetLength);
   
   // Copio la memoria dalla scheda video.
   for(int i = 1; i < frontalMatrix->order; i++)
   {
      // Copio la fine di ogni riga, il resto dei dati non mi interessa
      // li lascio stare e poi li cancello
      cudaMemcpy(updateMatrix->data + (i - 1) * updateMatrix->order,
                 update_data + (i * size) + 1, 
                 sizeof(double) * updateMatrix->order,
                 cudaMemcpyDeviceToHost);
   }
   
   // Libero la memoria sulla scheda grafica
   // cudaFree(frontal_data);
   // cudaFree(update_data);
   
   // E quella usata per copiare i dati di cholesky
   free(cholesky_data);
   
   // Ritorno la matrice di aggiornamento
   return updateMatrix;
}

/**
 * Alloca memoria sulla scheda e ritorno il puntatore a tale memoria.
 * @param matrixOrder L'ordine della matrice da fattorizzare.
 */
double* allocateMemoryCuda(int matrixOrder)
{
   // Dichiaro un puntatore che userò per gestire la memoria
   double* cudaMemoryArea;
   
   // Alloco memoria sulla scheda
   cudaMalloc((void**)&cudaMemoryArea, matrixOrder * matrixOrder * 3 * sizeof(double));
   
   // Ritorno il nuovo puntatore
   return cudaMemoryArea;
}


/**
 * Kernel per la creazione della matrice di aggiornamento.
 * @param frontal I dati della matrice frontale da fattorizzare
 * @param update  I dati della martice di aggiornamento
 * @param inv_d   L'inverso di d per velocizzare i calcoli
 * @param size    La dimensione della matrice allocata
 * @param order   La dimensione della matrice concettualmente
 */
__global__ void calcUpdateMatrix_kernel(double* frontal, 
                                        double* update, 
                                        double  inv_d,
                                        int     size,
                                        int     order  )
{
   // Individuo la posizione dell'elemento corrente
   int row     = blockDim.y * blockIdx.y + threadIdx.y;
   int column  = blockDim.x * blockIdx.x + threadIdx.x;

   // Calcolo il prodotto esterno per questa posizione e lo moltiplico
   // per il valore di 1 / d (inv_d)
   double actual_coeff = frontal[row] * frontal[column] * inv_d;

   // Ora posso salvare nella posizione corretta il valore 
   update[row * size + column] = frontal[row * order + column] - actual_coeff;

}

/**
 * Kernel per il calcolo del vettore dei valori di L
 * @param frontal         I dati della matrice frontale
 * @param cholesky_vector Il vettore dei cofficienti che si calcolano
 * @param sqrt_d          La radice quadrata di d 
 * @param order           L'ordine della matrice frontale
 */
__global__ void calcCholeskyCoeff_kernel(double* frontal        ,
                                         double* cholesky_vector,
                                         double  sqrt_d,
                                         int     order          )
{
   // Calcolo l'offset dell'elemento da calcolare
   int offset = blockDim.x * blockIdx.y + threadIdx.x;
   
   // Calcolo l'elemento 
   // Il primo è radice quadrata di d.
   if(offset == 0)
   {
      // Lo assegno alla cella di memoria
      cholesky_vector[0] = sqrt_d;
   }
   
   // Evito di fare operazioni inutili
   if(offset < order)
   {
      // Calcolo gli altri
      cholesky_vector[offset] = frontal[offset] / sqrt_d;
   }
}

// Funzione per il salvataggio dei dati 
inline void saveCholeskyData(int* indexSubset, int subsetLength, 
                             Matrix_t cholesky, double* choleskyData)
{
   // Cerco di mantenere tutto in registro per velocizzare le routine
   register int offset = 0;
   // Devo scorrere il sotto insieme di indici
   for(register int i = 0; i < subsetLength; i++)
   {
      // Vado a inserire nel fattore i dati, basandomi sugli indici del 
      // fronte corrente. Calcolo l'offset
      offset = indexSubset[i] * cholesky->order + indexSubset[0];
      
      // Inserisco il dato all'interno della matrice
      cholesky->data[offset] = choleskyData[i];
   }
}
