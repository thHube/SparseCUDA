/**
 * @file   gpu_full_op.cu
 * @author Alberto Franco
 * @date   20/05/2010
 * 
 * Contiene l'implementazione delle funzioni elaborate per il calcolo nel
 * metodo del gradiente coniugato.
 */
#include "gpu_cg.h"

// Kernel per il calcolo di alpha
__global__ void alphaCalculate(double*        result, const double* vector_r,
                               const double* vector_z, 
                               const double*  matrix, const double* vector_p, 
                               double*       support, int             length,
                               int*       row_offset, int*      column_index);
                               
// Kernel per il calcolo di beta
__global__ void betaCalculate(double*          result,  const double* vector_rk1,
                              const double* vector_zk1, const double* vector_rk, 
                              const double* vector_zk,  int             length);

/**
 * Calcola il valore di alpha in una passata sola. 
 * alpha = (r_k dot r_k) / (p^t A p)
 * @param result Il risultato dell'operazione(alpha)
 * @param vector_r Il vettore r_k da scalare con se stesso
 * @param matrix La matrice da applicare al vettore p
 * @param vector_p Il vettore p
 * @param support Un vettore di supporto dove salvare le informazioni temporanee.
 * @param length La lunghezza dei vettori
 */
void gpuAlphaCalculate(double*       result, const double* vector_rk,
                       const double* vector_zk, 
                       const double* matrix, const double* vector_p, 
                       double*      support, int             length,
                       int*      row_offset, int*      column_index)
{
   // Devo capire quanti blocchi di thread avviare
   int blockNumber = length / g_BlockSize;
   
   // Controllo che la lunghezza dei vettori sia divisibile per g_BlockSize
   blockNumber += (length % g_BlockSize != 0)? 1 : 0;
   
   // Avvio i kernel
   alphaCalculate<<<blockNumber, g_BlockSize>>>(result,  vector_rk,
                                                vector_zk, 
                                                matrix,  vector_p, 
                                                support, length,
                                                row_offset, column_index);
}

/**
 * Calcola il valore di beta in un'unica volta.
 * beta = (r_k1 dot r_k1) / (r_k dot r_k).
 * @param result Il risultato della operazione
 * @param vector_k Il vettore r_k.
 * @param vector_k1 Il vettore r_k1.
 * @param length La lunghezza del vettore.
 */
void gpuBetaCalculate(double*          result,  const double* vector_rk1,
                      const double* vector_zk1, const double* vector_rk,
                      const double* vector_zk,  int             length)
{
   // Invoco il kernel direttamente
   betaCalculate<<<1, 2>>>(result, vector_rk1, vector_zk1, 
                           vector_rk, vector_zk, length);
}

///////////////////////////////////////////////////////////////////////////////
/// FUNZIONI GRAFICHE (__device__)

/// Funzionalità di supporto che mi calcola il prodotto matrice vettore
__device__ void apply(double*       result, const double* values, 
                      const double* vector, int           length, 
                      int           index,  int*          row_offset,
                      int*          column_index)
{
   // Controllo che possa calcolare l'elemento corrente
   if(index < length)
   {
      // Inizializzo l'elemento del vettore
      result[index] = 0.0;
      
      // Vado a scorrere il vettore 
      for(int i = row_offset[index]; i < row_offset[index + 1]; i++)
      {
         // Aggiungo l'i-eismo prodotto
         result[index] += values[i] * vector[column_index[i]];
      }
   }
}

// Funzionalità di supporto per il calcolo del prodotto scalare
__device__ double dot(const double* a, const double* b, int length)
{
   // Inizializzo il risultato a zero
   double result = 0.0;
   
   // Scorro tutti i due vettori, moltiplico membro a membro e li sommo.
   for(int i = 0; i < length; i++)
   {
      // Moltiplico e sommo al risultato
      result += a[i] * b[i];
   }
   
   // Ritorno il risultato
   return result;
}
///
///////////////////////////////////////////////////////////////////////////////
/// IMPLEMENTAZIONE DEL KERNEL(alpha)
__global__ void alphaCalculate(double*        result, const double* vector_r, 
                               const double* vector_zk,
                               const double*  matrix, const double* vector_p, 
                               double*       support, int             length,
                               int*       row_offset, int*     column_index)
{
   // Calcolo l'id del thread in esecuzione
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   
   // Vado a calcolare la moltiplicazione tra il vettore e la matrice
   // result = A p
   apply(support, matrix, vector_p, length, idx, row_offset, column_index);
   
   // Facciamo in modo di avere tutto calcolato
   __syncthreads();
   
   // Ci servono delle variabili condivise
   __shared__ double dividend;
   __shared__ double divisor;
   
   // Ora a seconda del thread in cui siamo facciamo operazioni diverse.
   switch(idx)
   {
      // Il thread 0 calcola il dividendo
      case 0:
         dividend = dot(vector_r, vector_zk, length);
      break;
      
      // Il thread 1 calcola il divisore
      case 1:
         divisor = dot(vector_p, support, length);
      break;
   }
   
   // Altro step di sincronizzazione
   __syncthreads();
   
   // Il thread zero calcola la divisione
   if(!idx)
   {
      // calcolo alpha
      *result = dividend / divisor;
   }
}
///
///////////////////////////////////////////////////////////////////////////////
/// IMPLEMENTAZIONE DEL KERNEL(beta)
__global__ void betaCalculate(double*          result,  const double* vector_rk1,
                              const double* vector_zk1, const double* vector_rk, 
                              const double* vector_zk,  int             length)
{
   // Calcolo l'id del thread 
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   
   // Mi servono delle variabili di supporto
   __shared__ double dividend;
   __shared__ double divisor;
   
   // Vado a fare il prodotto scalare
   switch(idx)
   {
      // Al thread 0 facciamo calcolare il dividendo
      case 0:
         dividend = dot(vector_rk1, vector_zk1, length);
      break;

      // Al thread 1 il divisore
      case 1:
         divisor = dot(vector_rk, vector_zk, length);
      break;
   }
   
   // Punto di sincronizzazione
   __syncthreads();
   
   // Il thread zero calcola la divisione
   if(!idx)
   {
      // beta = dividend / divisor
      *result = dividend / divisor;
   }
}
///
///////////////////////////////////////////////////////////////////////////////
