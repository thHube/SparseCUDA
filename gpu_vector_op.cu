//
//     Copyright (C) 2010  Alberto Franco <afranco87@gmail.com>
// 
//     This program is free software: you can redistribute it and/or modify
//     it under the terms of the GNU General Public License as published by
//     the Free Software Foundation, either version 3 of the License, or
//     (at your option) any later version.
// 
//     This program is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//     GNU General Public License for more details.
// 
//     You should have received a copy of the GNU General Public License
//     along with this program.  If not, see <http://www.gnu.org/licenses/>.

/**
 * @file   gpu_vector_op.cu
 * @author Alberto Franco
 * @date   19/05/2010
 * 
 * Contiene l'implementazione delle funzioni che utilizzano i vettori.
 */
#include "gpu_cg.h"

__global__ void dotProduct(const double* vector_a, 
                           const double* vector_b,
                           double*       result,
                           int           length);

__global__ void scaleAndSum(const double* vector_a,
                            const double* vector_b,
                            double*       result,
                            double*       alpha,
                            int           length);

__global__ void scaleAndSumNeg(const double* vector_a,
                            const double* vector_b,
                            double*       result,
                            double*       alpha,
                            int           length);


/**
 * Prodotto scalare tra due vettori calcolato in GPU.
 * @param vector_a Il primo vettore da moltiplicare.
 * @param vector_b Il secondo vettore da moltiplicare.
 * @param result   Il vettore risultato
 * @param length   La lunghezza dei vettori
 */
void gpuDotProduct(const double* vector_a, const double* vector_b,
                double*       result,   int           length)
{
   // Invoco il kernel
   dotProduct<<<1, 1>>>(vector_a, vector_b, result, length);
}


/**
 * Effettua l'operazione result = vector_a + alpha * vector_b.
 * @param vector_a il primo vettore da sommare.
 * @param vector_b il secondo vettore da sommare e scalare con alpha
 * @param result il vettore dei risultati
 * @param length la lunghezza dei vettori
 * @param bool se è vero alpha = -alpha
 * @param alpha il fattore di scala per vector_b.
 */
void gpuScaleAndSum(const double* vector_a, const double* vector_b, 
                    double*       result,   int           length, 
                    double*       alpha)
{
   // Mi serve allocare un multiplo della dimensione del blocco
   int blockCount = length / g_BlockSize;
   
   // Controllo se è multiplo della dimensione dela blocco
   if(length % g_BlockSize != 0)
   {
      // Aggiungo un blocco a quanti già allocati
      blockCount += 1;
   }
   
   // Invoco il kernel
   scaleAndSum<<<blockCount, g_BlockSize>>>(vector_a, vector_b, result, 
                                           alpha, length);
}

void gpuScaleAndSumNeg(const double* vector_a, const double* vector_b, 
                    double*       result,   int           length, 
                    double*       alpha)
{
   // Mi serve allocare un multiplo della dimensione del blocco
   int blockCount = length / g_BlockSize;
   
   // Controllo se è multiplo della dimensione dela blocco
   if(length % g_BlockSize != 0)
   {
      // Aggiungo un blocco a quanti già allocati
      blockCount += 1;
   }
   
   // Invoco il kernel
   scaleAndSumNeg<<<blockCount, g_BlockSize>>>(vector_a, vector_b, result, 
                                           alpha, length);
}


///////////////////////////////////////////////////////////////////////////////
/// KERNELS !!!

// Prodotto scalare
__global__ void dotProduct(const double* vector_a, 
                           const double* vector_b,
                           double*       result,
                           int           length)
{
   // Inizializzo il risultato a zero.
   *result = 0.0;
   
   // Scorro tutti i due vettori e sommmo
   for(int i = 0; i < length; i++)
   {
      // Aggiungo a[i] * b[i]
      *result += vector_a[i] * vector_b[i];
   }
}



// v = a + alpha b
__global__ void scaleAndSum(const double* vector_a,
                            const double* vector_b,
                            double*       result,
                            double*       alpha,
                            int           length)
{
   // Individuo il thread id
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   
   // Controllo di essere dentro il vettore. Per non leggere e scrivere su
   // aree di memoria non allocate.
   if(idx < length)
   {
      double support_alpha = (*alpha);
      // Eseguo l'operazione
      result[idx] = vector_a[idx] + support_alpha * vector_b[idx];
   } 
}


// v = a - alpha b
__global__ void scaleAndSumNeg(const double* vector_a,
                            const double* vector_b,
                            double*       result,
                            double*       alpha,
                            int           length)
{
   // Individuo il thread id
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   
   // Controllo di essere dentro il vettore. Per non leggere e scrivere su
   // aree di memoria non allocate.
   if(idx < length)
   {  
      double support_alpha = -(*alpha);
      // Eseguo l'operazione
      result[idx] = vector_a[idx] + support_alpha * vector_b[idx];
   }
}
