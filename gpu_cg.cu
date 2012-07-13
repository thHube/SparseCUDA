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
 * @file   gpu_cg.h
 * @author Alberto Franco
 * @date   20/05/2010
 * 
 * Contiene la definizione della funzione di calcolo del gradiente coniugato
 * tutto eseguito in GPU.
 */
#include "gpu_cg.h"

__global__ void conjGrad(int max_it, 
                          const sparse_matrix_t linear_system,
                          double* residual_odd,
                          double* residual_even,
                          double* direction_odd,
                          double* direction_even,
                          double* solution_odd,
                          double* solution_even,
                          double* support);

/**
 * Risolve il gradiente coniugato in GPU.
 * @param max_it Numero di iterazioni da effettuare.
 * @param linear_system Il sistema da risolvere
 * @param residual_odd Il vettore residuo dispari
 * @param residual_even Il vettore residuo pari
 * @param direction_odd Il vettore direzione dispari
 * @param direction_even Il vettore direzione pari
 * @param solution_odd   Il vettore soluzione dispari
 * @param solution_even Il vettore soluzione pari
 * @param support Un vettore di supporto
 */
void gpuConjugateGradient(int max_it, 
                          const sparse_matrix_t linear_system,
                          double* residual_odd,
                          double* residual_even,
                          double* direction_odd,
                          double* direction_even,
                          double* solution_odd,
                          double* solution_even,
                          double* support)
{


}

///////////////////////////////////////////////////////////////////////////////
/// Funzioni di utilità

// Prodotto scalare tra due vettori
__device__ double dot(const double* a, const double* b, int length)
{
   // Inizializziamo il risultato a zero.
   double result = 0.0;
   
   // Ora scorriamo i due vettori
   for(int i = 0; i < length; i++)
   {
      // Sommiamo il risultato l'i-esima moltiplicazione
      result += a[i] * b[i];
   }
   
   // Restituiamo il risultato al chiamante
   return result;
}

// Prodotto matrice vettore.
__device__ void apply(double* result, const sparse_matrix_t matrix, 
                      const double* vector, int thread_idx)
{
   // Non voglio scrivere in aree di memoria non allocate.
   if(thread_idx < matrix->order)
   {
      // Inizializzo il valore di result_i
      result[thread_idx] = 0.0;
      
      // scorro tutto l'array considerando solo i non nulli
      for(int i = matrix->row_offset[thread_idx];
          i < matrix->row_offset[thread_idx]; i++)
      {
         // Aggiungo al valore l'i-esimo prodotto
         result[thread_idx] += matrix->values[i] * vector[matrix->column_index[i]];
      }       
   }
   // Sincronizzo per avere che al ritorno dalla funzione ho tutti il vettore
   // calcolato correttamente.
   __syncthreads();
}

// alpha = v dot A v
__device__ double applyDot(const sparse_matrix_t matrix, const double* vector, 
                           double* support, int thread_idx)
{
   // Prima applico il vettore alla matrices
   apply(support, matrix, vector, thread_idx);
   
   // Solo con un thread calcolo il prodotto scalare
   if(!thread_idx)
   {
      // Ora faccio il prodotto scalare tra i due vettori
      return dot(vector, support, matrix->order);
   }   
   
   // Gli altri thread ritornano zero.
   return 0.0;
}

// r = v + alpha v2
__device__ void sumScaleVector(double* result, const double*  vector_first,
                               double   alpha, const double* vector_second,
                               int thread_idx, int                  length)
{
   // Controllo sempre si non scrivere fuori dalla memoria allocata
   if(thread_idx < length)
   {
      // Eseguo l'operazione
      result[thread_idx] = vector_first[thread_idx] + alpha * vector_second[thread_idx];
   }
   
   // Punto di sincronizzazione
   __syncthreads();
}

// r = v + alpha A v2
__device__ void applyScaleSum(double*               result, 
                              double*              support,
                              const double*   vector_first,
                              double                 alpha, 
                              const sparse_matrix_t matrix,
                              const double*  vector_second, 
                              int               thread_idx)
{
   // Prima applico il vettore alla matrice
   apply(support, matrix, vector_second, thread_idx);
   
   // Ora sommo e scalo il vettore appena trovato
   sumScaleVector(result, vector_first, alpha, support, thread_idx, matrix->order);
}

///
///////////////////////////////////////////////////////////////////////////////
/// IMPLEMENTAZIONE DEL KERNEL
__global__ void conjGrad(int max_it, 
                          const sparse_matrix_t linear_system,
                          double* residual_odd,
                          double* residual_even,
                          double* direction_odd,
                          double* direction_even,
                          double* solution_odd,
                          double* solution_even,
                          double* support)
{
   // Calcolo l'identificativo del thread
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   
   // Utilizzo dei valori temporanei per gestire il caso k e k +1
   double *x_k, *x_k1, *r_k, *r_k1, *p_k, *p_k1, alpha, beta, r_dot;
   
   // Salvo l'ordine del sistema per chiarezza 
   int order = linear_system->order;
   
   // Eseguo tante operazioni, quante richieste
   for(int k = 0; k < max_it; k++)
   {
      // Calcolo i valori corretti dei miei vettori a seconda dell'iterazione 
      // pari o dispari
      if(k % 2 == 0)
      {
         // k è pari
         x_k = solution_even;  x_k1 = solution_odd;
         r_k = residual_even;  r_k1 = residual_odd;
         p_k = direction_even; p_k1 = direction_odd;
      }
      else
      {
         // k è dispari
         x_k1 = solution_even;  x_k = solution_odd;
         r_k1 = residual_even;  r_k = residual_odd;
         p_k1 = direction_even; p_k = direction_odd;   
      }
      
      // Calcolo alpha 
      r_dot = dot(r_k, r_k, order);
      alpha = r_dot / applyDot(linear_system, p_k, support, idx);
      
      // Calcolo x_k1
      sumScaleVector(x_k1, x_k, alpha, p_k, idx, order);
      
      // Calcolo r_k
      applyScaleSum(r_k1, support, r_k, -alpha, linear_system, p_k, idx);
      
      // Calcolo beta
      beta = dot(r_k1, r_k1, order) / r_dot;
      
      // Calcolo p_k1
      sumScaleVector(p_k1, r_k1, beta, p_k, idx, order);
   }
}
