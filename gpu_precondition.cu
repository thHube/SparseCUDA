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
 * @file   gpu_precondition.cu
 * @author Alberto Franco
 * @date   24/05/2010
 * 
 * Contiene la definizione per la funzione di calcolo del precondizionante 
 * KK^t e di tutte le funzioni di supporto ad esso necessarie.
 */
#include "gpu_cg.h"

// Kernel per il calcolo della matrice K
__global__ void kValues(double* result, int* row_offset, int* column_index,
                        double* values, int order);

// Kernel per l'inversione della matrice M.
__global__ void inversion(double* result, int* row_offset, int* column_index, 
                          double* values, int order);

/**
 * Calcola i valori del precondizionante KK^t, ogni valore è computato con
 * la fattorizzazione incompleta di Cholesky e la matrice originale.
 * @param prec_vales Il risultato dell'operazione.
 * @param row_offset gli offset di riga.
 * @param column_index Gli indici di colonna
 * @param values I valori della matrice originale.
 * @param order L'ordine della matrice originale.
 */
void gpuCalcKKValues(double* prec_values, int* row_offset, int* column_index,
                     double* values, int order, int non_zero)
{
   // Alloco spazio per la matrice K 
   double* k = (double*)gpuAllocate(non_zero * sizeof(double));
   
   // Mi serve sapere la dimensione di allocazione dei thread
   int size = order / g_BlockSize;
   
   // se L'ordine non è multiplo della dimensione di blocco
   if(order % g_BlockSize != 0)
   {
      // Incremento la dimensione di uno
      size += 1;
   }
   
   // Imposto k tutto a zero.
   cudaMemset(k, 0, non_zero * sizeof(double));
   
   // Ora che il fattore è calcolato posso calcolare k
   dim3 blocks(size, size), threads(g_BlockSize, g_BlockSize);
   kValues<<<blocks, threads>>>(k, row_offset, column_index, values, order);
   
   // Infine i valori dell'inversione
   inversion<<<size, g_BlockSize>>>(prec_values, row_offset, column_index, 
                                    k, order);
   
   // Libero la memoria allocata
   gpuDelete(k); 
}
///////////////////////////////////////////////////////////////////////////////
/// IMPLEMENTAZIONE DEI KERNEL

__device__ int binarySearch(const int* ref, const int start, const int end, const int& key)
{
   // Usiamo due interi e chiediamo che restino in registro.
   register int first = start;
   register int last  = end;
   register int mid;
   
   // Via con la ricerca binaria
   while(first <= last)
   {
      // Calcolo il valore medio
      mid = (first + last) / 2;
      
      // Se la chiave è minore del mio elemento medio
      if(ref[mid] < key)
      {
         // ripartiamo dall'elemento successivo.
         first = mid + 1;
      }
      else if(ref[mid] > key)
      {
         // Altrimenti ripartiamo dall'elemento precedente
         last = mid - 1;
      }
      else
      {
         // Abbiamo trovato l'elemento
         return mid;
      }
   }
   // Errore, elemento non trovato.
   return -1;
}

// Kernel per il calcolo della matrice K
__global__ void kValues(double* result, int* row_offset, int* column_index,
                        double* values, int order)
{
   // Calcolo l'identificativo del thread
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   int idy = blockDim.y * blockIdx.y + threadIdx.y;
   
   // Ora controllo di non scrivere su memoria non allocata.
   if(idx < order && idy < order)
   {
      // Calcolo la posizione della diagonale.
      int diag_offset = binarySearch(column_index, row_offset[idx], 
                                      row_offset[idx + 1], idx);
      // Calcolo la posizone dell'altro elemento  
      int pos_offset  = binarySearch(column_index, row_offset[idy],
                                      row_offset[idy + 1], idx);
                                      
      if(idx == idy)
      {
         result[pos_offset] = 1.0;
      }
      else if(idx < idy)
      {
         result[pos_offset] = 0.0 - (values[pos_offset] * (1/values[diag_offset]));
      }
   }
}

// Kernel per l'inversione della matrice M.
__global__ void inversion(double* result, int* row_offset, int* column_index, 
                          double* values, int order)
{
   // Calcolo l'indice del thread
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   
   // Controllo di non uscire dalla memoria
   if(idx < order)
   {
      // Un paio di valori di supporto.
      double dot; int i, offset_i, offset_j;
      bool increment;
      
      // Scorro la riga corrente
      for(int j = row_offset[idx]; j < row_offset[idx + 1]; j++)
      {
         // Azzero il valore del prototto scalare e trovo i
         dot = 0.0; i = column_index[j];
         
         // Inizializzo gli offset
         offset_i = row_offset[i];
         offset_j = row_offset[idx];
         
         // Per l'elemento i j devo calcolare il prodotto vettoriale tra le
         // righe i e j. La matrice è simmetrica.
         for(int k = 0; k < order; k++)
         {
            increment = true;
            // Tutti e due gli elementi devono essere non nulli per calcolare
            // il prodotto nell'elemento
            if(column_index[offset_i] == k)
            {
               // Controllo il secondo
               if(column_index[offset_j] == k)
               {
                  // Calcolo il prodotto
                  dot += values[offset_i] * values[offset_j];
                  
                  // Incremento il valore di offset_j
                  offset_j += 1;
                  
                  // Non vogliamo che il valore sia aumentato due volte
               }
               // Incremento offset_i
               offset_i += 1;
            }
            
            // Se non sono entrato nell'altro if
            if(column_index[offset_j] == k && increment)
            {
               // Aumento l'offset
               offset_j += 1;
            }
         }
         
         // Assegno il valore appena calcolato.
         result[j] = dot;
      }
   }
}
///
///////////////////////////////////////////////////////////////////////////////
                          