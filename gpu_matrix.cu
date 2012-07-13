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
 * @file   gpu_matrix.cu
 * @author Alberto Franco
 * @date   10/05/2010
 * 
 * Implementazione delle funzionalità di manipolazione
 * delle matrici utilizzando la GPU.
 */
#include "gpu_operations.h"

// Kernel per la moltiplicazione tra matrici.
__global__ void matrixMultiply_kernel(const Matrix first  , 
                                      const Matrix second , 
                                            Matrix result );

// Kernel per la somma tra matrici.
__global__ void matrixSum_kernel(double* first,  double* second, 
                                 double* result, int size      );


// Kernel per la somma estesa tra matrici
__global__ void extendAdd_kernel(const Matrix first,  const Matrix second,
                                       Matrix result, int size          );

// Somma tra matrici.
void matrix_sum(Matrix first, Matrix second, Matrix_t result);

// Funzione per l'allocazione dello spazio per la matrice.
void matrix_Allocate(Matrix_t matrix);

// Somma tra matrici in CPU
void matrix_sum_cpu(Matrix first, Matrix second, Matrix_t result);

/**
 * Somma estende le due matrici passate come parametro
 * @param first la prima matrice da somma estendere 
 * @param second la seconda matrice da somma estendere 
 * @return la matrice somma estesa
 */
Matrix_t extendAdd_cuda(Matrix_t first, Matrix_t second)
{
   // Dobbiamo scorrere tutto l'insieme degli indici.
   // Ci servono due valori interi di supporto.
   int i = 0;
   int j = 0;
   bool a_finish = false;
   bool b_finish = false;
   int newLength = 0;
   
   // Utilizziamo una matrice di supporto
   Matrix_t result = new Matrix;
   // Usiamo un array che ha la lunghezza dei 
   
   int  allocSize = (first->subsetLength + second->subsetLength);
   // int* newSubset = Allocator<int>::get().allocate(allocSize);
   int* newSubset = (int*)malloc(sizeof(int) * allocSize);
   
   // Ora passiamo attraverso gli array con gli indici
   // e ricostruiamo il nuovo array con gli indici
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
   
   // Ricostrutisco le matrici con il nuovo sotto-insieme di indici
   // sulla VRAM. Mi servono due nuove matrici.
   Matrix first_extend;
   Matrix second_extend;
   
   // Calcolo lo spazio necessario per l'allocazione
   int allocation_size = newLength * newLength;
   
   // Alloco spazio sulla memoria per i miei risultati
   // first_extend.data  = Allocator<double>::get().allocate(allocation_size);
   // second_extend.data = Allocator<double>::get().allocate(allocation_size);
   first_extend.data  = (double*)malloc(sizeof(double) * allocation_size * 2);
   second_extend.data = first_extend.data + allocation_size;// (double*)malloc(sizeof(double) * allocation_size);
   
   memset(first_extend.data, 0, sizeof(double) * allocation_size * 2);
   // memset(second_extend.data, 0, sizeof(double) * allocation_size);
   
   // Ho bisogno di un paio di valori di supporto
   int i_A = 0, i_B = 0, j_A, j_B;
   int offset_vram, offset_ram;
   // Ora copio i valori 
   for(i = 0; i < newLength; i++)
   { 
      // Devo scorrere ogni volta tutta la riga.
      j_A = j_B = 0;
      
      for(j = 0; j < newLength; j++)
      {
         // Se il valore newSubset[i] e newSubset[j] appartengono al 
         // insieme degli indici di first(o di second) allora devo
         // copiare i dati nelle nuove matrici.
         if(newSubset[i] == first->indexSubset[i_A] && 
            newSubset[j] == first->indexSubset[j_A]) 
         {
            offset_vram = newLength * i + j;
            offset_ram  = first->order * i_A + j_A;
            // Copio i valori sulla scheda grafica.  
            memcpy(first_extend.data + offset_vram, 
                       first->data + offset_ram, 
                       sizeof(double));
            j_A++;
         }  
          
         // Stessa cosa per la seconda matrice
         if(newSubset[i] == second->indexSubset[i_B] && 
            newSubset[j] == second->indexSubset[j_B]) 
         {
            offset_vram = newLength * i + j;
            offset_ram  = second->order * i_B + j_B;
            // Copio i valori sulla scheda grafica.  
               memcpy(second_extend.data + offset_vram, 
                       second->data + offset_ram, 
                       sizeof(double));
            j_B++;
         }
      }
      
      // Se il valore di newSubset[i] è uguale a first.indexSubset[i_A]
      // (o da second) allora incremento i_A
      i_A = (newSubset[i] == first->indexSubset[i_A])? i_A + 1: i_A;
      i_B = (newSubset[i] == second->indexSubset[i_B])? i_B + 1: i_B;
   }

   first_extend.order = second_extend.order = newLength;

   // Assegno tutti i valori
   result->order = result->subsetLength = newLength;
   result->indexSubset = newSubset;
   
   // Calcolo la somma 
   matrix_sum_cpu(first_extend, second_extend, result);
   
   // Libero la memoria che non mi serve più
   free(first_extend.data);
   // free(second_extend.data);

   // Ritorno il risultato dell'operazione
   return result;
}

/**
 * Somma estende le due matrici e salva il risultato 
 * nella prima.
 * @param first la prima matrice da somma estendere in cui
 *        verrà salvato il risultato.
 * @param second la seconda matrice da somma estendere 
 */
void extendAddFirst(Matrix_t first, Matrix_t second)
{
   // Dobbiamo scorrere tutto l'insieme degli indici.
   // Ci servono due valori interi di supporto.
   int i = 0;
   int j = 0;
   bool a_finish = false;
   bool b_finish = false;
   int newLength = 0;
   
   // Usiamo un array che ha la lunghezza dei 
   int numByte = (first->subsetLength + second->subsetLength) * sizeof(int);
   int* newSubset = (int*)malloc(numByte);
   
   // Ora passiamo attraverso gli array con gli indici
   // e ricostruiamo il nuovo array con gli indici
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
   
   // Alloco la nuova matrice da ritornare
   Matrix_t result = (Matrix_t)malloc(sizeof(Matrix));
   
   // Già conosco l'ordine della matrice
   result->order = result->subsetLength = newLength;
   
   // e anche il suo sotto insieme di indici, lo copio in VRAM
   cudaMalloc((void**)&result->indexSubset, newLength * sizeof(int));
   
   cudaMemcpy(result->indexSubset, // Copio nell'oggetto che voglio usare
              newSubset,           // in Video RAM l'insieme degli indici
              newLength * sizeof(int),
              cudaMemcpyHostToDevice);
   
   // Copio le due matrici in GPU
   Matrix device_1, device_2;
   
   device_1.order = first->order;
   device_2.order = second->order;
   
   device_1.subsetLength = first->subsetLength;
   device_2.subsetLength = second->subsetLength;
   
   int size_1 = device_1.order * device_1.order * sizeof(double);
   int size_2 = device_2.order * device_2.order * sizeof(double);
   // Alloco lo spazio per i valori
   cudaMalloc((void**)&device_1.data, size_1);
   cudaMalloc((void**)&device_2.data, size_2);
   
   // Copio i valori
   cudaMemcpy(device_1.data, first->data, size_1, cudaMemcpyHostToDevice);
   cudaMemcpy(device_2.data, second->data, size_2, cudaMemcpyHostToDevice);
   
   // Alloco lo spazio per il vettore degli indici
   cudaMalloc((void**)&device_1.indexSubset, device_1.subsetLength * sizeof(int));
   cudaMalloc((void**)&device_2.indexSubset, device_2.subsetLength * sizeof(int));
   
   // Copio i dati di indice.
   cudaMemcpy(device_1.indexSubset, first->indexSubset, 
              first->subsetLength * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(device_2.indexSubset, second->indexSubset, 
              second->subsetLength * sizeof(int), cudaMemcpyHostToDevice);
   
   // Ora devo eseguire il kernel aumentando la matrice di risultato
   // per evitare problemi con i thread
   const int blockSize = 16;
   int size = result->order;
   
   // Nel caso il numero non sia divisibile per blockSize
   if(size % blockSize != 0)
   {
      // Aumento la dimenione fino ad un multiplo
      size = blockSize * (size / blockSize + 1);
   } 
   
   // Alloco la matrice sui cui effettuerò operazioni in GPU
   cudaMalloc((void**)&result->data, size * size * sizeof(double));
   cudaMemset(result->data, 0, size * size * sizeof(double));
   
   // Calcolo il numero di blocchi
   dim3 threadsPerBlock(blockSize, blockSize);
   dim3 blockCount(size / blockSize, size / blockSize);
   
   // Avvio il kernel
   extendAdd_kernel<<<blockCount, threadsPerBlock>>>(device_1, device_2,
                                                     *result,  size    );
                                                     
   // Recupero i risultati, prima alloco la memoria
   int     result_data_size   = result->order * result->order * sizeof(double);
   double* result_data        = (double*)malloc(result_data_size);
   
   // Copio la memoria dalla scheda
   for(int i = 0; i < result->order; i++)
      cudaMemcpy(result_data  + i * result->order, // Copio nella riga i
                 result->data + i * size,          // Dalla riga i 
                 result->order * sizeof(double),   // Dimensione 
                 cudaMemcpyDeviceToHost);          // Direzione
   
   // Libero la memoria allocata sulla scheda
   cudaFree(result->data);
   cudaFree(result->indexSubset);
   
   cudaFree(device_1.data);
   cudaFree(device_1.indexSubset);
   
   cudaFree(device_2.data);
   cudaFree(device_2.indexSubset);
   
   result->data = result_data;
   result->indexSubset = newSubset;
   
   *first = *result;
}

/**
 * Moltiplicazione tra matrici. 
 * ! ATTENZIONE ! 
 * ! Attualmente sembra funzionare solo per multipli di 16(che è la   !
 * ! dimensione del blocco). Bisogna capire come aggirare il problema !
 * !
 * @param first La prima matrice da moltiplicare
 * @param second La seconda matrice da moltiplicare
 * @return la matrice moltiplicata
 */
Matrix_t multipy(Matrix_t first, Matrix_t second)
{
   // Dichiaro i puntatori con cui agganciarmi alla VRAM
   Matrix device_first;
   Matrix device_second;
   Matrix device_result;
   
   // Alloco lo spazio per la matrice con i risultati
   Matrix_t result = (Matrix_t)malloc(sizeof(Matrix));
   // Copio i dati
   result->order = first->order;
   // Alloco la matrice
   matrix_Allocate(result);   
   
   device_first.order = first->order;
   device_second.order = second->order;
   
   int size = first->order * first->order * sizeof(double);
   
   // Alloco lo spazio in VRAM e copio i dati
   cudaMalloc((void**)&device_first.data, size);
   cudaMalloc((void**)&device_second.data, size);
   cudaMalloc((void**)&device_result.data, size);
   
   // Copio la memoria
   cudaMemcpy(device_first.data, first->data, size, cudaMemcpyHostToDevice);
   cudaMemcpy(device_second.data, second->data, size, cudaMemcpyHostToDevice);
   
   // Calcolo dimensione del blocco e 
   dim3 blockSize(16, 16);
   dim3 blockNum(first->order/ blockSize.x, first->order/ blockSize.y);
   
   // Effettuo la moltiplicazione 
   matrixMultiply_kernel<<<blockNum, blockSize>>> (device_first,
                                                          device_second,
                                                          device_result);
   // Copio i risultati in memoria centrale 
   cudaMemcpy(result->data, device_result.data, size, cudaMemcpyDeviceToHost);

   // Liberiamo la memoria allocata
   cudaFree(device_first.data);
   cudaFree(device_second.data);
   cudaFree(device_result.data);
   
   // Ritorno i risultati 
   return result;
}


/**
 * Kernel per la moltiplicazione tra matrici.
 */
__global__ void matrixMultiply_kernel(const Matrix first, 
                                      const Matrix second, Matrix result)
{
   // Calcolo l'elemento da calcolare
   int row    = blockIdx.y * blockDim.y + threadIdx.y;
   int column = blockIdx.x * blockDim.x + threadIdx.x;
   
   // Dichiaro il valore da poi copiare nella zona di memoria giusta.
   double value = 0.0;
   
   // Ora calcolo la somma degli elementi della colonna.
   for(int i = 0; i < first.order; i++)
   {
      value += first.data[row * first.order + i] * 
               second.data[i * first.order + column];
   }
   
   __syncthreads();
   // Copio il valore calcolato nella posizione corretta
   result.data[row * first.order + column] = value;
}

// Funzione per l'allocazione dello spazio per la matrice.
void matrix_Allocate(Matrix_t matrix)
{
   // Alloco lo spazio per i dati della matrice
   matrix->data = (double*)malloc(matrix->order * matrix->order * 
                                  sizeof(double));
}


/**
 * Kernel per la somma tra matrici.
 */
__global__ void matrixSum_kernel(double* first,  double* second, 
                                 double* result, int size      )
{
   // Calcolo la posizione di riga e colonna
   int row    = blockIdx.y * blockDim.y + threadIdx.y;
   int column = blockIdx.x * blockDim.x + threadIdx.x;
   
   // Calocolo l'offset della zona di memoria su cui scrivere
   int offset = size * row + column;
   
   // Sommo i due valori
   result[offset] = first[offset] + second[offset];
}

/**
 * Calcola la somma tra le matrici in GPU.
 * @return un array con la somma della matrici.
 */
void matrix_sum(Matrix first, Matrix second, Matrix_t result)
{
   // Alloco la dimesione del blocco e la dimensione della matrice
   // aumentata fino al multiplo più vicino di blockSize
   const int blockSize  = 16;
   int size             = first.order;
   
   // Dobbiamo capire come comportarci e poi vedere di organizzare le 
   // operazioni in modo efficiente.
   if(size % blockSize != 0)
   {
      // In questo caso la dimensione NON è divisibile per blockSize
      // allochiamo più spazio del necessario.
      size = blockSize * (size / blockSize + 1); 
   }
   
   // Allochiamo lo spazio sulla scheda video.
   double* first_data;
   double* second_data;
   // La dimensione di allocazione
   int allocation_size = size * size * sizeof(double);
   // Ci serve il limite su cui scrivere.
   int source_pitch    = first.order * first.order * sizeof(double);
   
   // Allochiamo lo spazio che ci serve in ram video
   cudaMalloc((void**)&first_data, allocation_size);
   cudaMalloc((void**)&second_data, allocation_size);
   cudaMalloc((void**)&result->data, allocation_size);
   
   // Preparo la memoria, inizializzo tutto a zero.
   cudaMemset(first_data, 0, allocation_size);
   cudaMemset(second_data, 0, allocation_size);
      
   // Copio i dati dalla memoria centrale alla memoria video
   for(int i = 0; i < first.order; i++)
   {
      cudaMemcpy(first_data + i * size, first.data + i * first.order, 
                 first.order * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(second_data + i * size, second.data + i * first.order, 
                 second.order * sizeof(double), cudaMemcpyHostToDevice);
   }
   
   // Alloco le variabili utili per capire come compoprtarmi
   dim3 threadsPerBlock(blockSize, blockSize);
   dim3 blockCount(size / blockSize, size / blockSize);
   
   // Lancio il kernel 
   matrixSum_kernel<<<blockCount, threadsPerBlock>>> (first_data, second_data, result->data, size);
   
   // Ora copio i risultati in memoria principale
   // double* result_data = Allocator<double>::get().allocate(source_pitch / sizeof(double));
   double* result_data = (double*)malloc(source_pitch);
   
   printf("%d \n", source_pitch);
   
   // Copio i dati dalla memoria video
   for(int i = 0 ; i < first.order; i++)
   {
      cudaMemcpy(result_data + i * first.order, result->data + i  * size,
                 first.order * sizeof(double), cudaMemcpyDeviceToHost);
   }
   
   // Libero la memoria allocata
   cudaFree(first_data);
   cudaFree(second_data);
   cudaFree(result->data);
   
   // Assegno il puntatore alla struttura
   result->data = result_data;
}

/**
 * Funzionalità che invoco dal dispositivo per sommare due elementi
 */
__device__ double sumElements_device(const Matrix first, 
                                     const Matrix second, 
                                     int row, int column)
{
   // Andiamo a fare la somma dei due elementi.
   // Se uno non è nella matrice, lo poniamo a zero.
   double first_operand    = 0.0;
   double second_operand   = 0.0;
   
   // Controllo l'esitenza degli indici all'interno del
   // sottoinsieme della prima matrice
   for(int i = 0; i < first.subsetLength; i++)
   {
      for(int j = 0; j < first.subsetLength; j++)
      {
         // Se troviamo un match
         if(first.indexSubset[i] == row && first.indexSubset[j] == column)
         {
            // Modifichiamo il primo operando
            first_operand = first.data[i * first.order + j];
         }
      }
   }
   
   // Stessa cosa facciamo per il secondo
   for(int i = 0; i < second.subsetLength; i++)
   {
      for(int j = 0; j < second.subsetLength; j++)
      {
         // Se c'è il valore 
         if(second.indexSubset[i] == row && second.indexSubset[j] == column)
         {
            // Modifichiamo l'operando
            second_operand = second.data[i * second.order + j];
         }
      }
   }
   
   return first_operand + second_operand;
}

/**
 * Kernel per il calcolo della somma estesa.
 */
__global__ void extendAdd_kernel(const Matrix first,  const Matrix second,
                                       Matrix result, int size           )
{
   // Calcolo la posizione di riga e colonna
   int row    = blockIdx.y * blockDim.y + threadIdx.y;
   int column = blockIdx.x * blockDim.x + threadIdx.x;
   
   // Individuo gli indici 
   int row_index     = result.indexSubset[row];
   int column_index  = result.indexSubset[column]; 

   // Calcolo l'offset
   int offset = row * size + column;
   
   // Sommo i due elementi all'interno della cella di memoria corretta
   result.data[offset] = sumElements_device(first, second, row_index, column_index);
}

/** 
 * Somma tra matrici in CPU, vediamo se è più efficente.
 */
inline void matrix_sum_cpu(Matrix first, Matrix second, Matrix_t result)
{
   int order_2 = first.order * first.order;
   
   // Alloco la matrice.
   result->data = (double*)malloc(order_2 * sizeof(double));
   memset(result->data, 0, order_2 * sizeof(double));
   
   // Sommo direttamente le due matrici
   for(register int i = 0; i < order_2; i++)
   {
      result->data[i] = first.data[i] + second.data[i];
   }
}
