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
 * @date   19/05/2010
 * 
 * Contiene i prototipi delle funzioni che vanno ad 
 * agire sulla GPU per il CG e le strutture per agire con tali 
 * funzioni.
 */
#include "gpu_cg.h"

__global__ void applyVector(const int*    row_offset, const int*    column_index,
                            const double* values,     const double* vector, 
                            double*       result,     int           length);


/**
 * Applica il vettore alla matrice e salva i risultati in result.
 * @param row_offset L'offset delle righe della matrice.
 * @param column_index Le posizioni dei non nulli della matrice 
 * @param values I valori della matrice a cui applicare il vettore
 * @param vector Il vettore da applicare.
 * @param result Il vettore in cui salvare i risultati
 * @param length La lunghezza del vettore da applicare.
 */
void gpuApplyVector(const int*    row_offset, const int*    column_index,
                    const double* values,     const double* vector, 
                    double*       result,     int           length)
{
   // Calcolo il numero di blocchi da invocare per eseguire correnttamente
   // l'operazione richiesta.
   int blockCount = length / g_BlockSize;
   
   // Se non è divisibile per la dimensione di blocco
   if(length % g_BlockSize != 0)
   {
      // Incremento il numero di blocchi di uno
      blockCount += 1;
   }
   
   // Invoco il kernel
   applyVector<<<blockCount, g_BlockSize>>>(row_offset, column_index, values,
                                            vector, result, length);
   
}


// Kernel per la divisione degli scalari
__global__ void divide(double* a, double* b)
{
   // Divido i due numeri e metto il risultato in b
   *b = *a / *b;
}

/**
 * Divide i valori e li inserisce nel secondo. a / b
 * @param first Il primo valore (a)
 * @param second Il secondo valore (b) <- a / b
 */
void gpuDivide(double* first, double* second)
{
   // Invoco il kernel
   divide<<<1, 1>>>(first, second);
}



// Kernel per la applicazione matrice vettore
__global__ void applyVector(const int*    row_offset, const int*    column_index,
                            const double* values,     const double* vector, 
                            double*       result,     int           length)
{
   // Calcolo l'id del thread
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   
   // Se sono uscito dalla lunghezza non eseguo operazioni per evitare 
   // di scrivere su memoria non allocata.
   if(idx < length)
   {
      // Inizializzo la cella risultato a zero.
      result[idx] = 0.0;
      
      // Scorro tutti gli elementi della riga che mi interessano e li sommo
      for(int i = row_offset[idx]; i < row_offset[idx +1]; i++)
      {  
         // Aggiungo al risultato il valore della moltiplicazione,
         // i valori che sono nulli nella riga non mi interessano perchè
         // cmq la moltiplicazione è zero.
         result[idx] += values[i] * vector[column_index[i]];
      }
   }
}

