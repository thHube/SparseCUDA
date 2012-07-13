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
#ifndef GPU_CG__H__
#define GPU_CG__H__

#include "cuda.h"

// Un parametro globale per la dimensione del blocco
const int g_BlockSize = 16;

/**
 * Alloca sulla scheda grafica un segmento di memoria lungo quanto richiesto.
 * @param size La dimensione in byte del segmento.
 * @return Il puntatore alla zona di memoria.
 */
void* gpuAllocate(int size);

/**
 * Libera la memoria allocata puntata dal parametro.
 * @param memoryPointer il puntatore alla zona di memoria da liberare
 */
void gpuDelete(void* memoryPointer);

/**
 * Copia la memoria sull'host e ritorna il puntatore alla zona 
 * di memoria allocata.
 * @param gpuPointer Il punto da cui copiare la memoria.
 * @param size La dimensione in byte della memoria da copiare.
 * @return Il puntatore alla zona di memoria con i dati copiati in RAM.
 */
void* gpuCopyMemoryToHost(void* gpuPointer, int size);

/**
 * Trasferisce la memoria della RAM alla VRAM.
 * @param ramPointer La sorgente da cui copiare la memoria
 * @param gpuPointer La destinazione su cui copiare la memoria
 * @param size La dimensione in byte della memoria da copiare.
 */
void gpuCopyMemoryToDevice(void* ramPointer, void* gpuPointer, int size);

/**
 * Copia memoria dalla scheda alla scheda.
 * @param origin Il puntatore sorgente da cui copiare la memoria.
 * @param destination Il puntatore dove copiare la memoria.
 * @param size La dimensione in byte della memoria da copiare.
 */
void gpuCopyMemory(void* origin, void* destination, int size);

/**
 * Prodotto scalare tra due vettori calcolato in GPU.
 * @param vector_a Il primo vettore da moltiplicare.
 * @param vector_b Il secondo vettore da moltiplicare.
 * @param result   Il vettore risultato
 * @param length   La lunghezza dei vettori
 */
void gpuDotProduct(const double* vector_a, const double* vector_b,
                   double*       result,   int           length);
                   
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
                    double*       alpha);

void gpuScaleAndSumNeg(const double* vector_a, const double* vector_b, 
                    double*       result,   int           length, 
                    double*       alpha);


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
                    double*       result,     int           length);

/**
 * Divide i valori e li inserisce nel secondo. a / b
 * @param first Il primo valore (a)
 * @param second Il secondo valore (b) <- a / b
 */
void gpuDivide(double* first, double* second);

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
                       int*      row_offset, int*      column_index);

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
                      const double* vector_zk,  int             length);

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
                     double* values, int order, int non_zero);
///////////////////////////////////////////////////////////////////////////////
/// GRADIENTE CONIUGATO IN GPU!
typedef struct sparse_matrix
{
   // offset di riga
   int* row_offset;
   
   // indice di colonna
   int* column_index;
   
   // valori
   double* values;
   
   // ordine della matrice.
   int order;
   
}* sparse_matrix_t;

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
                          double* support);

#endif // GPU_CG__H__
