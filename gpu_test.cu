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
 * @file   gpu_test.cu
 * @author Alberto Franco
 * @date   10/05/2010
 * 
 * Implementazione del test per la GPU.
 */
#include "gpu_operations.h"

// Prototipo per il kernel di somma vettore 
__global__ void vectorAddGpu(double* a, double* b, double* result);

// Prototipo della funzione campione sulla CPU
void vectorAddCpu(double* a, double* b, double* result, int length);

// Prototipo della funzione di inizializzazione 
// dei vettori da sommare.
void initialize(double* a, double* b, int lenght);

// Prototipo della funzione di comparazione.
double compare(double* a, double* b, int length);

/**
 * Funzione di test della funzionalità di computazione
 * sulla GPU.
 */
int testGpuComputation()
{
   // La grandezza dei vettori
   const int length  = 256;
   const int size    = length * sizeof(double);
   
   // Dichiaro i puntatori che userò in GPU.
   double* device_a; 
   double* device_b;
   double* device_res;
   
   // Alloco la memoria sulla scheda video.
   cudaMalloc((void**)&device_a, size);
   cudaMalloc((void**)&device_b, size);
   cudaMalloc((void**)&device_res, size);
   
   // I puntatori dei vettori che restano in 
   // memoria centrale
   double* host_a = (double*)malloc(size);
   double* host_b = (double*)malloc(size);
   
   // I puntatori dei risultati da confrontare
   double* res_device   = (double*)malloc(size);
   double* res_host     = (double*)malloc(size);
   
   // Inizializzo i vettori
   initialize(host_a, host_b, length);
   
   // Copio i vettori appena inizializzati in gpu
   cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
   cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);
   
   // Lancio l'esecuzione del kernel
   vectorAddGpu<<<1, length>>> (device_a, device_b, device_res);
   
   // Eseguo la somma in CPU
   vectorAddCpu(host_a, host_b, res_host, length);
   
   // Trasferisco i dati dalla memoria video alla 
   // memoria principale.
   cudaMemcpy(res_device, device_res, size, cudaMemcpyDeviceToHost);
   
   // Confronto i risultati, se stanno sotto 
   // una certa tolleranza ritorno vero
   double error = compare(res_device, res_host, length);
   
   if(error < 1e-15)
      return true;
   
   // Altrimenti ritorno falso
   return false;
}

// Il kernel di somma vettore 
__global__ void vectorAddGpu(double* a, double* b, double* result)
{
   // Calcolo la posizione da sommare 
   int idx = threadIdx.x;
   // e sommo
   result[idx] = a[idx] + b[idx];
}

// La funzione campione sulla CPU
void vectorAddCpu(double* a, double* b, double* result, int length)
{
   for(int i = 0; i < length; i++) 
   {
      // Sommo una ad una le celle del vettore
      result[i] = a[i] + b[i];
   }
}

// Funzione di inizializzazione dei vettori da sommare.
void initialize(double* a, double* b, int length)
{
   for(int i = 0; i < length; i++) 
   {
      // Utilizzo la funzione random per generare i numeri
      a[i] = rand() % 300;
      b[i] = rand() % 300;
   }
}

// Funzione di comparazione.
double compare(double* a, double* b, int length)
{
   // Inizializzo la variabile d'errore
   double error = 0.0;
   
   // Calcolo l'errore per ogni somma
   for(int i = 0; i < length; i++)
   {
      // Calcolo l'errore assoluto della somma
      double currentError = fabs(a[i] - b[i]);
      
      // Confronto l'errore di questa cella con l'errore totale
      if(currentError > error)
         error = currentError;
   }
   
   return error;
}
