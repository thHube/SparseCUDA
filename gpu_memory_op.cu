/**
 * @file   gpu_memory_op.cu
 * @author Alberto Franco
 * @date   19/05/2010
 * 
 * Contiene l'implementazione delle funzioni di gestione della memoria per il
 * metodo del gradiente coniugato.
 */
#include "gpu_cg.h"

/**
 * Alloca sulla scheda grafica un segmento di memoria lungo quanto richiesto.
 * @param size La dimensione in byte del segmento.
 * @return Il puntatore alla zona di memoria.
 */
void* gpuAllocate(int size)
{
   // Uso un puntatore temporaneo
   void* memory_pointer;
   
   // Richiedo l'allocazione di memoria
   cudaMalloc(&memory_pointer, size);
   
   // Ritorno la nuova area di memoria allocata.
   return memory_pointer;
}

/**
 * Libera la memoria allocata puntata dal parametro.
 * @param memoryPointer il puntatore alla zona di memoria da liberare
 */
void gpuDelete(void* memoryPointer)
{
   // Libero la zona di memoria puntata
   cudaFree(memoryPointer);
}


/**
 * Copia la memoria sull'host e ritorna il puntatore alla zona 
 * di memoria allocata.
 * @param gpuPointer Il punto da cui copiare la memoria.
 * @param size La dimensione in byte della memoria da copiare.
 * @return Il puntatore alla zona di memoria con i dati copiati in RAM.
 */
void* gpuCopyMemoryToHost(void* gpuPointer, int size)
{
   // Mi serve un puntatore temporaneo
   void* ram_pointer = malloc(size);
   
   // Copio la memoria sull'area appena allocata
   cudaMemcpy(ram_pointer, gpuPointer, size, cudaMemcpyDeviceToHost);
   
   // Ritorno il puntatore alla memoria RAM.
   return ram_pointer;
}


/**
 * Trasferisce la memoria della RAM alla VRAM.
 * @param ramPointer La sorgente da cui copiare la memoria
 * @param gpuPointer La destinazione su cui copiare la memoria
 * @param size La dimensione in byte della memoria da copiare.
 */
void gpuCopyMemoryToDevice(void* ramPointer, void* gpuPointer, int size)
{
   // Invoco la funzione di copia
   cudaMemcpy(gpuPointer, ramPointer, size, cudaMemcpyHostToDevice);

}

/**
 * Copia memoria dalla scheda alla scheda.
 * @param origin Il puntatore sorgente da cui copiare la memoria.
 * @param destination Il puntatore dove copiare la memoria.
 * @param size La dimensione in byte della memoria da copiare.
 */
void gpuCopyMemory(void* origin, void* destination, int size)
{
   // Invoco la funzione di copia
   cudaMemcpy(destination, origin, size, cudaMemcpyDeviceToDevice);
}


