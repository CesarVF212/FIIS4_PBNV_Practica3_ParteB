#include "mathSSE.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <cuda_runtime.h>

// Autor: Caesar

//tomar tiempos de inicio
#define TIME_INIT(TimeSuffix) \
    auto start_##TimeSuffix = std::chrono::high_resolution_clock::now();
//tomar tiempos de final y calcular milisegundos
#define TIME_END(TimeSuffix) \
    auto end_##TimeSuffix = std::chrono::high_resolution_clock::now();\
    std::chrono::duration<double> duration_##TimeSuffix = end_##TimeSuffix - start_##TimeSuffix;\
    double milliseconds_##TimeSuffix = duration_##TimeSuffix / 1ms;
#define PRETTY_PRINT_TIME(TimeSuffix, message) \
    std::cout << "Execution time "<<message<<" :" << milliseconds_##TimeSuffix << " milliseconds." << std::endl;

using namespace std::chrono_literals;

#define NUM_MATRIX 1000
#define NUM_MATRIX_GPU 1000000

#define THREADS_PER_BLOCK 256
#define THREADS_PER_BLOCK_SM 16

matrix4x4f* fillRandom(matrix4x4f* matrix, unsigned int arraySize)
{
    srand(time(NULL));
    for(unsigned int iter = 0; iter < arraySize; iter++)
    {
        for(unsigned int i = 0; i < matrix[0].size; i++)
        {
            for(unsigned int j = 0; j < matrix[0].size; j++)
            {
                float r = ((float)rand() / RAND_MAX) * 200.0f - 100.0f;
                matrix[iter].m_grid[i][j] = r;
            }
        }
    }
    return matrix;
}

// --- KERNEL PARTE 1 (un thread por matriz) --- //
__global__ void multiplicaMatricesGPU_kernel(matrix4x4f* A1_d, matrix4x4f* A2_d, matrix4x4f* AResultadosGPU_d, unsigned int arraySize)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < arraySize)
    {
        AResultadosGPU_d[i] = multMatrix_tradicional(A1_d[i], A2_d[i]);
    }
}

void multiplicaMatricesGPU(matrix4x4f* A1, matrix4x4f* A2, matrix4x4f* AResultadosGPU_h, unsigned int arraySize)
{
    matrix4x4f* A1_d;
    matrix4x4f* A2_d;
    matrix4x4f* AResultadosGPU_d;

    size_t bytes = arraySize * sizeof(matrix4x4f);

    cudaMalloc((void**)&A1_d, bytes);
    cudaMalloc((void**)&A2_d, bytes);
    cudaMalloc((void**)&AResultadosGPU_d, bytes);

    cudaMemcpy(A1_d, A1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(A2_d, A2, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = THREADS_PER_BLOCK;
    int numBlocks = (arraySize + threadsPerBlock - 1) / threadsPerBlock;

    multiplicaMatricesGPU_kernel<<<numBlocks, threadsPerBlock>>>(A1_d, A2_d, AResultadosGPU_d, arraySize);
    cudaDeviceSynchronize();

    cudaMemcpy(AResultadosGPU_h, AResultadosGPU_d, bytes, cudaMemcpyDeviceToHost);

    cudaFree(A1_d);
    cudaFree(A2_d);
    cudaFree(AResultadosGPU_d);
}

// --- KERNEL PARTE 2 (16 threads por matriz, memoria compartida) --- //
// Cada bloque procesa UNA matriz. Cada thread calcula UNA casilla del resultado.
__global__ void multiplicaMatricesGPU_kernelSM(matrix4x4f* A1_d, matrix4x4f* A2_d, matrix4x4f* AResultadosGPU_d, unsigned int arraySize)
{
    // Memoria compartida para las dos matrices del bloque.
    // M2 se guarda traspuesta para que filaM1 * filaM2T sea producto fila*fila.
    __shared__ float M1[4][4];
    __shared__ float M2T[4][4];

    unsigned int matIdx = blockIdx.x;
    if(matIdx >= arraySize) return;

    unsigned int tid = threadIdx.x;       // 0..15
    unsigned int fila = tid / 4;
    unsigned int columna = tid % 4;

    // Cada thread copia una casilla de M1 y una de M2 (traspuesta) a memoria compartida.
    M1[fila][columna] = A1_d[matIdx].m_grid[fila][columna];
    M2T[columna][fila] = A2_d[matIdx].m_grid[fila][columna];

    // Sincronizamos para asegurar que todos los datos estan en memoria compartida.
    __syncthreads();

    // Calculamos el elemento [fila][columna] del resultado.
    // Como M2 esta traspuesta, multiplicamos fila por fila.
    float suma = 0.0f;
    for(int k = 0; k < 4; k++)
    {
        suma += M1[fila][k] * M2T[columna][k];
    }

    // Escribimos el resultado directamente a memoria global.
    AResultadosGPU_d[matIdx].m_grid[fila][columna] = suma;
}

void multiplicaMatricesGPU_SM(matrix4x4f* A1, matrix4x4f* A2, matrix4x4f* AResultadosGPU_h, unsigned int arraySize)
{
    matrix4x4f* A1_d;
    matrix4x4f* A2_d;
    matrix4x4f* AResultadosGPU_d;

    size_t bytes = arraySize * sizeof(matrix4x4f);

    // Reservamos memoria de GPU.
    cudaMalloc((void**)&A1_d, bytes);
    cudaMalloc((void**)&A2_d, bytes);
    cudaMalloc((void**)&AResultadosGPU_d, bytes);

    // Copia de datos a GPU con sincronizacion para medir tiempos correctamente.
    TIME_INIT(GPU_SM_CopyIn);
    cudaMemcpy(A1_d, A1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(A2_d, A2, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    TIME_END(GPU_SM_CopyIn);
    PRETTY_PRINT_TIME(GPU_SM_CopyIn, "  copia H2D (SM)");

    // 16 threads por bloque, un bloque por matriz.
    int threadsPerBlock = THREADS_PER_BLOCK_SM;
    unsigned int numBlocks = arraySize;   // un bloque = una matriz

    // Lanzamos el kernel y medimos solo la ejecucion.
    TIME_INIT(GPU_SM_Kernel);
    multiplicaMatricesGPU_kernelSM<<<numBlocks, threadsPerBlock>>>(A1_d, A2_d, AResultadosGPU_d, arraySize);
    cudaDeviceSynchronize();
    TIME_END(GPU_SM_Kernel);
    PRETTY_PRINT_TIME(GPU_SM_Kernel, "  kernel SM");

    // Copia de resultados de vuelta al host.
    TIME_INIT(GPU_SM_CopyOut);
    cudaMemcpy(AResultadosGPU_h, AResultadosGPU_d, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    TIME_END(GPU_SM_CopyOut);
    PRETTY_PRINT_TIME(GPU_SM_CopyOut, "  copia D2H (SM)");

    cudaFree(A1_d);
    cudaFree(A2_d);
    cudaFree(AResultadosGPU_d);
}

int main(int args, char** argv)
{
    // Calentamiento de la GPU para que la primera medida no incluya la inicializacion del contexto.
    cudaFree(0);

    // --- PARTE CPU (referencia) --- //

    matrix4x4f* A1 = new matrix4x4f[NUM_MATRIX];
    matrix4x4f* A2 = new matrix4x4f[NUM_MATRIX];
    matrix4x4f* AResultados = new matrix4x4f[NUM_MATRIX];
    A1 = fillRandom(A1, NUM_MATRIX);
    A2 = fillRandom(A2, NUM_MATRIX);

    TIME_INIT(MultiplicarMatricesCPU);
    for(int iter = 0; iter < NUM_MATRIX; iter++)
    {
        AResultados[iter] = multMatrix_tradicional(A1[iter], A2[iter]);
    }
    TIME_END(MultiplicarMatricesCPU);
    PRETTY_PRINT_TIME(MultiplicarMatricesCPU,"Tiempo en CPU");

    // --- DATOS PARA GPU --- //

    matrix4x4f* A1_GPU = new matrix4x4f[NUM_MATRIX_GPU];
    matrix4x4f* A2_GPU = new matrix4x4f[NUM_MATRIX_GPU];
    matrix4x4f* AResultadosGPU_h = new matrix4x4f[NUM_MATRIX_GPU];
    matrix4x4f* AResultadosGPU_SM_h = new matrix4x4f[NUM_MATRIX_GPU];

    A1_GPU = fillRandom(A1_GPU, NUM_MATRIX_GPU);
    A2_GPU = fillRandom(A2_GPU, NUM_MATRIX_GPU);

    // --- PARTE 1 - GPU SIN MEMORIA COMPARTIDA --- //

    TIME_INIT(MultiplicarMatricesGPU);
    multiplicaMatricesGPU(A1_GPU, A2_GPU, AResultadosGPU_h, NUM_MATRIX_GPU);
    TIME_END(MultiplicarMatricesGPU);
    PRETTY_PRINT_TIME(MultiplicarMatricesGPU,"Tiempo en GPU (sin SM)");

    // --- PARTE 2 - GPU CON MEMORIA COMPARTIDA --- //

    TIME_INIT(MultiplicarMatricesGPU_SM);
    multiplicaMatricesGPU_SM(A1_GPU, A2_GPU, AResultadosGPU_SM_h, NUM_MATRIX_GPU);
    TIME_END(MultiplicarMatricesGPU_SM);
    PRETTY_PRINT_TIME(MultiplicarMatricesGPU_SM,"Tiempo en GPU (con SM)");

    // --- SPEEDUP --- //
    double cpuEscalado = milliseconds_MultiplicarMatricesCPU * (double)(NUM_MATRIX_GPU / NUM_MATRIX);
    double speedup_GPU = cpuEscalado / milliseconds_MultiplicarMatricesGPU;
    double speedup_GPU_SM = cpuEscalado / milliseconds_MultiplicarMatricesGPU_SM;
    double speedup_SM_vs_GPU = milliseconds_MultiplicarMatricesGPU / milliseconds_MultiplicarMatricesGPU_SM;

    std::cout << "----------------------------------" << std::endl;
    std::cout << "Tiempo CPU extrapolado a " << NUM_MATRIX_GPU << " matrices: " << cpuEscalado << " ms" << std::endl;
    std::cout << "Speedup CPU/GPU (sin SM): " << speedup_GPU << std::endl;
    std::cout << "Speedup CPU/GPU (con SM): " << speedup_GPU_SM << std::endl;
    std::cout << "Speedup GPU/GPU_SM: " << speedup_SM_vs_GPU << std::endl;

    // Liberamos memoria.
    delete[] A1;
    delete[] A2;
    delete[] AResultados;
    delete[] A1_GPU;
    delete[] A2_GPU;
    delete[] AResultadosGPU_h;
    delete[] AResultadosGPU_SM_h;

    return 0;
}