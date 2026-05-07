# Práctica 3-B: Multiplicación de matrices en GPU con memoria compartida

**Autor:** Caesar
**Asignatura:** Programación de Bajo Nivel
**Entrega:** PBN_PR3B_NOMBRE_APELLIDO.zip

## Descripción

Esta práctica implementa la multiplicación de matrices 4x4 en GPU usando CUDA, ampliando la Parte A con una versión que aprovecha **memoria compartida**. Se comparan tres implementaciones:

1. **CPU tradicional** (referencia, 1.000 matrices).
2. **GPU sin memoria compartida** (1 thread por matriz, 1.000.000 de matrices).
3. **GPU con memoria compartida** (16 threads por matriz, 1 bloque por matriz, 1.000.000 de matrices).

## Requisitos

- NVIDIA CUDA Toolkit 13.x (o compatible)
- GPU NVIDIA con compute capability ≥ 5.0
- Compilador `nvcc`
- Linux / WSL2 / Windows

## Estructura

```
ParteB/
├── main.cu          # Programa principal con las tres implementaciones
├── mathSSE.h        # Definición de matrix4x4f y multMatrix_tradicional
├── mathSSE.cpp      # Implementación auxiliar
└── README.md
```

## Compilación

```bash
nvcc main.cu mathSSE.cpp -o main
```

## Ejecución

```bash
./main
```

## Implementación

### Kernel sin memoria compartida (Parte A, referencia)

Un thread procesa una matriz completa llamando a `multMatrix_tradicional`:

```cpp
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
AResultadosGPU_d[i] = multMatrix_tradicional(A1_d[i], A2_d[i]);
```

### Kernel con memoria compartida (Parte B)

- **Un bloque por matriz**, **16 threads por bloque** (uno por casilla del resultado).
- Cada thread carga una casilla de cada matriz a memoria compartida en paralelo.
- La segunda matriz se almacena **traspuesta** para que el producto sea fila × fila (más sencillo).
- `__syncthreads()` asegura que la carga colaborativa termine antes de operar.
- Cada thread calcula su casilla `[fila][columna]` del resultado leyendo solo de memoria compartida.

```cpp
__shared__ float M1[4][4];
__shared__ float M2T[4][4];

unsigned int fila    = threadIdx.x / 4;
unsigned int columna = threadIdx.x % 4;

M1[fila][columna]   = A1_d[blockIdx.x].m_grid[fila][columna];
M2T[columna][fila]  = A2_d[blockIdx.x].m_grid[fila][columna];
__syncthreads();

float suma = 0.0f;
for(int k = 0; k < 4; k++)
    suma += M1[fila][k] * M2T[columna][k];

AResultadosGPU_d[blockIdx.x].m_grid[fila][columna] = suma;
```

## Toma de tiempos

Se utiliza `std::chrono::high_resolution_clock` para CPU y se incluye `cudaDeviceSynchronize()` antes de cada `TIME_END` en GPU para asegurar que se mide el tiempo real (las llamadas CUDA son asíncronas). Se realiza un _warm-up_ con `cudaFree(0)` al inicio para que la primera medida no incluya la inicialización del contexto CUDA.

Dentro de la versión SM se desglosan los tiempos en:

- **Copia H2D** (host → device)
- **Ejecución del kernel**
- **Copia D2H** (device → host)

## Resultados obtenidos

GPU: NVIDIA GeForce RTX 3070 (driver 596.36, CUDA 13.2), Ubuntu 26.04 sobre WSL2.

| Versión                    | Tiempo (1.000.000 matrices) | Speedup vs CPU |
| -------------------------- | --------------------------- | -------------- |
| CPU (extrapolado de 1.000) | 177.7 ms                    | 1.00×          |
| GPU sin memoria compartida | ~184 ms                     | 0.96×          |
| GPU con memoria compartida | ~78 ms                      | **2.26×**      |

**Speedup GPU_SM vs GPU (sin SM): 2.34×**

## Análisis

- **La versión sin memoria compartida apenas iguala a la CPU.** Cada thread realiza 16 multiplicaciones-suma leyendo de memoria global sin patrón coalescido, y a eso se suma el coste de transferencia PCIe.
- **La versión con memoria compartida supera claramente a la CPU (2.26×)** y a la versión GPU sin SM (2.34×). La mejora viene de:
  - Mayor paralelismo: 16M threads vs 1M.
  - Accesos coalescidos a memoria global durante la carga colaborativa.
  - Bucle de producto sobre memoria compartida (≈100× más rápida que global).
  - La traspuesta de M2 da accesos contiguos en SM.
- **El speedup no llega al teórico 16×** porque las copias H2D/D2H dominan el tiempo total. El kernel puro es muy rápido; el cuello de botella es el bus PCIe, tal como anticipa el enunciado.

## Configuración óptima

- Versión sin SM: probada con 32, 64, 128, 256, 512 threads/bloque. La mejor configuración fue **256 threads/bloque**.
- Versión con SM: tamaño fijado por el enunciado (16 threads/bloque, 1 bloque por matriz).

Cada medida se repitió varias veces y se reportó el valor más representativo, descartando la primera ejecución (más lenta por la inicialización).

## Notas

- `multMatrix_tradicional` se marcó como `__host__ __device__` para poder reutilizarse desde la versión GPU sin SM.
- Para la versión SM se implementó el producto manualmente dentro del kernel en lugar de llamar a `multMatrix_tradicional`, ya que el cómputo se reparte entre 16 threads.
- El número de matrices (1.000.000) está dentro del límite de bloques permitido por CUDA (`MAX_INTEGER`).
