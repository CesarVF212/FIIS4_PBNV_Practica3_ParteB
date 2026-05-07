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

## Notas

- `multMatrix_tradicional` se marcó como `__host__ __device__` para poder reutilizarse desde la versión GPU sin SM.
- Para la versión SM se implementó el producto manualmente dentro del kernel en lugar de llamar a `multMatrix_tradicional`, ya que el cómputo se reparte entre 16 threads.
- El número de matrices (1.000.000) está dentro del límite de bloques permitido por CUDA (`MAX_INTEGER`).
