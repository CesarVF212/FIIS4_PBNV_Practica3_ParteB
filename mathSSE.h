#include <stdio.h>
#include <iostream>

// Estructura de matrices corregida
struct alignas(64) matrix4x4f {
    union {
        float m_grid[4][4]; 
        float m[16];        
    };
    static constexpr unsigned int size = 4;
};

// FUNCIÓN: multiplicación de matriz tradicional.
// Hacemos una versión que puede usarse tanto en GPU como en CPU.
inline __host__ __device__ matrix4x4f multMatrix_tradicional(const matrix4x4f &m1, const matrix4x4f &m2)
{
    matrix4x4f result = {}; // Inicializamos a cero para evitar basura
    
    // Usamos el size de la estructura (que es 4)
    unsigned int s = matrix4x4f::size;

    for(int i = 0; i < s; i++)      // i = Fila de m1
    {
        for(int j = 0; j < s; j++)  // j = Columna de m2
        {
            for(int k = 0; k < s; k++) // k = Índice compartido
            {
                result.m_grid[i][j] += m1.m_grid[i][k] * m2.m_grid[k][j];
            }
        }
    }
    return result;
}

inline __host__ void printMatrix(const matrix4x4f &matrix) 
{
    std::cout << "\n";
    
    for(int i = 0; i < 4; i++) 
    {
        std::cout << "[\t";
        for(int y = 0; y < 4; y++) 
            std::cout << matrix.m_grid[i][y] << "\t";
        
        std::cout << "]\n";
    }
    std::cout << "\n";
}