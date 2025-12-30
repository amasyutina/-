//умножение матриц (простой вариант)
//Реализует классическое умножение матриц на GPU без оптимизации.
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 16
#define BASE_TYPE double

// Макрос проверки ошибок
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s:%d: %s (code %d)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err), err); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Округление вверх до кратного BLOCK_SIZE
int toMultiple(int a, int b) {
    int mod = a % b;
    return mod == 0 ? a : a + (b - mod);
}
// Ядро: умножение матриц C = A × B
__global__ void matrixMult(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C,
                           int Acols, int Bcols) {
    // Вычисление глобального индекса строки и столбца в результирующей матрице C
    int i = blockDim.y * blockIdx.y + threadIdx.y;  // Индекс строки в C
    int j = blockDim.x * blockIdx.x + threadIdx.x;  // Индекс столбца в C

    // Проверка границ (если размеры не кратны BLOCK_SIZE)
    if (i >= gridDim.y * blockDim.y || j >= gridDim.x * blockDim.x) {
        return;  // Если индекс выходит за пределы, завершаем выполнение потока
    }

    BASE_TYPE sum = 0.0;  // Переменная для накопления суммы произведений
    // Основной цикл для умножения матриц
    for (int k = 0; k < Acols; ++k) {
        sum += A[i * Acols + k] * B[k * Bcols + j];  // Умножение и накопление результата
    }
    C[i * Bcols + j] = sum;  // Запись результата в матрицу C
}

int main() {
    printf("=== Matrix Multiplication (CUDA, Simple) ===\n");

    // 1. Исходные размеры матриц
    int Arows = 100, Acols = 200;  // Размеры матрицы A
    int Brows = Acols, Bcols = 150; // Размеры матрицы B (число строк B равно числу столбцов A)

    // 2. Приведение размеров к кратности BLOCK_SIZE
    Arows = toMultiple(Arows, BLOCK_SIZE);  // Округление количества строк матрицы A
    Acols = toMultiple(Acols, BLOCK_SIZE);  // Округление количества столбцов матрицы A
    Brows = toMultiple(Brows, BLOCK_SIZE);   // Округление количества строк матрицы B
    Bcols = toMultiple(Bcols, BLOCK_SIZE);   // Округление количества столбцов матрицы B

    printf("Matrix sizes: A(%d,%d) × B(%d,%d) = C(%d,%d)\n",
           Arows, Acols, Brows, Bcols, Arows, Bcols);  // Вывод размеров матриц

    // 3. Вычисление размеров в байтах
    size_t Asize = Arows * Acols * sizeof(BASE_TYPE);  // Размер матрицы A в байтах
    size_t Bsize = Brows * Bcols * sizeof(BASE_TYPE);  // Размер матрицы B в байтах
    size_t Csize = Arows * Bcols * sizeof(BASE_TYPE);  // Размер матрицы C в байтах

    // 4. Выделение памяти на CPU для матриц A, B и C
    BASE_TYPE *h_A = (BASE_TYPE*)malloc(Asize);
    BASE_TYPE *h_B = (BASE_TYPE*)malloc(Bsize);
    BASE_TYPE *h_C = (BASE_TYPE*)malloc(Csize);

    // Проверка успешности выделения памяти
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Error: malloc failed\n");  // Сообщение об ошибке при выделении памяти
        return EXIT_FAILURE;  // Завершение программы с кодом ошибки
    }

    // 5. Инициализация матриц случайными значениями [0, 1)
    for (int i = 0; i < Arows * Acols; ++i) {
        h_A[i] = rand() / (BASE_TYPE)RAND_MAX;  // Заполнение матрицы A случайными числами
    }
    for (int i = 0; i < Brows * Bcols; ++i) {
        h_B[i] = rand() / (BASE_TYPE)RAND_MAX;  // Заполнение матрицы B случайными числами
    }

    // 6. Выделение памяти на GPU для матриц A, B и C
    BASE_TYPE *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, Asize));  // Выделение памяти для d_A на GPU
    CUDA_CHECK(cudaMalloc((void**)&d_B, Bsize));  // Выделение памяти для d_B на GPU
    CUDA_CHECK(cudaMalloc((void**)&d_C, Csize));  // Выделение памяти для d_C на GPU

    // 7. Копирование данных с CPU на GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice));  // Копирование матрицы A на GPU
    CUDA_CHECK(cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice));  // Копирование матрицы B на GPU

    // 8. Конфигурация запуска ядра
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);  // Определение размера блока потоков (BLOCK_SIZE x BLOCK_SIZE)
    dim3 blocksPerGrid(Bcols / BLOCK_SIZE, Arows / BLOCK_SIZE);  // Определение количества блоков по сетке

    // 9. Замер времени выполнения ядра
    cudaEvent_t start, stop;  // События для замера времени выполнения
    CUDA_CHECK(cudaEventCreate(&start));  // Создание события для начала замера времени
    CUDA_CHECK(cudaEventCreate(&stop));   // Создание события для окончания замера времени

    CUDA_CHECK(cudaEventRecord(start));  // Запись начала замера времени
    matrixMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, Acols, Bcols);  // Запуск ядра умножения матриц
    CUDA_CHECK(cudaGetLastError());  // Проверка ошибок после запуска ядра
    CUDA_CHECK(cudaEventRecord(stop));  // Запись конца замера времени
    CUDA_CHECK(cudaDeviceSynchronize());  // Ожидание завершения всех потоков

    float kernelTime;  // Переменная для хранения времени выполнения ядра
    CUDA_CHECK(cudaEventElapsedTime(&kernelTime, start, stop));  // Вычисление времени выполнения ядра
    printf("Kernel execution time: %.2f ms\n", kernelTime);  // Вывод времени выполнения

    // 10. Копирование результата с GPU на CPU
    CUDA_CHECK(cudaMemcpy(h_C, d_C, Csize, cudaMemcpyDeviceToHost));  // Копирование результирующей матрицы C с GPU на CPU

    // 11. Проверка корректности результата умножения матриц
    printf("Starting verification...\n");
    bool passed = true;  // Флаг для проверки корректности результата
    for (int i = 0; i < Arows && passed; ++i) {  // Проход по строкам результата
        for (int j = 0; j < Bcols && passed; ++j) {  // Проход по столбцам результата
            BASE_TYPE expected = 0.0;  // Переменная для хранения ожидаемого значения
            for (int k = 0; k < Acols; ++k) {  // Цикл для вычисления ожидаемого значения
                expected += h_A[i * Acols + k] * h_B[k * Bcols + j];  // Умножение и накопление результата
            }
            // Проверка совпадения результата с ожидаемым значением
            if (fabs(h_C[i * Bcols + j] - expected) > 1e-3) {
                fprintf(stderr, "Verification failed at [%d,%d]: "
                        "got %.6f, expected %.6f\n",
                        i, j, h_C[i * Bcols + j], expected);
                passed = false;  // Установка флага несоответствия
            }
        }
    }
    printf("Verification: %s\n", passed ? "PASSED" : "FAILED");  // Вывод результата проверки

    // 12. Освобождение ресурсов
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
