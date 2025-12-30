//заполнение матрицы на GPU
//Назначение:
//Демонстрирует базовую работу с CUDA: выделение памяти, запуск ядра, копирование данных, замер времени

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

// Макрос для проверки ошибок CUDA
// - Выполняет вызов CUDA-функции
// - Проверяет код возврата на cudaSuccess
// - При ошибке: выводит файл, строку и описание ошибки, завершает программу

#define CUDA_CHECK(call) 
do { 
    cudaError_t err = call; 
    if (err != cudaSuccess) { 
        fprintf(stderr, "[CUDA ERROR] %s:%d: %s (code %d)\n", 
                __FILE__, __LINE__, cudaGetErrorString(err), err); 
        exit(EXIT_FAILURE); 
    } 
} while (0)

// Ядро: заполняет матрицу по правилу A[i][j] = 10*i + j
__global__ void createMatrix(int *A, const int n) {
    int row = threadIdx.y;  // номер строки потока в блоке
    int col = threadIdx.x;  // номер столбца потока в блоке
    
    // Проверка, что индексы не выходят за границы матрицы
    if (row < n && col < n) {  
        int idx = row * n + col;  // вычисление линейного индекса
        A[idx] = 10 * row + col;   // заполнение матрицы по заданному правилу
    }
}

int main() {
    const int n = 10;  // размерность матрицы (n x n)
    size_t size = n * n * sizeof(int);  // размер памяти для хранения матрицы

    // 1. Выделение памяти на CPU для матрицы h_A
    int *h_A = (int*)malloc(size);
    if (!h_A) {
        fprintf(stderr, "Error: malloc failed for h_A\n");
        return EXIT_FAILURE;  // выход при ошибке выделения памяти
    }

    // 2. Инициализация на CPU (аналогично ядру)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_A[i * n + j] = 10 * i + j;  // заполнение матрицы на CPU
        }
    }

    // 3. Выделение памяти на GPU для матрицы d_B
    int *d_B = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));  // выделение памяти на устройстве

    // 4. Конфигурация запуска ядра
    dim3 threadsPerBlock(10, 10);  // количество потоков в блоке (10x10)
    dim3 blocksPerGrid(1, 1);       // количество блоков в сетке (1 блок)

    // 5. Замер времени выполнения ядра
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));  // создание события для начала замера
    CUDA_CHECK(cudaEventCreate(&stop));   // создание события для конца замера

    CUDA_CHECK(cudaEventRecord(start));  // начало замера времени
    createMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_B, n);  // запуск ядра
    CUDA_CHECK(cudaGetLastError());  // проверка ошибок запуска ядра
    CUDA_CHECK(cudaEventRecord(stop));  // окончание замера времени
    CUDA_CHECK(cudaDeviceSynchronize());  // ожидание завершения всех потоков

    float kernelTimeMs;  // переменная для хранения времени выполнения ядра
    CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, start, stop));  // вычисление времени выполнения

    // 6. Копирование результата с GPU на CPU
    int *h_B = (int*)malloc(size);  // выделение памяти на CPU для результата
    if (!h_B) {
        fprintf(stderr, "Error: malloc failed for h_B\n");
        CUDA_CHECK(cudaFree(d_B));  // освобождение памяти на GPU при ошибке
        free(h_A);                  // освобождение памяти на CPU
        return EXIT_FAILURE;        // выход при ошибке выделения памяти
    }
    CUDA_CHECK(cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost));  // копирование данных с GPU

    // 7. Проверка корректности полученных данных
    bool match = true;  // флаг для проверки совпадения матриц
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (h_A[i * n + j] != h_B[i * n + j]) {  // сравнение значений
                printf("Mismatch at [%d][%d]: CPU=%d, GPU=%d\n",
                       i, j, h_A[i * n + j], h_B[i * n + j]);  // вывод информации о несоответствии
                match = false;  // установка флага несоответствия
            }
        }
    }

    printf("Matrices %s!\n", match ? "match" : "do NOT match");  // вывод результата проверки
    printf("Kernel time: %.6f ms\n", kernelTimeMs);  // вывод времени выполнения ядра

    // 8. Освобождение ресурсов
    CUDA_CHECK(cudaFree(d_B));  // освобождение памяти на GPU
    free(h_A);                  // освобождение памяти на CPU для h_A
    free(h_B);                  // освобождение памяти на CPU для h_B
    CUDA_CHECK(cudaEventDestroy(start));  // уничтожение события начала замера
    CUDA_CHECK(cudaEventDestroy(stop));   // уничтожение события конца замера

    return 0;  // успешное завершение программы
}