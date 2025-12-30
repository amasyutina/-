//сложение матриц
//Демонстрирует параллельное сложение двух матриц на GPU.
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

// Константы конфигурации
#define BLOCK_SIZE 16          // Размер блока потоков (16×16)
#define ROWS 100               // Исходная высота матрицы
#define COLS 200              // Исходная ширина матрицы
#define BASE_TYPE float       // Тип данных элементов матрицы

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s:%d: %s (code %d)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err), err); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Функция округления размера до ближайшего кратного BLOCK_SIZE
inline int toMultiple(int a, int b) {
    return (a + b - 1) / b * b;  // Округление a до ближайшего кратного b
}

// Ядро CUDA: сложение матриц A + B = C
__global__ void matrixAdd(const BASE_TYPE* A, const BASE_TYPE* B, BASE_TYPE* C,
                           int rows, int cols) {
    // Вычисление глобальных индексов элемента в результирующей матрице
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // Индекс столбца
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // Индекс строки

    // Проверка выхода за границы матрицы
    if (x < cols && y < rows) {
        int idx = y * cols + x;  // Линейный индекс элемента в матрице
        C[idx] = A[idx] + B[idx];  // Сложение элементов матриц A и B и сохранение результата в C
    }
}

int main() {
    printf("=== Matrix Addition (CUDA) ===\n");

    // 1. Подготовка размеров (округление до кратных BLOCK_SIZE)
    int rows = toMultiple(ROWS, BLOCK_SIZE);  // Округление количества строк
    int cols = toMultiple(COLS, BLOCK_SIZE);  // Округление количества столбцов
    size_t size = rows * cols * sizeof(BASE_TYPE);  // Вычисление общего размера памяти для матриц

    printf("Matrix size: %d × %d\n", rows, cols);  // Вывод размеров матрицы

    // 2. Выделение памяти на CPU для матриц A, B и C
    BASE_TYPE *h_A = (BASE_TYPE*)malloc(size);
    BASE_TYPE *h_B = (BASE_TYPE*)malloc(size);
    BASE_TYPE *h_C = (BASE_TYPE*)malloc(size);

    // Проверка успешности выделения памяти
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Error: malloc failed\n");  // Сообщение об ошибке при выделении памяти
        return EXIT_FAILURE;  // Завершение программы с кодом ошибки
    }

    // 3. Инициализация матриц случайными значениями [0, 1)
    for (int i = 0; i < rows * cols; ++i) {
        h_A[i] = (BASE_TYPE)rand() / RAND_MAX;  // Заполнение матрицы A случайными числами
        h_B[i] = (BASE_TYPE)rand() / RAND_MAX;  // Заполнение матрицы B случайными числами
    }

    // 4. Выделение памяти на GPU для матриц A, B и C
    BASE_TYPE *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));  // Выделение памяти для d_A на GPU
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));  // Выделение памяти для d_B на GPU
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));  // Выделение памяти для d_C на GPU

    // 5. Копирование данных с CPU на GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));  // Копирование матрицы A на GPU
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));  // Копирование матрицы B на GPU

    // 6. Конфигурация запуска ядра
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);  // Определение размера блока потоков (BLOCK_SIZE x BLOCK_SIZE)
    dim3 gridSize((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,  // Вычисление количества блоков по ширине (X)
                  (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);   // Вычисление количества блоков по высоте (Y)

    // 7. Замер времени выполнения ядра
    cudaEvent_t start, stop;  // События для замера времени выполнения
    CUDA_CHECK(cudaEventCreate(&start));  // Создание события для начала замера времени
    CUDA_CHECK(cudaEventCreate(&stop));   // Создание события для окончания замера времени

    CUDA_CHECK(cudaEventRecord(start));  // Запись начала замера времени
    matrixAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);  // Запуск ядра сложения матриц
    CUDA_CHECK(cudaGetLastError());  // Проверка ошибок после запуска ядра
    CUDA_CHECK(cudaEventRecord(stop));  // Запись конца замера времени
    CUDA_CHECK(cudaDeviceSynchronize());  // Ожидание завершения всех потоков

    float time_ms;  // Переменная для хранения времени выполнения
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));  // Вычисление времени выполнения ядра
    printf("Matrix addition (%dx%d) time: %.3f ms\n", rows, cols, time_ms);  // Вывод времени выполнения

    // 8. Копирование результата с GPU на CPU
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));  // Копирование результирующей матрицы C с GPU на CPU

    // 9. Проверка корректности результата сложения матриц
    bool ok = true;  // Флаг для проверки корректности результата
    for (int i = 0; i < rows * cols && ok; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f) {  // Проверка совпадения результата с ожидаемым значением
            ok = false;  // Установка флага несоответствия
        }
    }
    printf("Verification: %s\n", ok ? "PASSED" : "FAILED");  // Вывод результата проверки

    // 10. Освобождение ресурсов: память на GPU и CPU
    CUDA_CHECK(cudaFree(d_A));  // Освобождение памяти на GPU для d_A
    CUDA_CHECK(cudaFree(d_B));  // Освобождение памяти на GPU для d_B
    CUDA_CHECK(cudaFree(d_C));  // Освобождение памяти на GPU для d_C
    free(h_A);                   // Освобождение памяти на CPU для h_A
    free(h_B);                   // Освобождение памяти на CPU для h_B
    free(h_C);                   // Освобождение памяти на CPU для h_C
    CUDA_CHECK(cudaEventDestroy(start));  // Уничтожение события начала замера времени
    CUDA_CHECK(cudaEventDestroy(stop));   // Уничтожение события конца замера времени

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;  // Возврат кода завершения программы в зависимости от проверки корректности
}


