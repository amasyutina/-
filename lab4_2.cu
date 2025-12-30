//транспонирование матрицы
//Транспонирует матрицу на GPU, замеряет время, проверяет результат
#include "cuda_runtime.h"  // Подключение библиотеки для работы с CUDA
#include "device_launch_parameters.h"  // Подключение параметров запуска устройства
#include <stdio.h>  // Подключение стандартной библиотеки ввода-вывода
#include <stdlib.h>  // Подключение стандартной библиотеки для работы с памятью и случайными числами

#define BLOCK_SIZE 16  // Размер блока потоков
#define BASE_TYPE float  // Определение типа данных для матрицы

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call) 
do { 
    cudaError_t err = call; 
    if (err != cudaSuccess) { 
        fprintf(stderr, "[CUDA ERROR] %s:%d: %s (code %d)\n", 
                __FILE__, __LINE__, cudaGetErrorString(err), err); 
        exit(EXIT_FAILURE); 
    } 
} while (0)

// Ядро для транспонирования матрицы
__global__ void matrixTranspose(const BASE_TYPE *A, BASE_TYPE *AT, int rows, int cols) {
    // Вычисление глобальных индексов строки и столбца для потока
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка, что индексы находятся в пределах размеров матрицы
    if (row < rows && col < cols) {
        int idx_A = row * cols + col;  // Линейный индекс в исходной матрице
        int idx_AT = col * rows + row;  // Линейный индекс в транспонированной матрице
        AT[idx_AT] = A[idx_A];  // Транспонирование
    }
}

// Функция для округления до ближайшего кратного значения
int toMultiple(int a, int b) {
    return (a + b - 1) / b * b;  // Округление a до ближайшего кратного b
}

int main() {
    printf("=== Matrix Transpose (CUDA) ===\n");

    int rows = 1000, cols = 2000;  // Размеры матрицы
    rows = toMultiple(rows, BLOCK_SIZE);  // Округление строк до кратного BLOCK_SIZE
    cols = toMultiple(cols, BLOCK_SIZE);  // Округление столбцов до кратного BLOCK_SIZE
    size_t size = rows * cols * sizeof(BASE_TYPE);  // Общий размер памяти для матрицы

    printf("Adjusted size: %d x %d\n", rows, cols);  // Вывод скорректированных размеров матрицы

    // 1. Выделение памяти на CPU для исходной и транспонированной матриц
    BASE_TYPE *h_A = (BASE_TYPE*)malloc(size);
    BASE_TYPE *h_AT = (BASE_TYPE*)malloc(size);
    if (!h_A || !h_AT) {  // Проверка успешности выделения памяти
        fprintf(stderr, "Error: malloc failed\n");
        return EXIT_FAILURE;  // Выход при ошибке выделения памяти
    }

    // 2. Инициализация исходной матрицы случайными значениями
    for (int i = 0; i < rows * cols; i++) {
        h_A[i] = rand() / (BASE_TYPE)RAND_MAX;  // Заполнение матрицы случайными числами от 0 до 1
    }

    // 3. Выделение памяти на GPU для исходной и транспонированной матриц
    BASE_TYPE *d_A, *d_AT;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));  // Выделение памяти для d_A на GPU
    CUDA_CHECK(cudaMalloc((void**)&d_AT, size));  // Выделение памяти для d_AT на GPU

    // 4. Копирование исходной матрицы с CPU на GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    // 5. Конфигурация ядра: размер блока и сетки
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);  // Размер блока потоков (BLOCK_SIZE x BLOCK_SIZE)
    dim3 gridSize((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,  // Количество блоков по ширине
                  (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);  // Количество блоков по высоте

    // 6. Замер времени выполнения ядра
    cudaEvent_t start, stop;  // События для замера времени
    CUDA_CHECK(cudaEventCreate(&start));  // Создание события для начала замера
    CUDA_CHECK(cudaEventCreate(&stop));   // Создание события для окончания замера

    CUDA_CHECK(cudaEventRecord(start));  // Запись начала замера времени
    matrixTranspose<<<gridSize, blockSize>>>(d_A, d_AT, rows, cols);  // Запуск ядра транспонирования
    CUDA_CHECK(cudaGetLastError());  // Проверка ошибок после запуска ядра
    CUDA_CHECK(cudaEventRecord(stop));  // Запись конца замера времени
    CUDA_CHECK(cudaDeviceSynchronize());  // Ожидание завершения всех потоков
    float kernelTime;  // Переменная для хранения времени выполнения ядра
    CUDA_CHECK(cudaEventElapsedTime(&kernelTime, start, stop));  // Вычисление времени выполнения ядра
    printf("Kernel execution time: %.2f ms\n", kernelTime);  // Вывод времени выполнения ядра

    // 7. Копирование результата с GPU на CPU
    CUDA_CHECK(cudaMemcpy(h_AT, d_AT, size, cudaMemcpyDeviceToHost));

    // 8. Проверка корректности транспонирования
    bool correct = true;  // Флаг для проверки корректности результата
    for (int i = 0; i < rows && correct; i++) {
        for (int j = 0; j < cols && correct; j++) {
            BASE_TYPE expected = h_A[i * cols + j];  // Ожидаемое значение в исходной матрице
            BASE_TYPE actual = h_AT[j * rows + i];   // Фактическое значение в транспонированной матрице
            if (fabs(expected - actual) > 1e-6) {  // Проверка на совпадение значений с учетом погрешности
                printf("Error at [%d,%d]: expected %.6f, got %.6f\n",
                       i, j, expected, actual);  // Вывод информации о несоответствии
                correct = false;  // Установка флага несоответствия
            }
        }
    }

    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");  // Вывод результата проверки

    // 9. Освобождение ресурсов: память на GPU и CPU
    CUDA_CHECK(cudaFree(d_A));  // Освобождение памяти на GPU для d_A
    CUDA_CHECK(cudaFree(d_AT));  // Освобождение памяти на GPU для d_AT
    free(h_A);                   // Освобождение памяти на CPU для h_A
    free(h_AT);                  // Освобождение памяти на CPU для h_AT
    CUDA_CHECK(cudaEventDestroy(start));  // Уничтожение события начала замера времени
    CUDA_CHECK(cudaEventDestroy(stop));   // Уничтожение события конца замера времени

    return correct ? 0 : EXIT_FAILURE;  // Возврат кода завершения программы в зависимости от проверки корректности
}