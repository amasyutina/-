//оптимизированное умножение матриц:
//Использует разделяемую память для кэширования подматриц,
// Разбивает вычисление на тайлы (блоки),
// Минимизирует обращения к глобальной памяти.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 16
#define BASE_TYPE double

// Макрос проверки ошибок CUDA
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s:%d: %s (code %d)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err), err); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Функция округления размера до кратного BLOCK_SIZE
int toMultiple(int a, int b) {
    int mod = a % b;  // Вычисление остатка от деления
    return mod == 0 ? a : a + (b - mod);  // Если остаток равен нулю, возвращаем a; иначе округляем до следующего кратного
}

// Ядро: умножение матриц с использованием разделяемой памяти
__global__ void matrixMult(const BASE_TYPE* A, const BASE_TYPE* B, BASE_TYPE* C,
                           int Acols, int Bcols) {
    // Индексы начала обрабатываемых подматриц
    int aBegin = Acols * BLOCK_SIZE * blockIdx.y;  // Начало подматрицы A для текущего блока
    int bBegin = BLOCK_SIZE * blockIdx.x;          // Начало подматрицы B для текущего блока

    // Шаги для итерации по подматрицам
    int aStep = BLOCK_SIZE;                          // Шаг по строкам в A
    int bStep = BLOCK_SIZE * Bcols;                 // Шаг по столбцам в B

    // Разделяемая память для подматриц A и B
    __shared__ BASE_TYPE As[BLOCK_SIZE][BLOCK_SIZE];  // Разделяемая память для подматрицы A
    __shared__ BASE_TYPE Bs[BLOCK_SIZE][BLOCK_SIZE];  // Разделяемая память для подматрицы B

    BASE_TYPE sum = 0.0;  // Переменная для накопления суммы произведений

    // Цикл по тайлам (подматрицам)
    for (int a = aBegin, b = bBegin; a < aBegin + Acols; a += aStep, b += bStep) {
        // Загрузка в разделяемую память
        As[threadIdx.y][threadIdx.x] = A[a + threadIdx.y * Acols + threadIdx.x];  // Загрузка элемента из A
        Bs[threadIdx.y][threadIdx.x] = B[b + threadIdx.y * Bcols + threadIdx.x];  // Загрузка элемента из B

        // Синхронизация потоков в блоке
        __syncthreads();  // Ожидание, пока все потоки загрузят свои данные в разделяемую память

        // Вычисление частичного произведения для текущего тайла
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];  // Умножение и накопление результата
        }

        // Синхронизация перед загрузкой следующего тайла
        __syncthreads();  // Ожидание завершения вычислений текущего тайла перед загрузкой следующего
    }

    // Индекс результирующего элемента в глобальной памяти
    int row = BLOCK_SIZE * blockIdx.y + threadIdx.y;  // Вычисление глобального индекса строки
    int col = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // Вычисление глобального индекса столбца
    int idx = row * Bcols + col;                       // Вычисление индекса в результирующей матрице C

    // Запись результата
    C[idx] = sum;  // Запись накопленной суммы в результирующую матрицу C
}

int main() {
    printf("=== Matrix Multiplication (CUDA, Shared Memory) ===\n");

    // 1. Исходные размеры матриц
    int Arows = 100, Acols = 200;  // Определение размеров матрицы A
    int Brows = Acols, Bcols = 150; // Определение размеров матрицы B (число строк B равно числу столбцов A)

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
    BASE_TYPE *h_A = (BASE_TYPE*)malloc(Asize);  // Выделение памяти для матрицы A на CPU
    BASE_TYPE *h_B = (BASE_TYPE*)malloc(Bsize);  // Выделение памяти для матрицы B на CPU
    BASE_TYPE *h_C = (BASE_TYPE*)malloc(Csize);  // Выделение памяти для матрицы C на CPU
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
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);// Определение числа потоков в блоке (BLOCK_SIZE x BLOCK_SIZE)
    dim3 blocksPerGrid(Bcols / BLOCK_SIZE, Arows / BLOCK_SIZE);// Определение числа блоков в сетке (по количеству блоков для столбцов и строк)

    // 9. Замер времени выполнения
    cudaEvent_t start, stop; // Создание переменных для событий, которые будут использоваться для замера времени
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));// Запись события начала выполнения ядра
    matrixMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, Acols, Bcols); // Запуск ядра умножения матриц на GPU
    CUDA_CHECK(cudaGetLastError());// Проверка на наличие ошибок после запуска ядра
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());// Синхронизация устройства, ожидание завершения всех операций на GPU

    float kernelTime;// Переменная для хранения времени выполнения ядра
    CUDA_CHECK(cudaEventElapsedTime(&kernelTime, start, stop));
    printf("Kernel execution time: %.2f ms\n", kernelTime);

    // 10. Копирование результата на CPU
    CUDA_CHECK(cudaMemcpy(h_C, d_C, Csize, cudaMemcpyDeviceToHost));// Копирование результата из GPU (d_C) обратно на CPU (h_C)

    // 11. Проверка корректности результата (для первых 10×10 элементов)
    printf("Starting verification (first 10×10 elements)...\n");
    bool passed = true;//Флаг для отслеживания успешности проверки
    for (int i = 0; i < 10 && i < Arows && passed; ++i) {// Цикл по строкам для проверки (до 10 или до Arows)
        for (int j = 0; j < 10 && j < Bcols && passed; ++j) {
            BASE_TYPE expected = 0.0;// Переменная для хранения ожидаемого значения
            for (int k = 0; k < Acols; ++k) {// Цикл для вычисления ожидаемого значения
                expected += h_A[i * Acols + k] * h_B[k * Bcols + j];// Вычисление ожидаемого значения умножением элементов матриц A и B
            }
            if (fabs(h_C[i * Bcols + j] - expected) > 1e-3) {// Проверка на соответствие полученного и ожидаемого значений с заданной точностью
                fprintf(stderr, "Verification failed at [%d,%d]: "
                        "got %.6f, expected %.6f\n",
                        i, j, h_C[i * Bcols + j], expected);
                passed = false;
            }
        }
    }
    printf("Verification: %s\n", passed ? "PASSED" : "FAILED");// Вывод результата проверки

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