#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

// #include <immintrin.h>

#define ALIGN 64

float **matrix_1 = NULL;
float **matrix_2 = NULL;
float **matrix_res = NULL;

typedef struct start_end
{
    unsigned int start;
    unsigned int end;
    unsigned matrix_size;
} start_end_t;

int alloc_matrix(unsigned int matrix_size)
{
    matrix_1 = (float **)aligned_alloc(ALIGN, matrix_size * sizeof(float *));
    matrix_2 = (float **)aligned_alloc(ALIGN, matrix_size * sizeof(float *));
    matrix_res = (float **)aligned_alloc(ALIGN, matrix_size * sizeof(float *));

    if (matrix_1 == NULL || matrix_2 == NULL || matrix_res == NULL)
        return 1;

    for (int i = 0; i < matrix_size; i++)
    {
        matrix_1[i] = (float *)aligned_alloc(ALIGN, matrix_size * sizeof(float));
        matrix_2[i] = (float *)aligned_alloc(ALIGN, matrix_size * sizeof(float));
        matrix_res[i] = (float *)aligned_alloc(ALIGN, matrix_size * sizeof(float));
        if (matrix_1[i] == NULL || matrix_2[i] == NULL || matrix_res[i] == NULL)
            return 1;
    }

    for (int i = 0; i < matrix_size; i++)
    {
        for (int j = 0; j < matrix_size; j++)
        {
            matrix_1[i][j] = i * 0.34 + 1;
            matrix_2[i][j] = j * 0.28 + 1;
        }
    }

    return 0;
}

int mult_matrix(unsigned int matrix_size)
{
    for (int i = 0; i < matrix_size; i++)
    {
        for (int j = 0; j < matrix_size; j++)
        {
            for (int k = 0; k < matrix_size; k++)
            {
                matrix_res[i][j] += matrix_1[i][k] * matrix_2[k][j];
            }
        }
    }

    return 0;
}

void *mult_matrix_pthread(void *args)
{
    for (int i = 0; i < ((start_end_t *)args)->matrix_size; i++)
    {
        for (int j = 0; j < ((start_end_t *)args)->matrix_size; j++)
        {
            for (int k = ((start_end_t *)args)->start; k < ((start_end_t *)args)->end; k++)
            {
                matrix_res[i][j] += matrix_1[i][k] * matrix_2[k][j];
            }
        }
    }

    return 0;
}

int free_matrix(unsigned int matrix_size)
{
    for (int i = 0; i < matrix_size; i++)
    {
        free(matrix_res[i]);
        free(matrix_1[i]);
        free(matrix_2[i]);
    }
    free(matrix_res);
    free(matrix_1);
    free(matrix_2);

    return 0;
}

int main(int argc, char **argv)
{
    unsigned int matrix_size = 32;
    unsigned int num_threads = 8;

    if (argc == 2)
    {
        if (atoi(argv[1]) < 1)
        {
            printf("error: matrix_size < 1\n");
            return -1;
        }
        matrix_size = atoi(argv[1]);
    }
    else if (argc == 3)
    {
        if (atoi(argv[1]) < 1)
        {
            printf("error: matrix_size < 1\n");
            return -1;
        }
        else if (atoi(argv[2]) < 1)
        {
            printf("error: num_threads < 1\n");
            return -1;
        }
        // need add check on devision matrix_size on num_threads
        matrix_size = atoi(argv[1]);
        num_threads = atoi(argv[2]);
    }
    printf("matrix_size: %d\n"
           "num_threads: %d\n",
           matrix_size, num_threads);

    if (num_threads > matrix_size)
    {
        printf("error: num_threads > matrix_size\n");
        return -1;
    }

    if (alloc_matrix(matrix_size))
    {
        printf("error: allocation issue\n");
    }

    unsigned int chunk_size = matrix_size / num_threads;
    unsigned int last_chunk_size = matrix_size - chunk_size * num_threads;
    pthread_t *thread_insts;
    thread_insts = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    start_end_t start_end_inst = ;//need to allocate memory
    // Need to summarize results from threads
    for (int i = 0; i < num_threads; i++)
    {
        
        start_end_inst.start = chunk_size * i;
        start_end_inst.end = (i == num_threads - 1) ? chunk_size * (i + 1) - 1 + last_chunk_size : chunk_size * (i + 1) - 1;
        start_end_inst.matrix_size = matrix_size;
        pthread_create(&thread_insts[i], NULL, mult_matrix_pthread, (void *)&start_end_inst);
    }

    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(thread_insts[i], NULL);
    }
    // mult_matrix(matrix_size);

    free_matrix(matrix_size);
    return 0;
}