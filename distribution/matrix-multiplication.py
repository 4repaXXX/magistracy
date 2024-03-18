import numpy as np
from numba import njit, prange
import time

@njit(parallel=True)
def matrix_multiply_parallel(A, B):
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    if cols_A != rows_B:
        raise ValueError("Number of columns in first should be equal to rows in second")

    # Создаем матрицу результата
    result = np.zeros((rows_A, cols_B), dtype=np.float64)

    # Перемножаем матрицы в параллельном цикле
    for i in prange(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i, j] += A[i, k] * B[k, j]

    return result



def matrix_multiply(A, B):
    if len(A[0]) != len(B):
        raise ValueError("Number of columns in first should be equal to rows in second")


    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    return result


if __name__ == '__main__':
    
    N = 500
    A = 1 * np.ones([N,N])
    B = 2 * np.ones([N,N])
    start = time.time()
    r = matrix_multiply_parallel(A, B)
    stop=time.time()
    print(f"Pi={r}, time={stop-start}")