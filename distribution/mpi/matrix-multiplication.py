from mpi4py import MPI
import numpy as np

def matrix_multiply(A, B, C, rows_per_proc):
    for i in range(rows_per_proc):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i][j] += A[i][k] * B[k][j]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

M, N, P = 2,2,2

if rank == 0:
    A = np.random.randint(0, 5, size=(M,N))
    B = np.random.randint(0, 5, size=(N, P))
    C = np.zeros((M, P))


comm.bcast(B, root=0)

rows_per_proc = M // size
if rank == size - 1:
    rows_per_proc += M % size
local_A = np.zeros((rows_per_proc, N))
comm.Scatterv([A, (rows_per_proc,) * size, None, MPI.INT], local_A, root=0)

local_C = np.zeros((rows_per_proc, P))

matrix_multiply(local_A, B, local_C, rows_per_proc)


rows_per_proc_list = [M // size for _ in range(size)]
rows_per_proc_list[-1] += M % size

displs = [sum(rows_per_proc_list[:i]) * P for i in range(size)]
recvcounts = [rows_per_proc_list[i] * P for i in range(size)]

comm.Gatherv(local_C, [C, recvcounts, displs, MPI.DOUBLE], root=0)

if rank == 0:
    print("Matrix A:")
    print(A)
    print("Matrix B:")
    print(B)
    print("Matrix C (A x B):")
    print(C)
