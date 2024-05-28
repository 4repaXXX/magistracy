from mpi4py import MPI
import numpy as np
import time
import matplotlib.pyplot as plt

def local_fft(signal):
    N = len(signal)
    output = np.zeros_like(signal, dtype=np.complex128)
    for i in range(N): 
        sum_val = 0.0j
        for k in range(N):
            angle = -2j * np.pi * k * i / N
            sum_val += signal[k] * np.exp(angle)
        output[i] = sum_val
    return output

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    fs = 4096
    t = np.linspace(0, 1, fs, endpoint=False)
    
    if rank == 0:
        full_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    else:
        full_signal = None
        
    counts = [fs // size + (1 if i < fs % size else 0) for i in range(size)]
    counts = [2 * count for count in counts]
    displacements = [sum(counts[:i]) for i in range(size)]
    
    local_signal = np.empty(counts[rank]//2, dtype='complex128')  # Ensure correct size allocation

    comm.Scatterv([full_signal, counts, displacements, MPI.DOUBLE], local_signal, root=0)

    processed_signal = local_fft(local_signal)

    if rank == 0:
        processed_full_signal = np.empty(fs, dtype='complex128')
    else:
        processed_full_signal = None

    # Fix counts and displacements for gathering complex numbers
    comm.Gatherv(processed_signal, [processed_full_signal, counts, displacements, MPI.COMPLEX], root=0)

    if rank == 0:
        print("Processed full signal collected at root:", processed_full_signal)

if __name__ == '__main__':

# Example data: replace with your actual timings
    process_counts = [1, 2, 4, 8, 16]  # Number of processes used
    execution_times = [8.23, 9.67, 5.35, 3.20, 2.15]  # Corresponding execution times
        
    plt.figure(figsize=(10, 5))
    plt.plot(process_counts, execution_times, marker='o')
    plt.xlabel('Number of MPI Processes')
    plt.ylabel('Execution Time (seconds)')
    plt.title('FFT Execution Time vs. Number of MPI Processes')
    plt.grid(True)
    plt.show()

