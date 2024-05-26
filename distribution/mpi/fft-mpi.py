from mpi4py import MPI
import numpy as np

def local_fft(signal):
    n = len(signal)
    if n <= 1:
        return signal
    even = local_fft(signal[0::2])
    odd = local_fft(signal[1::2])
    T = [np.exp(-2j * np.pi * k / n) * odd[k] for k in range(n // 2)]
    return np.array([even[k] + T[k] for k in range(n // 2)] + [even[k] - T[k] for k in range(n // 2)], dtype=np.complex_)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    fs = 16  # Ensure this is a power of 2
    t = np.linspace(0, 1, fs, endpoint=False)
    
    if rank == 0:
        full_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    else:
        full_signal = None
        
    counts = [fs // size + (1 if i < fs % size else 0) for i in range(size)]
    counts = [2 * count for count in counts]  # Adjusting counts for complex numbers (2x size)
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
    main()
