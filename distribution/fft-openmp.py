import numpy as np
from numba import njit, prange
import time
from numba.np.ufunc.parallel import set_num_threads
import matplotlib.pyplot as plt


def fft_np(signal):
    n = len(signal)
    if n <= 1:
        return signal
    even = fft(signal[0::2])
    odd = fft(signal[1::2])
    T = [np.exp(-2j * np.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + T[k] for k in range(n // 2)] + [even[k] - T[k] for k in range(n // 2)]

def fft(signal):
    N = len(signal)
    output = np.zeros_like(signal, dtype=np.complex128)
    for i in range(N): 
        sum_val = 0.0j
        for k in range(N):
            angle = -2j * np.pi * k * i / N
            sum_val += signal[k] * np.exp(angle)
        output[i] = sum_val
    return output


@njit(parallel=True)
def fft_iterative(signal):
    N = len(signal)
    output = np.zeros_like(signal, dtype=np.complex128)
    for i in prange(N): 
        sum_val = 0.0j
        for k in range(N):
            angle = -2j * np.pi * k * i / N
            sum_val += signal[k] * np.exp(angle)
        output[i] = sum_val
    return output

def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args)
    return time.time() - start_time

def print_fft(fft_result):
    freqs = np.fft.fftfreq(len(fft_result), d=t[1] - t[0])

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, np.abs(fft_result), 'b')
    plt.title('Амплитудный спектр сигнала')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.xlim(0, 15) 
    plt.grid()
    plt.show()
    
def print_mesure_fft(num_threads_list, times, speedups):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(num_threads_list, times, marker='o')
    plt.xlabel('Number of Threads')
    plt.ylabel('Execution Time (seconds)')
    plt.title('FFT Execution Time vs. Number of Threads')
    plt.grid(True)

    
    plt.subplot(1, 2, 2)
    plt.plot(num_threads_list, speedups, marker='o', color='red')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.title('Speedup vs. Number of Threads')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def etalon_fft(signal):
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(t.size, d=t[1] - t[0])

    # Амплитудный спектр
    amplitude_spectrum = np.abs(fft_result)

    # Фазовый спектр
    phase_spectrum = np.angle(fft_result)

    # Построение графиков
    plt.figure(figsize=(12, 6))

    # Амплитудный спектр
    plt.subplot(2, 1, 1)
    plt.plot(fft_freq, amplitude_spectrum)
    plt.title('Амплитудный спектр')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.xlim(0, 15) 
    plt.grid(True)

    # Фазовый спектр
    plt.subplot(2, 1, 2)
    plt.plot(fft_freq, phase_spectrum)
    plt.title('Фазовый спектр')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Фаза (радианы)')
    plt.grid(True)
    plt.xlim(0, 100) 

    plt.tight_layout()
    plt.show()


# Signal
fs = 32768
t = np.linspace(0, 10, fs, endpoint=False)  # 256 точек в сигнале
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)  # Синусоидальный сигнал с частотами 5 Гц и 10 Гц


start = time.time()
fft_result = fft(signal)
end = time.time()
print('FFT:', end - start, 'Sec')
print_fft(fft_result)

# times = []
# for num_threads in range(1, 9):
#     set_num_threads(num_threads)
#     elapsed_time = measure_time(fft_iterative, signal)
#     times.append(elapsed_time)
#     print(f"Threads: {num_threads}, Time: {elapsed_time:.4f}s")

# base_time = times[0]
# speedups = [base_time / t for t in times]
# for num_threads, speedup in zip(range(1, 9), speedups):
#     print(f"Threads: {num_threads}, Speedup: {speedup:.2f}")
    
# print_mesure_fft(range(1, 9), times, speedups)

