import numpy as np
from numba import njit, prange
import time

@njit(parallel=True)
def fft_parallel(signal):
    n = len(signal)
    signal = signal.astype(np.complex128)  # Преобразование входного массива в комплексные числа
    if n <= 1:
        return signal
    even = fft_parallel(signal[0::2])
    odd = fft_parallel(signal[1::2])
    T = np.empty(n // 2, dtype=np.complex128)
    for k in prange(n // 2):  # Использование prange для параллельного выполнения цикла
        T[k] = np.exp(-2j * np.pi * k / n) * odd[k]
    result = np.empty(n, dtype=np.complex128)
    for k in range(n // 2):  # Ещё один параллельный цикл
        result[k] = even[k] + T[k]
        result[k + n // 2] = even[k] - T[k]
    return result

def fft(signal):
    n = len(signal)
    if n <= 1:
        return signal
    even = fft(signal[0::2])
    odd = fft(signal[1::2])
    T = [np.exp(-2j * np.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + T[k] for k in range(n // 2)] + [even[k] - T[k] for k in range(n // 2)]

def fft_iterative(signal):
    n = len(signal)
    signal = np.asarray(signal, dtype=np.complex128)
    
    # Bit-reversal permutation
    indices = np.arange(n)
    rev = 0
    for i in range(1, n):
        bit = n >> 1
        while rev >= bit:
            rev -= bit
            bit >>= 1
        rev += bit
        if i < rev:
            signal[i], signal[rev] = signal[rev], signal[i]
    
    # Cooley-Tukey FFT
    length = 2
    while length <= n:
        phase_shift_step = np.exp(-2j * np.pi / length)
        for start in range(0, n, length):
            phase_shift = 1
            for k in range(length // 2):
                pos = start + k
                offset = pos + length // 2
                temp = phase_shift * signal[offset]
                signal[offset] = signal[pos] - temp
                signal[pos] += temp
                phase_shift *= phase_shift_step
        length *= 2
        
    return signal

# Пример сигнала
fs = 8192
t = np.linspace(0, 10, fs, endpoint=False)  # 256 точек в сигнале
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)  # Синусоидальный сигнал с частотами 5 Гц и 10 Гц

# Вызов FFT

start = time.time()
fft_result = fft_parallel(signal)
end = time.time()
print('FFT numba:', end - start, 'Sec')

start = time.time()
fft_result = fft(signal)
end = time.time()
print('FFT:', end - start, 'Sec')

start = time.time()
fft_result = fft_iterative(signal)
end = time.time()
print('FFT iterative:', end - start, 'Sec')

# Рассчитаем частоты для оси x
freqs = np.fft.fftfreq(len(fft_result), d=1/fs)

# Визуализация
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(freqs, np.abs(fft_result), 'b')
plt.title('Амплитудный спектр сигнала')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.grid()
plt.show()
