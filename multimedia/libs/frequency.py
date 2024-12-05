import numpy as np
import cv2

def gausian(D, D0):
    return 1 - np.exp(-(D**2) / (2*(D0**2)))

def ideal(D, D0):
    if (D <= D0):
        return 0
    else:
        return 1

def spectrum(dft_shift):

    magnitude_spectrum = 20 * np.log(np.abs(dft_shift))
    return magnitude_spectrum

def compute_spectrum(dft_shift):
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
    return magnitude_spectrum


def ideal_high_pass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 1
    return 1 - mask

def gaussian_high_pass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    x, y = np.meshgrid(x, y)

    # Расчет радиуса от центра
    radius = np.sqrt((x ** 2) + (y ** 2))
    mask = 1 - np.exp(-(radius**2) / (2 * (cutoff**2)))
    
    # rows, cols = shape
    # crow, ccol = rows // 2, cols // 2
    # mask = np.zeros((rows, cols))
    # for x in range(rows):
    #     for y in range(cols):
    #         radius = np.sqrt((x - rows/2)**2 + (y - cols/2)**2)
    #         mask[x,y] = 1 - np.exp(-(radius**2) / (2 * (cutoff**2)))
    
    print(f"Mask min: {mask.min()}, max: {mask.max()}, mean: {mask.mean()}")

    return mask

def butterworth_high_pass_filter(shape, cutoff, order):
    # Возвращает ядро Баттерворта
    
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    x, y = np.meshgrid(x, y)
    radius = np.sqrt((x ** 2) + (y ** 2))
    
    mask = 1 / (1 + (cutoff / radius) ** (2 * order))
    # rows, cols = shape
    # mask = np.zeros((rows, cols))
    # for x in range(rows):
    #     for y in range(cols):
    #         radius = np.sqrt((x - rows / 2) ** 2 + (y - cols / 2) ** 2)
    #         mask[x, y] = 1 / (1 + (cutoff / radius) ** (2 * order))

    return mask

def laplasian_kernel(shape):
    # Возвращает ядро Лапласиана для заданного размера
    # rows, cols = shape
    # ky, kx = np.mgrid[0: rows, 0 : cols]
    # kernel = -4 * np.pi**2 * (kx**2 + ky**2)
    
    rows, cols = shape
    ky, kx = np.mgrid[-rows//2 + 1 : rows//2 + 1, -cols//2 + 1 : cols//2 + 1]
    kernel = -4 * np.pi**2 * (kx**2 + ky**2)
    return kernel