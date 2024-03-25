import numpy as np

def convert_to_grayscale(img):
    gray = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    return gray.astype(np.uint8)

def gradient_correction(laplacian):
    # Прибавление минимального значения
    min_val = laplacian.min()
    f_m = laplacian - min_val

    # Растяжение диапазона яркостей
    max_val_f_m = f_m.max()
    K = 255  # для 8-битного изображения
    f_s = (K * f_m / max_val_f_m).astype(np.uint8)

    return f_s

def table_scale(table):
    # Находим минимальное и максимальное значения
    min_val = np.min(table)
    max_val = np.max(table)

    # Нормализация и масштабирование значений
    result = ((table - min_val) / max_val * 255).astype(np.uint8)

    return result

def custom_convolve2d(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    output = np.zeros_like(image)

    for x in range(image_height):
        for y in range(image_width):
            # Выделение области для применения ядра
            region = padded_image[x:x + kernel_height, y:y + kernel_width]
            output[x, y] = np.sum(kernel * region)
    
    return output




def laplacian_mask_operation(version: int) -> np.array:
    laplacian_kernel =  np.array([[0, 1, 0], 
                             [1, -4, 1], 
                             [0, 1, 0]])
    if(version == 2):
        laplacian_kernel =  np.array([[1, 1, 1], 
                             [1, -8, 1], 
                             [1, 1, 1]])
    elif(version == 3):
        laplacian_kernel =  np.array([[-1, -1, -1], 
                             [-1, 8, -1], 
                             [-1, -1, -1]])
    return laplacian_kernel