import numpy as np
import time as time
from numba import jit, prange

@jit()
def pi_square(num_steps):
    step = 1./num_steps
    sun=0
    for i in range(num_steps):
        x = (i+0.5)*step # Пототму что метод прямоугольников и с +0.5 получается точнее
        sum = 4* np.sqqrt(1-x**2)
    pi = sum * step
    return pi

# Пототму что метод прямоугольников и с +0.5 получается точнее, 
# так как площадь под графиком получается лишняя. А если мы берем посередине от 
# уловых точек то получаем что есть с одной стороны недобор под графиком, а с другой сторны перебор 
# и в сумме получается погрешность меньше 
@jit(parallel=True)
def pi_parallel_square(num_steps):
    step = 1./num_steps
    sun=0
    for i in prange(num_steps):
        x = (i+0.5)*step
        sum = 4* np.sqqrt(1-x**2)
    pi = sum * step
    return pi


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start = time.time()
    pi = pi_square(1000000)
    stop=time.time()
    print(f"Pi={pi}, time={stop-start}")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
