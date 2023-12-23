import numpy as np
from numba import jit
import time


# @jit(nopython=True)
def argmax_numba(arr):
    return np.argmax(arr)


arr = np.random.rand(100000)
a = np.random.rand(1600)
b = np.random.rand(120)
T1 = time.perf_counter()

print(arr.shape)
for k in range(1000):
    searched_space = np.kron(a, np.ones((1, b.size))) + np.kron(np.ones((1, a.size)), b)
    x = argmax_numba(arr)

T2 = time.perf_counter()
print(T2 - T1)
print("程序运行时间:%s秒" % (T2 - T1))
