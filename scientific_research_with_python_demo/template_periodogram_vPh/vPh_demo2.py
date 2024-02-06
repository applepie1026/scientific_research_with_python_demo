import numpy as np
import json
import time
import template_periodogram.DTFT_LAB_vPh as DFT_vPh

v = 0.005
h = 10
dBn = 5
sigma_bn = 5
dT = 35
N = 30
flatten_num = 100
flatten_range = 60
W_v = np.arange(-1600, 1600 + 1, 1) * 0.0001
W_h = np.arange(-600, 600 + 1, 1) * 0.1
X = np.arange(1, N + 1, 1)
# T1 = time.perf_counter()
# dtft_af_all_array(X, X).xjw(W_v, 30)
# T2 = time.perf_counter()
# print("程序运行时间:%s秒" % (T2 - T1))
T1 = time.perf_counter()
v_data = np.zeros(1000)
h_data = np.zeros(1000)
success_num = 0
check_times = 100
for i in range(check_times):
    xjw, xjw_h, v_done, h_done = DFT_vPh.dtft_vflatten_af(v, h, dBn, sigma_bn, dT, N, flatten_num, flatten_range, W_v, W_h)
    v_data[i] = v_done
    h_data[i] = h_done
    if abs(v_done - v) <= 0.0005 and abs(h_done - h) <= 0.5:
        success_num += 1
    print(f"v_est={v_done},h_est={h_done}")

T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
print(f"success_rate={success_num/check_times}")
