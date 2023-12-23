import scientific_research_with_python_demo.ILS_estimator as ils
import numpy as np
import time

# ILS实验demo
# sig = np.sqrt(2 * (np.pi * 40 / 180) ** 2)
# print(sig)
# Nifg = 30
# noise_level = 10
# a_est, a_true, success_rate, x1, x2 = ils.main(v=0.001, h=30, Nifg=30, noise_level=10, dT=36, Bn=333, h_bound=30, v_bound=20, sig0=sig)
# print(a_est)
# print(np.round(a_true))
# print(success_rate)
# # 保留4位数
# print("%.4f" % x1[0], "%.4f" % x1[1])
# print("%.4f" % x2[0], "%.4f" % x2[1])

# 计算成功率
sig = np.sqrt(2 * (np.pi * 40 / 180) ** 2)
# v = 0.016
v = np.arange(1, 171, 1) * 0.001
h = 30
Nifg = 30
noise_level = 10
dT = 36
Bn = 333
h_bound = 30
v_bound = 10
T1 = time.time()
success_rate = np.zeros(len(v))
for i in range(len(v)):
    v = v[i]
    success_rate[i] = ils.check_success_rate(v, h, Nifg, noise_level, dT, Bn, h_bound, v_bound, sig, check_times=1000)
np.savetxt("scientific_research_with_python_demo/data_save/ils_demo1.txt", success_rate)
T2 = time.time()
print("程序运行时间:%s秒" % (T2 - T1))
print(success_rate)
param = np.array([[1], [2]])
print(param)
print(param.shape)
