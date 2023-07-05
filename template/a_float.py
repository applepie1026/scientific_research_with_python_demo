import scientific_research_with_python_demo.utils as af
import numpy as np

WAVELENGTH = 0.0056  # [unit:m]
Nifg = 3
v_orig = 0.05  # [mm/year] 减少v，也可以改善估计结果，相当于减少了重访周期
h_orig = 30  # [m]，整数 30 循环迭代搜索结果有问题
pseudo_obs = np.array([[30, 0.05]]).T
noise_level = 100
param_name = ["height", "velocity"]
normal_baseline = np.random.normal(size=(1, Nifg)) * 333
# print(normal_baseline)
time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg)  # 减小重访周期 dt 能明显改善结果
m2ph = 4 * np.pi / WAVELENGTH
# calculate the input parameters of phase
v2ph = af.v_coef(time_baseline).T
h2ph = af.h_coef(normal_baseline).T
phase_obs, srn, phase_true = af.sim_arc_phase(v_orig, h_orig, v2ph, h2ph, noise_level)
print(phase_true)
print(phase_obs)
A_desin, y = af.design_mat(h2ph, v2ph, phase_obs, pseudo_obs)
print(A_desin)
print(y)
# 基于 A_desin 和 y 最小二乘法求解参数
x = np.linalg.inv(A_desin.T @ A_desin) @ A_desin.T @ y
print(x)
