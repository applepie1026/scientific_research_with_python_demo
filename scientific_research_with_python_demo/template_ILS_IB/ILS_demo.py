import scientific_research_with_python_demo.utils as af
import numpy as np
import scientific_research_with_python_demo.ps_vce as vc
import scientific_research_with_python_demo.LAMBDA as ils

WAVELENGTH = 0.056  # [unit:m]
Nifg = 30
v_orig = 0.01  # [mm/year] 减少v，也可以改善估计结果，相当于减少了重访周期
h_orig = 30  # [m]，整数 30 循环迭代搜索结果有问题
pseudo_obs = np.array([[28, 0.04]]).T
noise_level = 20
param_name = ["height", "velocity"]
normal_baseline = np.random.randn(1, Nifg) * 333
# print(normal_baseline)
time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg)  # 减小重访周期 dt 能明显改善结果
m2ph = 4 * np.pi / WAVELENGTH
std_param = np.array([40, 0.06])
# calculate the input parameters of phase
v2ph = af.v_coef(time_baseline).T
h2ph = af.h_coef(normal_baseline).T
sig0 = np.sqrt(2 * 0.25**2)
# 设置搜索边界
dh = 40
vel = 10
# 获得design matrix 基于线性模型 y=2pi*a+v2ph*vel+h2ph*dh
A = np.hstack((h2ph, v2ph)) * m2ph
# 先随便猜一个与观测相位有关的 VC 参数
VC_guessed, Q_y, Q_y_i = vc.guess_VC(sig0, Nifg)
print(VC_guessed)
# 获得参数 vel 和 h 的协方差矩阵
Q_b0 = vc.pseudoVCM(dh, vel)
# 获取一个猜测的模糊度协方差矩阵
Q_a_guess = vc.ambiguity_VC(Q_y, A, Q_b0)

# 模拟解缠相位，以及缠绕后的观测相位
phase_obs, srn, phase_unwrap, h_phase, v_phase = af.sim_arc_phase(v_orig, h_orig, v2ph, h2ph, noise_level)
# print(phase_obs)
# 获得一个粗略的浮点解，默认 pseudo v , h 均为 0
a_hat = phase_obs / (2 * np.pi)
a_hat = a_hat.reshape((Nifg,))
print(a_hat)
phase_obs = phase_obs.reshape(Nifg)
print(phase_obs)

# 第一次ILS估计。基于猜测的模糊度协方差矩阵，以及粗略的浮点解
afixed = ils.main(a_hat, Q_a_guess, 1)[0]
print(afixed)

# 选择最可靠的候选模糊度解
afixed = afixed[:, 0]
print(afixed)

# 重新计算解缠相位
phase_unwrap_new = -2 * np.pi * afixed + phase_obs
print(phase_unwrap_new.shape)

# 根基解缠相位、猜测的 VC 参数，参数的设计矩阵，重新估计 VC 参数
VC = vc.VCE(VC_guessed, A, phase_unwrap_new)

print(VC)

# 根据新的VC参数，重新计算观测相位协方差矩阵，模糊度协方差矩阵
Qy_new = np.diag(VC)
Qy_new_i = np.diag(1 / np.diag(Qy_new))
Q_a_guess_new = vc.ambiguity_VC(Q_y, A, Q_b0)

# 第二次ILS估计。基于新的模糊度协方差矩阵，以及第一次估计的模糊度解
afixed_new = ils.main(a_hat, Q_a_guess_new, 1)[0]

print(afixed_new[:, 0].reshape(Nifg))

# 计算模拟的模糊度，方便对比估计的模糊度
a_true = (phase_obs - phase_unwrap.reshape(Nifg)) / (2 * np.pi)
print(np.round(a_true.reshape(Nifg)))
# a_true = (phase_obs - phase_true) / (2 * np.pi)
# print(a_true)
# print(phase_true)
# print(phase_obs)


# A_desin, y = af.design_mat(h2ph, v2ph, phase_obs, pseudo_obs)
# print(A_desin)
# # a_hat = y / (2 * np.pi)
# a_hat = af.compute_ahat(A_desin, y)
# print(a_hat)
# # print(y[0:5, :] / (2 * np.pi))
# # print(y)
# Q_ahat = af.cov_ahat(A_desin, Q_y, Nifg)
# print(Q_ahat)
# np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/a_hat1.csv", a_hat, delimiter=",")
# np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Q_ahat1.csv", Q_ahat, delimiter=",")
# np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/a_true1.csv", a_true, delimiter=",")
