import numpy as np
import matplotlib.pyplot as plt

phase_v = np.arange(1, 31, 1) * (36 / 365) * (4 * np.pi / 0.056) * 0.005
print(phase_v[29] / np.pi * 180)
# x_orig = (1 + 1j) / np.sqrt(2)
x_orig = np.exp(1j * phase_v[29])
phase_orig = np.angle(x_orig) / np.pi * 180
print(phase_orig)
SNR = 10
nifg = 1000
phase_signal_noise = np.zeros(nifg)
phase_noise = np.zeros(nifg)
noise_std = np.power(10, SNR / (-20)) / np.sqrt(2)
# 噪声分布
for i in range(nifg):
    x_noise = np.random.normal(0, noise_std, 1) + 1j * np.random.normal(0, noise_std, 1)
    x_signal_noise = x_orig + x_noise
    phase_signal_noise[i] = np.angle(x_signal_noise) / np.pi * 180
    # print(phase_signal_noise)
    phase_noise[i] = phase_signal_noise[i] - phase_orig
# print(phase_noise)
plt.figure()
plt.hist(phase_noise, bins=10)
plt.show()
