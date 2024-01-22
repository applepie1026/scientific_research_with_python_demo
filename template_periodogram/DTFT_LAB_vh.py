import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

H = 780000  # satellite vertical height[m]
Incidence_angle = 23 * np.pi / 180  # the local incidence angle
R = H / np.cos(Incidence_angle)
Lambda = 0.056
# x = np.random.normal(0, 5, 30) + np.arange(10, 310, 10)
# y = np.linspace(10, 333, 30)
# print(y)
# print(x)
# # plt.hist(x, bins=100)
# plt.plot([1, 30], [10, 300])
# plt.plot(np.arange(1, 31), x)
# plt.show()


def wrap(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def porabola_bn(N, sigma_bn):
    bn_max = 500
    # bn = -1.5 * np.arange(-N / 2, N / 2, 1) ** 2 + bn_max + np.random.normal(0, sigma_bn, N)
    bn = -50 * np.linspace(-3.16, 3.16, N) ** 2 + bn_max + np.random.normal(0, sigma_bn, N)
    return bn


def add_gaussian_noise(Nifg, noise_level_set):
    """construct gaussian noise based on signal and SNR

    Args:
        signal (_array_): arc phase of 'Nifgs' interferograms without phase ambiguity
        SNR (_type_): the signal phase to noise ratio of the arc based on interferograms

    Returns:
        _array_: gaussian noise base on signal size and SNR
    """
    """
    :param signal: 原始信号
    :param SNR: 添加噪声的信噪比
    :return: 生成的噪声
    """
    noise_std = np.zeros((1, Nifg + 1))
    noise_level = np.zeros((Nifg + 1, 1))
    noise_level[0] = np.pi * noise_level_set / 180
    noise_std[0][0] = np.random.randn(1) * noise_level[0]

    for v in range(Nifg):
        noise_level[v + 1] = (np.pi * noise_level_set / 180) + np.random.randn(1) * (np.pi * 5 / 180)
        noise_std[0][v + 1] = np.random.randn(1) * noise_level[v + 1]
    noise_phase = np.zeros((Nifg, 1))
    for i in range(Nifg):
        noise_phase[i] = noise_std[0][i] + noise_std[0][i + 1]

    return noise_phase


def add_gaussian_noise1(Nifg, noise_level_set):
    noise_phase = np.random.normal(0, np.pi * noise_level_set / 180, (Nifg, 1))
    return noise_phase


class dtft_vh:
    # xvalues:输入序列
    def __init__(self, xvalues, v2ph, h2ph, fre_v=[], fre_h=[]):
        self.yvalues = []
        self.xvalues = xvalues
        self.v2ph = v2ph
        self.h2ph = h2ph

    # fre:频率坐标
    def xjw(self, fre_v=[], fre_h=[], N=[]):
        # （式1-1）实现yvalues为X（jw）频谱值

        # for f_h, f_v in zip(fre_h.flatten(), fre_v.flatten()):
        #     p = 0
        #     for x, v_b, h_b in zip(self.xvalues, self.v2ph, self.h2ph):
        #         p = math.e ** (-1j * (f_v * v_b + f_h * h_b)) * x + p
        #     self.yvalues.append(p)
        for f_h in fre_h:
            for f_v in fre_v:
                p = 0
                for x, v_b, h_b in zip(self.xvalues, self.v2ph, self.h2ph):
                    p = math.e ** (-1j * (f_v * v_b + f_h * h_b)) * x + p
                self.yvalues.append(p)


def dtft_vh_af(v, h, dBn, sigma_bn, dT, N, W_v, W_h):
    v2ph = dT * np.arange(1, N + 1, 1) * 4 * np.pi / (Lambda * 365)
    # h2ph = ((np.arange(1, N + 1, 1) * dBn + np.random.normal(0, sigma_bn, N)) * 4 * np.pi / (Lambda * R * np.sin(Incidence_angle))).reshape(N)
    # h2ph = ((np.random.randn(1, N) * 333) * 4 * np.pi / (Lambda * R * np.sin(Incidence_angle))).reshape(N)
    h2ph = (porabola_bn(N, sigma_bn) * 4 * np.pi / (Lambda * R * np.sin(Incidence_angle))).reshape(N)
    fvt = v * v2ph + h * h2ph + add_gaussian_noise1(N, 5).reshape(N)
    fvt_wrap = wrap(fvt)
    print(fvt_wrap)
    fvt_cpl = np.exp(1j * fvt_wrap)
    DTFT = dtft_vh(fvt_cpl, v2ph, h2ph, W_v, W_h)
    DTFT.xjw(W_v, W_h, N)
    xjw = np.array([])
    for i in DTFT.yvalues:
        xjw = np.append(xjw, abs(i))
    return xjw / N


v = 0.01
h = 30
dBn = 10
sigma_bn = 5
dT = 35
N = 30
W_v = np.arange(-1600, 1600 + 1, 1) * 0.0001
W_h = np.arange(-60, 60 + 1, 1)
# x, y = np.meshgrid(W_v, W_h)
# bn = porabola_bn(N, sigma_bn)
# # print(bn)
# plt.figure()
# plt.plot(np.linspace(-3, 3, N), bn)
# plt.show()
# T1 = time.perf_counter()
xjw = dtft_vh_af(v, h, dBn, sigma_bn, dT, N, W_v, W_h)
# np.savetxt("data_dft/xjw_vh_linear1.txt", xjw, delimiter=",")
# np.savetxt("data_dft/xjw_vh_normal1.txt", xjw, delimiter=",")
np.savetxt("data_dft/xjw_vh_pora1.txt", xjw, delimiter=",")
# # np.savetxt("xjw_vh_nm0.txt", xjw, delimiter=",")
# # np.savetxt("xjw_vh_2_0.txt", xjw, delimiter=",")
# # np.savetxt("xjw_vh_noise.txt", xjw, delimiter=",")
# np.savetxt("xjw_vh_nm_nifg_50.txt", xjw, delimiter=",")
# T2 = time.perf_counter()
# print("程序运行时间:%s秒" % (T2 - T1))
# print(xjw.shape)

# 绘图
# xjw_vh1 = np.loadtxt("xjw_vh_nm0.txt", delimiter=",")
# xjw_vh1 = np.loadtxt("xjw_vh_2_0.txt", delimiter=",")
# xjw_vh1 = np.loadtxt("xjw_vh_noise.txt", delimiter=",")
# xjw_vh1 = np.loadtxt("xjw_vh1.txt", delimiter=",")
# xjw_vh1 = np.loadtxt("xjw_vh_nm_nifg_50.txt", delimiter=",")
# xjw = xjw_vh1.reshape(len(W_h), len(W_v))
# xi, yi = np.meshgrid(W_v, W_h)
# z = xjw

# fig = plt.figure()
# ax = fig.add_axes(Axes3D(fig))
# surf = ax.plot_surface(xi, yi, z, rstride=1, cstride=1, cmap="rainbow")
# plt.plot([0.01, 0.01], [30, 30], [0, 1.2], color="black", linewidth=2)
# # 给cmap添加标题
# # cmap="BuPu"
# ax.set_xlabel(r"Linear Displacement Rate $\Delta{v}\,$[mm/yr]", fontsize=14)
# ax.set_ylabel(r"Residual Height[m]", fontsize=14)
# # ax.set_zlabel("Temporal Coherence", fontsize=14)
# # ax.text2D(
# #     0.5, 0.9, "Noise Level=10deg,Nifg=30,$\Delta{T}$=36days,$\sigma_{Bn}=50[m]$,$\Delta{Bn}$=10[m],$\Delta{v}=10$[mm/yr],$\Delta{h}=30$[m]", fontsize=15, transform=ax.transAxes
# # )
# ax.text2D(0.5, 0.9, "Noise Level=20deg,Nifg=50,$\Delta{T}$=36days,$\sigma_{Bn}=333[m]$,$\Delta{v}=10$[mm/yr],$\Delta{h}=30$[m]", fontsize=15, transform=ax.transAxes)
# # 查看俯视图
# ax.view_init(90, -180)
# cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
# cbar.set_label(label="Temporal Coherence", fontsize=15)  # 设置颜色条刻度标签的字体大小
# plt.show()
