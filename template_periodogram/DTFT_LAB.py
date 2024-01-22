import numpy as np
import math
import matplotlib.pyplot as plt
import nfft


def wrap(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


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


class dtft:
    # xvalues:输入序列
    def __init__(self, xvalues=[]):
        self.yvalues = []
        self.xvalues = xvalues

    # fre:频率坐标
    def xjw(self, fre=[]):
        # （式1-1）实现yvalues为X（jw）频谱值
        for f in fre:
            p = 0
            for x in self.xvalues:
                p = math.e ** (-1j * f) * x + p
            self.yvalues.append(p)


class dtft_af:
    # xvalues:输入序列
    def __init__(self, xvalues, v2ph):
        self.yvalues = []
        self.xvalues = xvalues
        self.v2ph = v2ph

    # fre:频率坐标
    def xjw(self, fre=[]):
        # （式1-1）实现yvalues为X（jw）频谱值
        for f in fre:
            p = 0
            for x, b in zip(self.xvalues, self.v2ph):
                p = math.e ** (-1j * f * b) * x + p
            self.yvalues.append(p)


def dft_af(v, dT, N, W):
    # v2ph = dT * np.arange(1, N + 1, 1) * 4 * np.pi / (Lambda * 365)
    v2ph = (dT * np.arange(1, N + 1, 1) + np.random.normal(0, 10, N)) * 4 * np.pi / (Lambda * 365)
    # v2ph = (np.random.normal(0, 1000, N)) * 4 * np.pi / (Lambda * 365)
    fvt = wrap(v * v2ph + add_gaussian_noise1(N, 5).reshape(N))
    # fvt = wrap(v * v2ph + add_gaussian_noise(N, 30).reshape(N))
    fvt_cpl = np.exp(1j * fvt)
    DTFT = dtft_af(fvt_cpl, v2ph)
    DTFT.xjw(W)
    xjw = np.array([])
    for i in DTFT.yvalues:
        xjw = np.append(xjw, abs(i))
    return xjw / N


H = 780000  # satellite vertical height[m]
Incidence_angle = 23 * np.pi / 180  # the local incidence angle
R = H / np.cos(Incidence_angle)
Lambda = 0.056


def dft_af_h(h, dBn, N, W):
    h2ph = (np.random.normal(0, 333, N) * 4 * np.pi / (Lambda * R * np.sin(Incidence_angle))).reshape(N)
    print(h2ph.shape)
    # h2ph = dBn * np.arange(1, N + 1, 1) * 4 * np.pi / (Lambda * R * np.sin(Incidence_angle)) + np.random.normal(0, 10, N)
    # h2ph = dBn * np.ones(N) * 4 * np.pi / (Lambda * R * np.sin(Incidence_angle)) + np.random.normal(0, 5, N)
    fht = wrap(h * h2ph)
    fht_cpl = np.exp(1j * fht)
    DTFT = dtft_af(fht_cpl, h2ph)
    DTFT.xjw(W)
    xjw = np.array([])
    for i in DTFT.yvalues:
        xjw = np.append(xjw, abs(i))
    return xjw / N


N = 30
v = 0.05
dT = 36
# # fvt = v * v2ph
W = np.arange(-5000, 5000 + 1, 1) * 0.0001
# N_fft = 2048
# hz = np.arange(0, N_fft, 1)

# xjw1 = dft_af(v, dT, N, W)
# print(W[np.argmax(xjw1)])
# v = 0.1
# # dT = 10
# # N = 50
# xjw2 = dft_af(v, dT, N, W)
# # print(W[np.argmax(xjw2)])
# v = -0.1
# # dT = 1
# # N = 10
# xjw3 = dft_af(v, dT, N, W)
# xjw = dft_fft_af(v, dT, N, N_fft)
# max_index = np.argmax(xjw)
# v_est = Lambda * max_index / 2 * N_fft
# # np.savetxt("data_save1/xjw_0.05_peric_60days.txt", xjw1)
# for N in [10, 30]:
#     xjw = dft_af(v, dT, N, W)
#     np.savetxt(f"xjw_Nifg{N}.txt", xjw)
# for v in [0.05, 0.124]:
#     xjw = dft_af(v, dT, N, W)
#     np.savetxt(f"xjw{v}.txt", xjw)
# xjw1 = np.loadtxt("xjw.txt0.1")
# xjw2 = np.loadtxt("xjw.txt0.124")
# dT
# for dT in [36, 60]:
#     xjw = dft_af(v, dT, N, W)
#     np.savetxt(f"xjw.txt{dT}days", xjw)


# xjw3 = np.loadtxt("xjw.txt36days")
# xjw4 = np.loadtxt("xjw.txt60days")
# xjw5 = np.loadtxt("xjw0.05.txt")
# xjw6 = np.loadtxt("xjw0.124.txt")
# plt.figure(figsize=(10, 7))
# ax = plt.gca()
# ax.spines["right"].set_color("none")
# ax.spines["top"].set_color("none")
# ax.xaxis.set_ticks_position("bottom")
# ax.spines["bottom"].set_position(("data", 0))


# plt.plot(Lambda * 365 * hz / (2 * N_fft * 36), xjw)
# plt.plot(hz, xjw)
# plt.plot(W, xjw1, label="$\Delta{T}=36$days")
# plt.plot(W, xjw2, label="$\Delta{T}=10$days")
# plt.plot(W, xjw1, label="Nifg=30")
# plt.plot(W, xjw2, label="Nifg=50")
# plt.plot(W, xjw3, label="Nifg=10")
# plt.plot(W, xjw1, label="$\Delta{v}$=0.05[m/yr]")
# plt.plot(W, xjw2, label="$\Delta{v}$=0.1[m/yr]")
# plt.plot(W, xjw3, label="$\Delta{v}$=-0.1[m/yr]")
# plt.plot([0.1, 0.1], [0, 1.1], color="blue", lw=1.5, linestyle="--", label="$\Delta{v}$=0.1[m/yr]")
# plt.plot([0.05, 0.05], [0, 1.1], color="red", lw=1.5, linestyle="--", label="$\Delta{v}$=0.05[m/yr]")
# plt.plot([-0.1, -0.1], [0, 1.1], color="orange", lw=1.5, linestyle="--", label="$\Delta{v}$=-0.1[m/yr]")
# plt.plot([0.124, 0.124], [0, 1.1], color="Blue", lw=1.5, linestyle="--", label="$\Delta{v}$=0.124[m/yr]")
# plt.title("DFT-periodogram,$\Delta{v}$=0.05,0.124[m/yr],$\Delta{T}=36days$,Nifg=30", fontsize=15)
# plt.title("Noise Level=5deg,$\Delta{v}$=0.05[m/yr],$\Delta{T}=36$days,Nifg=30", fontsize=15)
# plt.xlabel("Searched Linear Displacement Rate[m/yr]", fontsize=15)
# plt.ylabel("$|\gamma_{temporal}|$", fontsize=15, rotation=0, labelpad=35)
# plt.xticks(np.arange(-0.52, 0.18, 0.04), fontsize=8)
# plt.xticks([-0.16, -0.10, -0.05, 0, 0.05, 0.10, 0.16])
# plt.xticks(np.arange(-320, 160, 20))
# plt.legend(loc="upper right", bbox_to_anchor=(1.12, 1), borderaxespad=0.1)
# plt.show()

# bn
N = 30
h = 30
dBn = 10
W_h = np.arange(-6000, 6000, 1) * 0.01
xjw = dft_af_h(h, dBn, N, W_h)

# xjw5 = np.loadtxt("xjw.txt30m")
plt.figure()
ax = plt.gca()
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")
ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_position(("data", 0))

plt.plot(W_h, xjw, label="$\Delta{h}$=30[m]")
plt.plot([30, 30], [0, 1.1], color="red", lw=1.5, linestyle="--", label="$\Delta{B_n}$=30[m]")
plt.title("DFT-periodogram,wrapped phase,$\Delta{h}$=30[m],$\Delta{B_n}$=10[m],Nifg=30", fontsize=15)
plt.xlabel("Searched Resudual Height[m]", fontsize=15)
plt.ylabel("Temporal Coherence", fontsize=15)
plt.show()

# v+noise
# N = 30
# v = 0.11
# dt = 36
# W = np.arange(-1600, 1600 + 1, 1) * 0.0001
# xjw = dft_af(v, dt, N, W)

# plt.figure()
# ax = plt.gca()
# ax.spines["right"].set_color("none")
# ax.spines["top"].set_color("none")
# ax.xaxis.set_ticks_position("bottom")
# ax.spines["bottom"].set_position(("data", 0))

# plt.plot(W, xjw, label="$\Delta{v}$=0.11[m/yr]")
# plt.plot([0.11, 0.11], [0, 1.1], color="red", lw=1.5, linestyle="--", label="$\Delta{v}$=0.11[m/yr]")
# plt.title("DFT-periodogram,wrapped phase,$\Delta{v}$=0.11[m/yr],$\Delta{T}=36days$,Nifg=30", fontsize=15)
# plt.xlabel("Searched Linear Displacement Rate[m/yr]", fontsize=15)
# plt.ylabel("Temporal Coherence", fontsize=15)
# plt.legend()
# plt.show()

# Nifg
# xjw5 = np.loadtxt("xjw_Nifg10.txt")
# xjw6 = np.loadtxt("xjw_Nifg30.txt")

# plt.figure()
# ax = plt.gca()
# ax.spines["right"].set_color("none")
# ax.spines["top"].set_color("none")
# ax.xaxis.set_ticks_position("bottom")
# ax.spines["bottom"].set_position(("data", 0))

# plt.plot(W, xjw5, label="Nifg=10,$\Delta{v}$=0.11[m/yr]")
# plt.plot(W, xjw6, label="Nifg=30,$\Delta{v}$=0.11[m/yr]")
# plt.plot([0.11, 0.11], [0, 1.1], color="red", lw=1.5, linestyle="--", label="$\Delta{v}$=0.11[m/yr]")
# plt.title("DFT-periodogram,wrapped phase,$\Delta{v}$=0.11[m/yr],$\Delta{T}=36days$,Nifg=30", fontsize=15)
# plt.xlabel("Searched Linear Displacement Rate[m/yr]", fontsize=15)
# plt.ylabel("$|\gamma_{temporal}|$", fontsize=15, rotation=0, labelpad=40)
# plt.legend()
# plt.show()
