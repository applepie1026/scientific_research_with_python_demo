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


def add_gaussian_noise1(Nifg, noise_level_set):
    noise_phase = np.random.normal(0, np.pi * noise_level_set / 180, (Nifg, 1))
    return noise_phase


def DFT_phase_flatten(wrapped_phase, h2ph, v2ph, flatten_num, W_v, N):
    DFT_flatten = np.zeros(len(W_v))
    for h in np.random.randint(-30, 30, flatten_num):
        # for h in np.random.normal(0, 60, flatten_num):
        flatten_phase = wrapped_phase - h * h2ph
        # flatten_signal = np.exp(1j * flatten_phase)
        # DFT_signal = dtft_af_all(flatten_signal, v2ph).xjw(W_v, N)
        DFT_signal = dtft_af_all_array(flatten_phase, v2ph).xjw(W_v, N)
        DFT_flatten += DFT_signal

    return DFT_flatten / flatten_num


# def DFT_phase_flatten_array(wrapped_phase, h2ph, v2ph, flatten_num, W_v, N):
#     # ----------------------------------------------------------------------------------------------
#     # 创建flatten后的wrapped_phase矩阵，维度为（flatten_num,N,len(W_v)），第一维度的每个元素进行一个DFT
#     # ----------------------------------------------------------------------------------------------
#     # 创建随机h的数组，三维数组，维度为（flatten_num,1,1）第一维度保存flatten_num个h值
#     h_flatten = (np.random.randint(-60, 60, flatten_num)).reshape(flatten_num, 1, 1)
#     # print(h_flatten.shape)
#     # 创建h2ph矩阵快，维度为（N，len(W_v)）
#     h2ph_flatten = h2ph.reshape(N, 1) * np.ones((1, len(W_v)))
#     # print(h2ph_flatten.shape)
#     # 创建h相位矩阵，维度为（flatten_num,N,len(W_v)）,第一维度保存flatten_num个h_searched_phase
#     h_phase_flatten = h_flatten * h2ph_flatten
#     # print(h_phase_flatten.shape)
#     # 创建wrapped_phase矩阵，维度为（N,N,len(W_v)）,第一维度保存N个wrapped_phase
#     wrapped_phase_flatten = (wrapped_phase.reshape(N, 1) * np.ones((1, len(W_v)))) * np.ones((flatten_num, 1, 1))
#     # print(wrapped_phase_flatten.shape)
#     # 创建flatten_phase矩阵，维度为（flatten_num,N,len(W_v)）,得到的是减去随机h的wrapped_phase矩阵
#     flatten_phase = wrapped_phase_flatten - h_phase_flatten
#     # print(flatten_phase.shape)
#     # ----------------------------------------------------------------------------------------------
#     # 对flatten_phase矩阵的第一维度的每个元素（每个元素的维度为（N，len(W_v)）的矩阵）进行DFT
#     # ----------------------------------------------------------------------------------------------
#     # 创建v_serached_phase矩阵，维度为（N，len(W_v)）
#     search_phase = v2ph.reshape(N, 1) * W_v
#     # print(search_phase.shape)

#     # 创建flatten_num个v_serached_phase矩阵，维度为（flatten_num,N,len(W_v)）
#     search_phase_flatten = search_phase * np.ones((flatten_num, 1, 1))
#     # print(search_phase_flatten.shape)

#     # 计算每个coh_phase矩阵，
#     coh_phase = flatten_phase - search_phase_flatten
#     # print(coh_phase.shape)
#     # 对每个DFT矩阵进行按列求和，得到每个元素矩阵对应的DTF，也就是经过随机h flatten后的DTF
#     xjw_stack = abs(np.sum(np.exp(1j * coh_phase), axis=1) / N)
#     # 将所有的DTF矩阵按列求和，得到最终的DTF
#     DFT_flatten = np.sum(xjw_stack, axis=0) / flatten_num
#     return DFT_flatten


class dtft_af_all:
    # xvalues:输入序列
    def __init__(self, xvalues, p2ph):
        self.yvalues = []
        self.xvalues = xvalues
        self.p2ph = p2ph

    # fre:频率坐标
    def xjw(self, fre=[], N=30):
        # （式1-1）实现yvalues为X（jw）频谱值
        for f in fre:
            p = 0
            for x, b in zip(self.xvalues, self.p2ph):
                p = math.e ** (-1j * f * b) * x + p
            self.yvalues.append(p)
        xjw_dft = np.array([])
        for i in self.yvalues:
            xjw_dft = np.append(xjw_dft, abs(i))
        return xjw_dft / N


class dtft_af_all_array:
    def __init__(self, wrapped_phase, p2ph):
        self.yvalues = []
        self.wrapped_phase = wrapped_phase
        self.p2ph = p2ph

    # fre:频率坐标
    def xjw(self, fre=[], N=30):
        # （式1-1）实现yvalues为X（jw）频谱值
        searched_phase = self.p2ph.reshape(N, 1) * fre
        # print(searched_phase.shape)
        coh_phase = self.wrapped_phase.reshape(N, 1) * np.ones((1, len(fre))) - searched_phase
        # print(coh_phase.shape)
        xjw_dft = np.sum(np.exp(1j * coh_phase), axis=0) / N
        coh = abs(xjw_dft)
        # print(xjw_dft.shape, coh.shape)
        # coh_phase = self.wrapped_phase.reshape(N, 1) - searched_phase
        # for i in self.yvalues:
        #     xjw_dft = np.append(xjw_dft, abs(i))
        return coh


class dtft_af_h_pv_array:
    # xvalues:输入序列
    def __init__(self, wrap_phase, h2ph, v2ph, v_est):
        self.yvalues = []
        self.phase_h = wrap_phase - v_est * v2ph
        self.h2ph = h2ph

    # fre:频率坐标
    def xjw(self, fre=[], N=30):
        # （式1-1）实现yvalues为X（jw）频谱值
        searched_phase = self.h2ph.reshape(N, 1) * fre
        coh_phase = self.phase_h.reshape(N, 1) * np.ones((1, len(fre))) - searched_phase
        xjw_dft = np.sum(np.exp(1j * coh_phase), axis=0) / N
        coh_all = abs(xjw_dft)

        return coh_all


class dtft_af_h_pv:
    # xvalues:输入序列
    def __init__(self, xvalues, h2ph, v2ph, v_est):
        self.yvalues = []
        self.xvalues = xvalues
        self.v2ph = v2ph
        self.h2ph = h2ph
        self.v_est = v_est

    # fre:频率坐标
    def xjw(self, fre=[], N=30):
        # （式1-1）实现yvalues为X（jw）频谱值
        for f in fre:
            p = 0
            for x, v_b, h_b in zip(self.xvalues, self.v2ph, self.h2ph):
                p = math.e ** (-1j * (self.v_est * v_b + f * h_b)) * x + p
            self.yvalues.append(p)
        xjw_dft = np.array([])
        for i in self.yvalues:
            xjw_dft = np.append(xjw_dft, abs(i))

        return xjw_dft / N


def dtft_vflatten_af(v, h, dBn, sigma_bn, dT, N, flatten_num, W_v, W_h):
    v2ph = dT * np.arange(1, N + 1, 1) * 4 * np.pi / (Lambda * 365)
    # h2ph = ((np.arange(1, N + 1, 1) * dBn + np.random.normal(0, sigma_bn, N)) * 4 * np.pi / (Lambda * R * np.sin(Incidence_angle))).reshape(N)
    h2ph = ((np.random.randn(1, N) * 333) * 4 * np.pi / (Lambda * R * np.sin(Incidence_angle))).reshape(N)
    # h2ph = (porabola_bn(N, sigma_bn) * 4 * np.pi / (Lambda * R * np.sin(Incidence_angle))).reshape(N)
    fvt = v * v2ph + h * h2ph + add_gaussian_noise1(N, 5).reshape(N)
    fvt_wrap = wrap(fvt)
    DFT_flatten = DFT_phase_flatten(fvt_wrap, h2ph, v2ph, flatten_num, W_v, N)
    # DFT_h, v_est, h_est = est_vph1(fvt_wrap, DFT_flatten, h2ph, v2ph, W_h, W_v, N)
    # DFT_h, v_est, h_est = est_vph2(fvt_wrap, DFT_flatten, h2ph, v2ph, W_h, W_v, N)
    DFT_h, v_est, h_est = est_vph3(fvt_wrap, DFT_flatten, h2ph, v2ph, W_h, W_v, N)
    return DFT_flatten, DFT_h, v_est, h_est


# def dtft_vflatten_af_array(v, h, dBn, sigma_bn, dT, N, W_v, W_h):
#     v2ph = dT * np.arange(1, N + 1, 1) * 4 * np.pi / (Lambda * 365)
#     # h2ph = ((np.arange(1, N + 1, 1) * dBn + np.random.normal(0, sigma_bn, N)) * 4 * np.pi / (Lambda * R * np.sin(Incidence_angle))).reshape(N)
#     h2ph = ((np.random.randn(1, N) * 333) * 4 * np.pi / (Lambda * R * np.sin(Incidence_angle))).reshape(N)
#     # h2ph = (porabola_bn(N, sigma_bn) * 4 * np.pi / (Lambda * R * np.sin(Incidence_angle))).reshape(N)
#     fvt = v * v2ph + h * h2ph + add_gaussian_noise1(N, 5).reshape(N)
#     fvt_wrap = wrap(fvt)
#     DFT_flatten = DFT_phase_flatten_array(fvt_wrap, h2ph, v2ph, 100, W_v, N)
#     # DFT_h, v_est, h_est = est_vph1(fvt_wrap, DFT_flatten, h2ph, v2ph, W_h, W_v, N)
#     DFT_h, v_est, h_est = est_vph2(fvt_wrap, DFT_flatten, h2ph, v2ph, W_h, W_v, N)
#     return DFT_flatten, DFT_h, v_est, h_est


def est_vph1(wrapped_phase, DFT_flatten, h2ph, v2ph, W_h, W_v, N):
    v = W_v[np.argmax(DFT_flatten)]
    phase_h = wrapped_phase - v * v2ph
    # DFT_h = dtft_af_all(np.exp(1j * phase_h), h2ph).xjw(W_h, N)
    DFT_h = dtft_af_all_array(phase_h, h2ph).xjw(W_h, N)
    h = W_h[np.argmax(DFT_h)]
    return DFT_h, v, h


# def est_vph2(wrapped_phase, DFT_flatten, h2ph, v2ph, W_h, W_v, N):
#     v_est = W_v[np.argmax(DFT_flatten)]
#     fvh_cpl = np.exp(1j * wrapped_phase)
#     DFT_h = dtft_af_h_pv(fvh_cpl, h2ph, v2ph, v_est).xjw(W_h, N)
#     # DFT_h = dtft_af_h_pv_array(wrapped_phase, h2ph, v2ph, v_est).xjw(W_h, N)
#     h_est = np.round(W_h[np.argmax(DFT_h)], 1)
#     phase_v = wrapped_phase - h_est * h2ph
#     DFT_v = dtft_af_all_array(phase_v, v2ph).xjw(W_v, N)
#     v_est = np.round(W_v[np.argmax(DFT_v)], 4)
#     return DFT_h, v_est, h_est


def est_vph3(wrapped_phase, DFT_flatten, h2ph, v2ph, W_h, W_v, N):
    v_est1 = W_v[np.argmax(DFT_flatten)]
    fvh_cpl = np.exp(1j * wrapped_phase)
    DFT_h = dtft_af_h_pv(fvh_cpl, h2ph, v2ph, v_est1).xjw(W_h, N)
    # DFT_h = dtft_af_h_pv_array(wrapped_phase, h2ph, v2ph, v_est).xjw(W_h, N)
    h_est1 = np.round(W_h[np.argmax(DFT_h)], 1)
    phase_v = wrapped_phase - h_est1 * h2ph
    DFT_v = dtft_af_all_array(phase_v, v2ph).xjw(W_v, N)
    v_est2 = np.round(W_v[np.argmax(DFT_v)], 4)
    phase_h = wrapped_phase - v_est2 * v2ph
    DFT_h = dtft_af_all_array(phase_h, h2ph).xjw(W_h, N)
    h_est2 = np.round(W_h[np.argmax(DFT_h)], 1)
    return DFT_h, v_est2, h_est2


v = 0.1
h = 10
dBn = 10
sigma_bn = 5
dT = 35
N = 30
flatten_num = 200
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
for i in range(1000):
    xjw, xjw_h, v_est1, h_est1 = dtft_vflatten_af(v, h, dBn, sigma_bn, dT, N, flatten_num, W_v, W_h)
    v_data[i] = v_est1
    h_data[i] = h_est1
    if abs(v_est1 - v) <= 0.0005 and abs(h_est1 - h) <= 0.5:
        success_num += 1
    # print(f"v_est={v_est1},h_est={h_est1}")

T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
print(f"success_rate={success_num/1000}")
# 求RMSE
# v_rmse = np.sqrt(np.mean((v_data - v) ** 2))
# h_rmse = np.sqrt(np.mean((h_data - h) ** 2))
# print(f"v_rmse={v_rmse},h_rmse={h_rmse}")
# np.savetxt("scientific_research_with_python_demo/data_save/DFT/DFT_vPh_v_data_2.txt", v_data)
# np.savetxt("scientific_research_with_python_demo/data_save/DFT/DFT_vPh_h_data_2.txt", h_data)
# print(f"v={v_est1},h={h_est1}")
# plt.figure()
# ax = plt.gca()
# ax.spines["right"].set_color("none")
# ax.spines["top"].set_color("none")
# ax.xaxis.set_ticks_position("bottom")
# ax.spines["bottom"].set_position(("data", 0))
# plt.plot(W_v, xjw)
# plt.plot(
#     [v, v],
#     [0, 0.5],
#     color="red",
#     lw=1,
#     linestyle="--",
#     label="$\Delta{v}$=5[mm/yr]",
# )
# plt.show()
# plt.savefig("scientific_research_with_python_demo/plot/DFT/DFT_flatten_v.png", dpi=300)
# print("data plot")

# plt.figure()
# ax = plt.gca()
# ax.spines["right"].set_color("none")
# ax.spines["top"].set_color("none")
# ax.xaxis.set_ticks_position("bottom")
# ax.spines["bottom"].set_position(("data", 0))
# plt.plot(W_h, xjw_h)
# plt.plot(
#     [h, h],
#     [0, 1.1],
#     color="red",
#     lw=1,
#     linestyle="--",
#     label="$\Delta{v}$=5[mm/yr]",
# )
# plt.show()
# plt.savefig("scientific_research_with_python_demo/plot/DFT/DFT_flatten_h.png", dpi=300)
# print("data plot")


# data_size = np.ones((2, 1, 1))
# x = np.ones((3, 1))
# x1 = np.ones((3, 2))
# x2 = 2 * np.ones((3, 1))
# x3 = np.array([[1, 1], [2, 2], [3, 3]])
# print(x3.shape)
# print(x3)
# y = x3 * data_size
# # y2 = x2 * data_size
# print(y)
# print(np.sum(y, axis=1))
# print(y2)
# print(y - y2)
# print(x2.reshape(2, 1, 1))
# # 对y的第一维度对应的元素逐个按列求和
# z = np.sum(np.exp(1j * y), axis=1)
# print(z)
# print(abs(z) / 3)
# z_sum = np.sum(abs(z) / 3, axis=0)
# print(z_sum)

# DFT_phase_flatten_array(np.arange(1, 3, 1), np.arange(1, 3, 1), np.arange(1, 3, 1), 100, np.arange(1, 4, 1), 2)
