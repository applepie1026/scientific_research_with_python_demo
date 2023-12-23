import numpy as np
import math
import matplotlib.pyplot as plt


def wrap(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


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
    v2ph = dT * np.arange(1, N + 1, 1) * 4 * np.pi / (Lambda * 365)
    fvt = wrap(v * v2ph)
    fvt_cpl = np.exp(1j * fvt)
    DTFT = dtft_af(fvt_cpl, v2ph)
    DTFT.xjw(W)
    xjw = np.array([])
    for i in DTFT.yvalues:
        xjw = np.append(xjw, abs(i))
    return xjw / N


Lambda = 0.056
N = 30
v = 0.124
dT = 36
# fvt = v * v2ph
W = np.arange(-5000, 1600 + 1, 1) * 0.0001
# for v in [0.1, 0.124]:
#     xjw = dft_af(v, dT, N, W)
#     np.savetxt(f"xjw.txt{v}", xjw)
# xjw1 = np.loadtxt("xjw.txt0.1")
# xjw2 = np.loadtxt("xjw.txt0.124")
# dT
# for dT in [36, 60]:
#     xjw = dft_af(v, dT, N, W)
#     np.savetxt(f"xjw.txt{dT}days", xjw)

xjw3 = np.loadtxt("xjw.txt36days")
xjw4 = np.loadtxt("xjw.txt60days")
plt.figure()
ax = plt.gca()
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")
ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_position(("data", 0))
# ax.yaxis.set_ticks_position("left")
# ax.spines["left"].set_position(("data", 0))

# plt.plot(W, xjw1)
# plt.plot(W, xjw2)
plt.plot(W, xjw3, label="$\Delta{T}=36days$,$\Delta{v}$=0.124[m/yr]")
plt.plot(W, xjw4, label="$\Delta{T}=60days$,$\Delta{v}$=0.124[m/yr]")
# plt.plot([0.1, 0.1], [0, 1.1], color="red", lw=1.5, linestyle="--", label="$\Delta{v}$=0.1[m/yr]")
plt.plot([0.124, 0.124], [0, 1.1], color="Blue", lw=1.5, linestyle="--", label="$\Delta{v}$=0.124[m/yr]")
# plt.title("DFT-periodogram,wrapped phase,$\Delta{v}$=0.1,0.124[m/yr],$\Delta{T}=36days$,Nifg=30", fontsize=15)
plt.title("DFT-periodogram,wrapped phase,$\Delta{v}$=0.124[m/yr],$\Delta{T}=36,60days$,Nifg=30", fontsize=15)
plt.xlabel("Searched Linear Displacement Rate[m/yr]", fontsize=15)
plt.ylabel("Temporal Coherence", fontsize=15)
plt.xticks(np.arange(-0.52, 0.18, 0.04), fontsize=8)

# plt.xticks(np.arange(-320, 160, 20))
plt.legend()
plt.show()
