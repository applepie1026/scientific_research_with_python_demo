import scientific_research_with_python_demo.utils as af
import scientific_research_with_python_demo.data_plot as dp
import matplotlib.pyplot as plt
import numpy as np

SNR = 20
noise_level = 10
Nifg = 30

time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg)
normal_baseline = np.random.normal(size=(1, Nifg)) * 333
v2ph = af.v_coef(time_baseline)
h2ph = af.h_coef(normal_baseline).T
v = np.array([[0.001, 0.01, 0.1]])
print(len(v))
signal_phase = np.zeros((3, Nifg))
noise1_all = np.zeros((3, Nifg))
for i in range(3):
    signal_phase[i] = af._coef2phase(v2ph, v[0][i])
    noise1_all[i] = af.add_gaussian_noise(signal_phase[i], SNR)
# signal = af._coef2phase(v2ph, 0.1)
# noise1 = af.add_gaussian_noise(signal, SNR)
# noise2 = af.add_gaussian_noise2(Nifg, noise_level)
# noise3 = af.add_gaussian_noise3(Nifg, noise_level)
# noise4 = af.add_gaussian_noise4(Nifg, noise_level)
# snr_check1 = af.check_snr2(signal, noise1)
# snr_check2 = af.check_snr2(signal, noise2)
# snr_check3 = af.check_snr2(signal, noise3)
# snr_check4 = af.check_snr2(signal, noise4)
# print("noise1:%s,%d db" % (noise1, snr_check1))
# print("noise2:%s,%d db" % (noise2, snr_check2))
# print("noise1_snr:%s db" % snr_check1)
# print("noise2_deg:%s db" % snr_check2)
# print("noise3_deg:%s db" % snr_check3)
# print("noise4_deg:%s db" % snr_check4)

# signal1 = signal + noise1
# signal2 = signal + noise2
# signal3 = signal + noise3
# signal4 = signal + noise4

dt = np.arange(1, Nifg + 1, 1)
# plt.figure()
# plt.plot(dt, signal1, "r", label="signal_noise")
# plt.plot(dt, signal, "b", label="signal")
# plt.plot(dt, noise1, "g", label="noise")
# plt.title("SNR=30db,dt=12,v=0.1")
# plt.legend()
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/signal_noise1_4.png")

# nosie_db
plt.figure()
plt.plot(dt, noise1_all[0] * 180 / np.pi, "r", label="v=0.001")
plt.plot(dt, noise1_all[1] * 180 / np.pi, "b", label="v=0.01")
plt.plot(dt, noise1_all[2] * 180 / np.pi, "g", label="v=0.1")
plt.title("SNR=20db,dt=12")
plt.ylabel("noise[deg]")

plt.legend()
plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/noise1_all_2.png")

# plt.figure()
# plt.plot(dt, signal3, "r", label="signal_noise")
# plt.plot(dt, signal, "b", label="signal")
# plt.legend()
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/signal_noise3.png")
# 直方图
# plt.figure()
# plt.hist(noise4 * 180 / np.pi, bins=20, edgecolor="black")
# plt.xlabel("noise[deg]")
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/noise2_2.png")
# plt.figure()
# plt.hist(noise3 / np.pi, bins=10, edgecolor="black")
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/noise2.png")
