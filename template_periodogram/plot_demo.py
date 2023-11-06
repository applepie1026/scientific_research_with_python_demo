import scientific_research_with_python_demo.data_plot as dp
import numpy as np
import matplotlib.pyplot as plt
import csv


# success_rate = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_test3.csv", delimiter=",")
# v_orig = np.arange(1, 171, 1) * 0.001
# dp.line_plot(v_orig * 1000, success_rate, "n=40db,dt=12,nifg=30", "V_test3", "v[mm/year]")

# v相关，deg相关绘制
# v_orig = np.arange(1, 171, 1) * 0.001
# success_rete = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/afV_test2.csv", delimiter=",")
# success_rete1 = success_rete[0, :]
# success_rete2 = success_rete[1, :]
# success_rete3 = success_rete[2, :]
# success_rete4 = success_rete[3, :]
# success_rete5 = success_rete[4, :]
# success_rete6 = success_rete[5, :]
# # deg = [5, 10, 20, 30, 40, 50]
# # success_rete1 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_test0_2.csv", delimiter=",")
# # success_rete2 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_test1_2.csv", delimiter=",")
# # success_rete3 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_test2_2.csv", delimiter=",")
# # success_rete4 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_test3_2.csv", delimiter=",")
# # success_rete5 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_test4_2.csv", delimiter=",")
# # success_rete6 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_test5_2.csv", delimiter=",")
# plt.figure()
# plt.plot(v_orig * 1000, success_rete1, "r", label="1deg")
# plt.plot(v_orig * 1000, success_rete2, "g", label="5deg")
# plt.plot(v_orig * 1000, success_rete3, "b", label="10deg")
# plt.plot(v_orig * 1000, success_rete4, "y", label="20deg")
# plt.plot(v_orig * 1000, success_rete5, "k", label="30deg")
# plt.plot(v_orig * 1000, success_rete6, "c", label="40deg")

# plt.xlabel(r"Linear Deformation rate $\Delta{v}\,$[mm/year]", fontsize=15)
# plt.ylabel("Success rate", fontsize=15)
# plt.title(r"Nifg=30,$\Delta{T}$=36$\,$days", fontsize=20)
# plt.legend()
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/V_test_all_2.png")

# dt相关
# v_orig = np.arange(1, 171, 1) * 0.001
# success_rete1 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dTsuccess_lab0_1.csv", delimiter=",")
# success_rete2 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dTsuccess_lab1_1.csv", delimiter=",")
# success_rete3 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dTsuccess_lab2_1.csv", delimiter=",")
# success_rete4 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dTsuccess_lab3_1.csv", delimiter=",")
# success_rete5 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dTsuccess_lab4_1.csv", delimiter=",")
# success_rete6 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dTsuccess_lab5_1.csv", delimiter=",")
# success_rate7 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dTsuccess_lab6_1.csv", delimiter=",")
# success_rate8 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dTsuccess_lab7_1.csv", delimiter=",")


# plt.figure()
# plt.plot(v_orig * 1000, success_rete1, "r", label=r"$\Delta{T}$=12")
# plt.plot(v_orig * 1000, success_rete2, "g", label=r"$\Delta{T}$=24")
# plt.plot(v_orig * 1000, success_rete3, "b", label=r"$\Delta{T}$=36")
# plt.plot(v_orig * 1000, success_rete4, "y", label=r"$\Delta{T}$=48")
# plt.plot(v_orig * 1000, success_rete5, "k", label=r"$\Delta{T}$=60")
# plt.plot(v_orig * 1000, success_rete6, "c", label=r"$\Delta{T}$=72")
# plt.plot(v_orig * 1000, success_rate7, "m", label=r"$\Delta{T}$=84")
# plt.plot(v_orig * 1000, success_rate8, "orange", label=r"$\Delta{T}$=96")

# plt.xlabel(r"Linear Deformation rate $\Delta{v}\,$[mm/year]", fontsize=15)
# plt.ylabel("Success rate", fontsize=15)
# plt.title(r"Noise_level=10deg,Nifg=30", fontsize=20)
# # plt.tick_params(labelsize=20)
# plt.legend(fontsize=8)
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/dT_test_all_1.png")

# # nifg相关
# v_orig = np.arange(1, 171, 1) * 0.001
# Nifg = [10, 20, 30, 40, 50]
# success_rete1 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Nifgsuccess_lab0_1.csv", delimiter=",")
# success_rete2 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Nifgsuccess_lab1_1.csv", delimiter=",")
# success_rete3 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Nifgsuccess_lab2_1.csv", delimiter=",")
# success_rete4 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Nifgsuccess_lab3_1.csv", delimiter=",")
# success_rete5 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Nifgsuccess_lab4_1.csv", delimiter=",")

# plt.figure()
# plt.plot(v_orig * 1000, success_rete1, "r", label="Nifg=10")
# plt.plot(v_orig * 1000, success_rete2, "g", label="Nifg=20")
# plt.plot(v_orig * 1000, success_rete3, "b", label="Nifg=30")
# plt.plot(v_orig * 1000, success_rete4, "y", label="Nifg=40")
# plt.plot(v_orig * 1000, success_rete5, "k", label="Nifg=50")

# plt.xlabel(r"Linear Deformation rate $\Delta{v}\,$[mm/year]", fontsize=15)
# plt.ylabel("Success rate", fontsize=15)
# plt.title(r"Noise_level=10deg,$\Delta{T}$=36$\,$days", fontsize=20)
# # plt.tick_params(labelsize=20)
# plt.legend()
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/Nifg_test_all_1.png")

# # dt,v
# dt_range = np.arange(10, 81, 1) * 0.1
# success_rete1 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dTsuccess_test1.csv", delimiter=",")
# success_rete2 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dTsuccess_test2.csv", delimiter=",")
# success_rete3 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dTsuccess_test3.csv", delimiter=",")
# success_rete4 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dTsuccess_test4.csv", delimiter=",")

# plt.figure()
# plt.plot(12 * dt_range, success_rete1, "r", label="v=0.005")
# plt.plot(12 * dt_range, success_rete2, "g", label="v=0.01")
# plt.plot(12 * dt_range, success_rete3, "b", label="v=0.1")
# plt.plot(12 * dt_range, success_rete4, "y", label="v=0.15")

# plt.xlabel("dt[days]")
# plt.ylabel("success rate")
# plt.title("noise=10deg,nifg=30")
# plt.legend()
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/dT_test_all_2.png")

# dt,v2
dt_range = np.arange(10, 90.5, 0.5)
v = [0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.15]
success_rate = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/afV_T_test2.csv", delimiter=",")
# success_rate1 = success_rate[:, 0]
# success_rate2 = success_rate[:, 1]
# success_rate3 = success_rate[:, 2]
# success_rate4 = success_rate[:, 3]
# success_rate5 = success_rate[:, 4]
# success_rate6 = success_rate[:, 5]
# success_rate7 = success_rate[:, 6]
# success_rate8 = success_rate[:, 7]

color = ["r", "g", "b", "y", "k", "c", "m", "orange"]

plt.figure()
for i in range(len(v)):
    plt.plot(dt_range, success_rate[:, i], color[i], label=r"$\Delta{v}$=%smm/year" % "%.0f" % (v[i] * 1000))

# plt.plot(dt_range, success_rate1, "r", label=r"$\Delta{v}$=1mm/year")
# plt.plot(dt_range, success_rate2, "g", label=r"$\Delta{v}$=5mm/year")
# plt.plot(dt_range, success_rate3, "b", label=r"$\Delta{v}$=10mm/year")
# plt.plot(dt_range, success_rate4, "y", label=r"$\Delta{v}$=15mm/year")
# plt.plot(dt_range, success_rate5, "k", label=r"$\Delta{v}$=20mm/year")
# plt.plot(dt_range, success_rate6, "c", label=r"$\Delta{v}$=50mm/year")
# plt.plot(dt_range, success_rate7, "m", label=r"$\Delta{v}$=100mm/year")
# plt.plot(dt_range, success_rate8, "orange", label=r"$\Delta{v}$=150mm/year")

plt.xlabel(r"$\Delta{T}t$[days]")
plt.ylabel("Success rate")
plt.title("Noise_level=10deg,Nifg=30")
plt.legend(fontsize=8)
plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/dT_v_test_all_3.png")

# # h
# h_orig = np.arange(1, 131, 1)
# noise_level = [1, 5, 10, 20, 30, 40]

# success_rete1 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Hsuccess_test0_1.csv", delimiter=",")
# success_rete2 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Hsuccess_test1_1.csv", delimiter=",")
# success_rete3 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Hsuccess_test2_1.csv", delimiter=",")
# success_rete4 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Hsuccess_test3_1.csv", delimiter=",")
# success_rete5 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Hsuccess_test4_1.csv", delimiter=",")
# success_rete6 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Hsuccess_test5_1.csv", delimiter=",")

# plt.figure()
# plt.plot(h_orig, success_rete1, "r", label="1deg")
# plt.plot(h_orig, success_rete2, "g", label="5deg")
# plt.plot(h_orig, success_rete3, "b", label="10deg")
# plt.plot(h_orig, success_rete4, "y", label="20deg")
# plt.plot(h_orig, success_rete5, "k", label="30deg")
# plt.plot(h_orig, success_rete6, "c", label="40deg")

# plt.xlabel("h[m]")
# plt.ylabel("success rate")
# plt.title("nifg=30,dt=36")
# plt.legend()
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/H_test_all_1.png")

# # 修改csv文件
# success_rate = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dT_Vsuccess_lab80_1.csv", delimiter=" ")
# a = success_rate[:, 1]
# print(a.shape)
# print(success_rate.shape)
# success_rate[:, 1] = a / 12
# np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dT_Vsuccess_lab_1.csv", success_rate, delimiter=",")
# # 加载csv文件的数组
