import scientific_research_with_python_demo.data_plot as dp
import numpy as np
import matplotlib.pyplot as plt
import csv


# success_rate = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_test3.csv", delimiter=",")
# v_orig = np.arange(1, 171, 1) * 0.001
# dp.line_plot(v_orig * 1000, success_rate, "n=40db,dt=12,nifg=30", "V_test3", "v[mm/year]")

# v相关，deg相关绘制
# v_orig = np.arange(1, 171, 1) * 0.001
# success_rete = np.loadtxt("scientific_research_with_python_demo/data_save/afV_H_mutiple_demo_v_noise1.txt", delimiter=",")

# success_rete1 = success_rete[0, :]
# success_rete2 = success_rete[1, :]
# success_rete3 = success_rete[2, :]
# success_rete4 = success_rete[3, :]
# success_rete5 = success_rete[4, :]
# success_rete6 = success_rete[5, :]
# plt.figure()
# plt.plot(v_orig * 1000, success_rete1, "r", label="1deg")
# plt.plot(v_orig * 1000, success_rete2, "g", label="5deg")
# plt.plot(v_orig * 1000, success_rete3, "b", label="10deg")
# plt.plot(v_orig * 1000, success_rete4, "y", label="20deg")
# plt.plot(v_orig * 1000, success_rete5, "k", label="30deg")
# plt.plot(v_orig * 1000, success_rete6, "c", label="40deg")

# plt.xlabel(r"Linear Displacement Rate $\Delta{v}\,$[mm/yr]", fontsize=15)
# plt.ylabel("Success rate", fontsize=15)
# plt.title(r"Nifg=30,$\Delta{T}$=36$\,$days", fontsize=20)
# plt.legend()
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/V_noise1.png")

# dt相关
# v_orig = np.arange(1, 171, 1) * 0.001
# success_rate = np.loadtxt("scientific_research_with_python_demo/data_save/afV_H_mutiple_demo_v_dT1.txt", delimiter=",")
# success_rate1 = success_rate[0, :]
# success_rate2 = success_rate[1, :]
# success_rate3 = success_rate[2, :]
# success_rate4 = success_rate[3, :]
# success_rate5 = success_rate[4, :]
# success_rate6 = success_rate[5, :]
# plt.figure()
# # color = ["r", "g", "b", "y", "k", "c", "m", "orange"]
# # rate_data = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/afV_T_test4.csv", delimiter=",")
# # for i in range(6):
# #     success_rate = rate_data[i, :]
# #     plt.plot(v_orig * 1000, success_rate, color[i], label=r"$\Delta{T}$=%sdays" % ((i + 1) * 12))
# plt.plot(v_orig * 1000, success_rate1, "r", label=r"$\Delta{T}$=12")
# plt.plot(v_orig * 1000, success_rate2, "g", label=r"$\Delta{T}$=24")
# plt.plot(v_orig * 1000, success_rate3, "b", label=r"$\Delta{T}$=36")
# plt.plot(v_orig * 1000, success_rate4, "y", label=r"$\Delta{T}$=48")
# plt.plot(v_orig * 1000, success_rate5, "k", label=r"$\Delta{T}$=60")
# plt.plot(v_orig * 1000, success_rate6, "c", label=r"$\Delta{T}$=72")


# plt.xlabel(r"Linear Displacement Rate $\Delta{v}\,$[mm/yr]", fontsize=15)
# plt.ylabel("Success rate", fontsize=15)
# plt.title(r"Noise Level=5deg,Nifg=30", fontsize=20)
# # plt.tick_params(labelsize=20)
# plt.legend(fontsize=8, loc="upper right")
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/V_dT1.png")

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
# dt_range = np.arange(10, 90.5, 0.5)
# v = [0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.15]
# success_rate = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/afV_T_test2.csv", delimiter=",")
# # success_rate1 = success_rate[:, 0]
# # success_rate2 = success_rate[:, 1]
# # success_rate3 = success_rate[:, 2]
# # success_rate4 = success_rate[:, 3]
# # success_rate5 = success_rate[:, 4]
# # success_rate6 = success_rate[:, 5]
# # success_rate7 = success_rate[:, 6]
# # success_rate8 = success_rate[:, 7]

# color = ["r", "g", "b", "y", "k", "c", "m", "orange"]

# plt.figure()
# for i in range(len(v)):
#     plt.plot(dt_range, success_rate[:, i], color[i], label=r"$\Delta{v}$=%smm/yr" % "%.0f" % (v[i] * 1000))

# # plt.plot(dt_range, success_rate1, "r", label=r"$\Delta{v}$=1mm/year")
# # plt.plot(dt_range, success_rate2, "g", label=r"$\Delta{v}$=5mm/year")
# # plt.plot(dt_range, success_rate3, "b", label=r"$\Delta{v}$=10mm/year")
# # plt.plot(dt_range, success_rate4, "y", label=r"$\Delta{v}$=15mm/year")
# # plt.plot(dt_range, success_rate5, "k", label=r"$\Delta{v}$=20mm/year")
# # plt.plot(dt_range, success_rate6, "c", label=r"$\Delta{v}$=50mm/year")
# # plt.plot(dt_range, success_rate7, "m", label=r"$\Delta{v}$=100mm/year")
# # plt.plot(dt_range, success_rate8, "orange", label=r"$\Delta{v}$=150mm/year")

# plt.xlabel(r"$\Delta{T}$[days]", fontsize=15)
# plt.ylabel("Success rate", fontsize=15)
# plt.title("Noise_level=5deg,Nifg=30", fontsize=20)
# plt.legend(fontsize=8)
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/dT_v_test_all_3.png")

# # 修改csv文件
# success_rate = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dT_Vsuccess_lab80_1.csv", delimiter=" ")
# a = success_rate[:, 1]
# print(a.shape)
# print(success_rate.shape)
# success_rate[:, 1] = a / 12
# np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dT_Vsuccess_lab_1.csv", success_rate, delimiter=",")
# # 加载csv文件的数组

# error solution1
# data = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/afV_T_error_solusion_solution_data.csv", delimiter=",")
# plt.figure(figsize=(20, 10))
# # for i in range(170):
# #     plt.hist(data[i], bins=100, edgecolor="black")
# solution = data.reshape((170 * 1000, 1))
# plt.hist(solution[0 : 150 * 1000 + 1], bins=400, edgecolor="black")
# plt.xlabel(r"Linear defomation rate $\Delta{v}$[m/year]", fontsize=15)
# plt.ylabel("Count", fontsize=15)
# plt.xlim(-0.2, 0.2)
# plt.title(r"$\Delta{T}$=10days,Nifg=30", fontsize=20)
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/error_solution_3.png")

# error solution2
# v = [0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.15]
# v = [0.01, 0.015, 0.02]
# v = [0.005]
# v = [0.001]
# v = [0.15]
# v = [0.1]
# v = [0.005, 0.02, 0.1, 0.15]
# for i in [0]:
#     plt.figure(figsize=(10, 5))
#     plt.xlabel(r"Linear defomation rate $\Delta{v}$[m/year]", fontsize=15)
#     plt.ylabel("Count", fontsize=15)
#     plt.title(r"$\Delta{T}$=65days,Nifg=30,Noise_level=5deg", fontsize=20)
#     #     # dt=10
#     # true_data = np.loadtxt(f"scientific_research_with_python_demo/data_save/V_data2_{i+1}true.txt", delimiter=",")
#     # error_data = np.loadtxt(f"scientific_research_with_python_demo/data_save/V_data2_{i+1}error.txt", delimiter=",")
#     # dt = 65
#     true_data = np.loadtxt(f"scientific_research_with_python_demo/data_save/V_data3_{i+1}true.txt", delimiter=",")
#     error_data = np.loadtxt(f"scientific_research_with_python_demo/data_save/V_data3_{i+1}error.txt", delimiter=",")
#     plt.hist(true_data, bins=50, edgecolor="black", alpha=1.0, label=[f"true_{v[i]}"], color=["red"], range=(-0.0005 + v[i], 0.0005 + v[i]))
#     plt.hist(error_data, bins=50, edgecolor="black", alpha=1.0, label=[f"erro_{v[i]}"], color=["blue"], range=(-0.004 - 0.15, 0.004 - 0.15))
#     plt.legend()
#     # plt.xlim((v[i] - 0.006, v[i] + 0.006))
#     # dt=10
#     # plt.savefig(f"scientific_research_with_python_demo/plot/V_data2_{i+1}.png", bbox_inches="tight")
#     # dt=65
#     plt.savefig(f"scientific_research_with_python_demo/plot/V_data3_{i+1}.png", bbox_inches="tight")
# plt.xlim(v[i] - 0.1, v[i] + 0.005)
# plt.xlim(-0.15 - 0.0005, -0.15 + 0.005)
# plt.savefig(f"/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/solution_data_dt10/solution_{v[i]}.png", bbox_inches="tight")
# plt.savefig(f"/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/solution_data_dt65/erro_solution_{v[i]}.png", bbox_inches="tight")

# plt.xlim(-0.2, 0.2)

# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/error_solution_3.png")

# error solution dt=65
# v = [0.005, 0.02, 0.1, 0.15]
# true_data_list = []
# error_data_list = []
# for i in range(len(v)):
#     true_data_list.append(np.loadtxt(f"scientific_research_with_python_demo/data_save/V_data3_{i+1}true.txt", delimiter=","))
#     error_data_list.append(np.loadtxt(f"scientific_research_with_python_demo/data_save/V_data3_{i+1}error.txt", delimiter=","))

# true_data = np.concatenate(true_data_list, axis=0)
# error_data = np.concatenate(error_data_list, axis=0)
# print(true_data)
# print(error_data)
# # plt.xlim(v[i] - 0.006, v[i] + 0.006)
# # dt=10
# # plt.savefig(f"scientific_research_with_python_demo/plot/V_data2_{i+1}.png", bbox_inches="tight")
# # dt=65
# plt.figure(figsize=(8, 5))
# plt.xlabel(r"Linear defomation rate $\Delta{v}$[m/year]", fontsize=15)
# plt.ylabel("Count", fontsize=15)
# plt.title(r"$\Delta{T}$=65days,Nifg=30,Noise_level=5deg", fontsize=20)
# plt.hist([true_data, error_data], bins=30, edgecolor="black", alpha=1.0, label=[f"true_{v[i]}", f"erro_{v[i]}"], color=["red", "blue"])
# plt.legend()
# plt.savefig(f"scientific_research_with_python_demo/plot/V_data3_all.png", bbox_inches="tight")
#  error solution3
# v = [0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.15]
# plt.figure(figsize=(40, 40))
# for i in range(len(v)):
#     plt.subplot(4, 2, i + 1)
#     plt.xlabel(r"Linear defomation rate $\Delta{v}$[m/year]")
#     plt.ylabel("Count", fontsize=15)
#     plt.title(r"$\Delta{T}$=65days,Nifg=30,Noise_level=5deg")
#     true_data = np.loadtxt(
#         f"/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/solution_data_dt10/afV_dt_error_solusion_{v[i]}_true_data.csv",
#         delimiter=",",
#     )
#     error_data = np.loadtxt(
#         f"/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/solution_data_dt10/afV_dt_error_solusion_{v[i]}_error_data.csv",
#         delimiter=",",
#     )
#     plt.hist([true_data, error_data], bins=50, edgecolor="black", alpha=1.0, label=[f"true_{v[i]}", f"erro_{v[i]}"], color=["red", "blue"])
#     plt.xlim(v[i] - 0.006, v[i] + 0.006)
#     plt.title(r"$\Delta{v}$=%s" % v[i], loc="center", fontsize=20)
#     plt.legend(fontsize=15)
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/solution_data_dt10/solution_dt10_all.png", bbox_inches="tight")

# 123-160mm error solution
# v = np.arange(124, 161, 1) * 0.001
# # true_data = np.loadtxt(
# #     "/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/solution_data_dt36/afV_dt36_solusion_true_data.csv", delimiter=","
# # )
# # error_data = np.loadtxt(
# #     "/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/solution_data_dt36/afV_dt36_solusion_error_data.csv", delimiter=","
# # )
# true_data = np.loadtxt("scientific_research_with_python_demo/data_save/V_data1_true.txt", delimiter=",")
# error_data = np.loadtxt("scientific_research_with_python_demo/data_save/V_data1_error.txt", delimiter=",")
# plt.figure(figsize=(15, 5))
# plt.xlabel(r"Linear Displacment Rate $\Delta{v}$[m/yr]", fontsize=15)
# plt.ylabel("Count", fontsize=15)
# plt.title(r"$\Delta{T}$=36days,Nifg=30,Noise Level=5deg", fontsize=20)
# plt.hist([true_data, error_data], bins=160, edgecolor="black", alpha=1.0, label=[f"true", f"erro"], color=["red", "blue"], width=0.001)
# plt.xlim(-0.2, 0.2)
# plt.legend(fontsize=10, loc="upper right")
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/V_data1.png", bbox_inches="tight")

# success_rate = np.loadtxt("scientific_research_with_python_demo/data_save/afV_H_mutiple_demo2.txt", delimiter=",")
# print(success_rate.shape)
# v = np.arange(1, 171, 1) * 0.001
# plt.figure()
# plt.plot(v * 1000, success_rate, "r", label="H=10m")
# plt.xlabel(r"Linear defomation rate $\Delta{v}$[m/year]", fontsize=15)

# plt.title(r"$\Delta{T}$=36days,Nifg=30,h=10m,Noise_level=5deg", fontsize=15)
# plt.savefig("scientific_research_with_python_demo/plot/afV_H_mutiple_demo2.png")
