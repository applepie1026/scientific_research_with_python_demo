import scientific_research_with_python_demo.data_plot as dp
import numpy as np
import matplotlib.pyplot as plt
import csv


# success_rate = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_test3.csv", delimiter=",")
# v_orig = np.arange(1, 171, 1) * 0.001
# dp.line_plot(v_orig * 1000, success_rate, "n=40db,dt=12,nifg=30", "V_test3", "v[mm/year]")

# v相关，deg相关绘制
# v_orig = np.arange(1, 171, 1) * 0.001
# success_rete1 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_test0_2.csv", delimiter=",")
# success_rete2 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_test1_2.csv", delimiter=",")
# success_rete3 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_test2_2.csv", delimiter=",")
# success_rete4 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_test3_2.csv", delimiter=",")
# success_rete5 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_test4_2.csv", delimiter=",")
# success_rete6 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_test5_2.csv", delimiter=",")
# deg = [1, 5, 10, 20, 30, 40]

# plt.figure()
# plt.plot(v_orig * 1000, success_rete1, "r", label="1deg")
# plt.plot(v_orig * 1000, success_rete2, "g", label="5deg")
# plt.plot(v_orig * 1000, success_rete3, "b", label="10deg")
# plt.plot(v_orig * 1000, success_rete4, "y", label="20deg")
# plt.plot(v_orig * 1000, success_rete5, "k", label="30deg")
# plt.plot(v_orig * 1000, success_rete5, "k", label="30deg")
# plt.plot(v_orig * 1000, success_rete6, "c", label="40deg")
# plt.xlabel("v[mm/year]")
# plt.ylabel("success rate")
# plt.title("nifg=30,dt=36days")
# plt.legend()
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/V_test_all_1.png")

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


# plt.figure(figsize=(30, 15))
# plt.plot(v_orig * 1000, success_rete1, "r", label="dt=12")
# plt.plot(v_orig * 1000, success_rete2, "g", label="dt=24")
# plt.plot(v_orig * 1000, success_rete3, "b", label="dt=36")
# plt.plot(v_orig * 1000, success_rete4, "y", label="dt=48")
# plt.plot(v_orig * 1000, success_rete5, "k", label="dt=60")
# plt.plot(v_orig * 1000, success_rete6, "c", label="dt=72")
# plt.plot(v_orig * 1000, success_rate7, "m", label="dt=84")
# plt.plot(v_orig * 1000, success_rate8, "orange", label="dt=96")

# plt.xlabel("v[mm/year]", fontsize=20)
# plt.ylabel("success rate", fontsize=20)
# plt.title("noise=10deg,nifg=30", fontsize=20)
# plt.tick_params(labelsize=20)
# plt.legend(fontsize=20)
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/dT_test_all_1.png")

# nifg相关
# v_orig = np.arange(1, 171, 1) * 0.001
# Nifg = [10, 20, 30, 40, 50]
# success_rete1 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Nifgsuccess_lab0_1.csv", delimiter=",")
# success_rete2 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Nifgsuccess_lab1_1.csv", delimiter=",")
# success_rete3 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Nifgsuccess_lab2_1.csv", delimiter=",")
# success_rete4 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Nifgsuccess_lab3_1.csv", delimiter=",")
# success_rete5 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Nifgsuccess_lab4_1.csv", delimiter=",")

# plt.figure()
# plt.plot(v_orig * 1000, success_rete1, "r", label="nifg=10")
# plt.plot(v_orig * 1000, success_rete2, "g", label="nifg=20")
# plt.plot(v_orig * 1000, success_rete3, "b", label="nifg=30")
# plt.plot(v_orig * 1000, success_rete4, "y", label="nifg=40")
# plt.plot(v_orig * 1000, success_rete5, "k", label="nifg=50")

# plt.xlabel("v[mm/year]")
# plt.ylabel("success rate")
# plt.title("noise=10deg,dt=36")
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

# 修改csv文件
success_rate = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dT_Vsuccess_lab80_1.csv", delimiter=" ")
a = success_rate[:, 1]
print(a.shape)
print(success_rate.shape)
success_rate[:, 1] = a / 12
np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dT_Vsuccess_lab_1.csv", success_rate, delimiter=",")
# 加载csv文件的数组
