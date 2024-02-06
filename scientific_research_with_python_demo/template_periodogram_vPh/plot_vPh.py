import numpy as np
import matplotlib.pyplot as plt
import scientific_research_with_python_demo.vPh_ambiguity_function as af_vPh

# vPh demo
# success_rate = np.loadtxt("scientific_research_with_python_demo/data_save/DFT/vPh_v_range_9.txt", delimiter=",")
# success_rate = np.loadtxt("scientific_research_with_python_demo/data_save/DFT/vPh_v_range_1/vPh_v_range_1.txt", delimiter=",")
# V_orig = np.arange(1, 171, 1) * 0.001

# plt.figure(figsize=(10, 7))
# ax = plt.gca()
# ax.spines["right"].set_color("none")
# ax.spines["top"].set_color("none")
# ax.xaxis.set_ticks_position("bottom")
# ax.spines["bottom"].set_position(("data", 0))
# plt.plot(V_orig, success_rate)
# plt.xlabel("Linear Displacement Rate $\Delta{v}$ [mm/yr]", fontsize=15)
# plt.ylabel("Success Rate", fontsize=15)
# plt.title("Nifg=30,$\Delta{h}$=10[m],$\Delta{T}$=35days,Noise Level=5deg,flatten_num=100", fontsize=20)
# plt.savefig("scientific_research_with_python_demo/plot/DFT/vPh_v_range_1.png")

# the flattening number
# success_rate1 = np.loadtxt("scientific_research_with_python_demo/data_save/DFT/vPh_v_range_2/vPh_v_range_2.txt", delimiter=",")
# success_rate2 = np.loadtxt("scientific_research_with_python_demo/data_save/DFT/vPh_v_range_3/vPh_v_range_3.txt", delimiter=",")
# success_rate3 = np.loadtxt("scientific_research_with_python_demo/data_save/DFT/vPh_v_range_1/vPh_v_range_1.txt", delimiter=",")
# success_rate4 = np.loadtxt("scientific_research_with_python_demo/data_save/DFT/vPh_v_range_4/vPh_v_range_4.txt", delimiter=",")
# success_rate5 = np.loadtxt("scientific_research_with_python_demo/data_save/DFT/vPh_v_range_5/vPh_v_range_5.txt", delimiter=",")
# V_orig = np.arange(1, 171, 1) * 0.001
# plt.figure(figsize=(10, 7))
# ax = plt.gca()
# ax.spines["right"].set_color("none")
# ax.spines["top"].set_color("none")
# ax.xaxis.set_ticks_position("bottom")
# ax.spines["bottom"].set_position(("data", 0))
# plt.plot(V_orig, success_rate1, label="flatten_num=20")
# plt.plot(V_orig, success_rate2, label="flatten_num=50")
# plt.plot(V_orig, success_rate3, label="flatten_num=100")
# plt.plot(V_orig, success_rate4, label="flatten_num=150")
# plt.plot(V_orig, success_rate5, label="flatten_num=200")
# plt.xlabel("Linear Displacement Rate $\Delta{v}$ [mm/yr]", fontsize=15)
# plt.ylabel("Success Rate", fontsize=15)
# plt.title("Nifg=30,$\Delta{h}$=10[m],$\Delta{T}$=35days,Noise Level=5deg,flatten_h$\in$[-60m,60m]", fontsize=15)
# plt.legend()
# plt.savefig("scientific_research_with_python_demo/plot/DFT/flatten_num.png")

# flatten range
# success_rate1 = np.loadtxt("scientific_research_with_python_demo/data_save/DFT/vPh_v_range_7/vPh_v_range_7.txt", delimiter=",")
# success_rate2 = np.loadtxt("scientific_research_with_python_demo/data_save/DFT/vPh_v_range_1/vPh_v_range_1.txt", delimiter=",")
# V_orig = np.arange(1, 171, 1) * 0.001

# plt.figure(figsize=(10, 7))
# ax = plt.gca()
# ax.spines["right"].set_color("none")
# ax.spines["top"].set_color("none")
# ax.xaxis.set_ticks_position("bottom")
# ax.spines["bottom"].set_position(("data", 0))
# plt.plot(V_orig, success_rate1, label="flatten_h$\in$[-30m,30m]")
# plt.plot(V_orig, success_rate2, label="flatten_h$\in$[-60m,60m]")
# plt.xlabel("Linear Displacement Rate $\Delta{v}$ [mm/yr]", fontsize=15)
# plt.ylabel("Success Rate", fontsize=15)
# plt.title("Nifg=30,$\Delta{h}$=10[m],$\Delta{T}$=35days,Noise Level=5deg,flatten_num=100", fontsize=20)
# plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1))
# plt.savefig("scientific_research_with_python_demo/plot/DFT/flatten_range.png")

# h_range
success_rate1 = np.loadtxt("scientific_research_with_python_demo/data_save/DFT/vPh_h_range_4/vPh_h_range_4.txt", delimiter=",")
success_rate2 = np.loadtxt("scientific_research_with_python_demo/data_save/DFT/vPh_h_range_3/vPh_h_range_3.txt", delimiter=",")
success_rate3 = np.loadtxt("scientific_research_with_python_demo/data_save/DFT/vPh_h_range_2/vPh_h_range_2.txt", delimiter=",")
success_rate4 = np.loadtxt("scientific_research_with_python_demo/data_save/DFT/vPh_h_range_5/vPh_h_range_5.txt", delimiter=",")
success_rate5 = np.loadtxt("scientific_research_with_python_demo/data_save/DFT/vPh_h_range_6/vPh_h_range_6.txt", delimiter=",")
success_rate6 = np.loadtxt("scientific_research_with_python_demo/data_save/DFT/vPh_h_range_1/vPh_h_range_1.txt", delimiter=",")
H_orig = np.arange(1, 61, 1)
plt.figure(figsize=(10, 7))
ax = plt.gca()
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")
ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_position(("data", 0))
plt.plot(H_orig, success_rate1, label="flatten_h$\in$[-10m,10m]")
plt.plot(H_orig, success_rate2, label="flatten_h$\in$[-20m,20m]")
plt.plot(H_orig, success_rate3, label="flatten_h$\in$[-30m,30m]")
plt.plot(H_orig, success_rate4, label="flatten_h$\in$[-40m,40m]")
plt.plot(H_orig, success_rate5, label="flatten_h$\in$[-50m,50m]")
plt.plot(H_orig, success_rate6, label="flatten_h$\in$[-60m,60m]")
plt.xlabel("Residual Height [m]", fontsize=15)
plt.ylabel("Success Rate", fontsize=15)
plt.title("Nifg=30,$\Delta{v}$=10mm/yr,$\Delta{T}$=35days,Noise Level=5deg,flatten_num=100", fontsize=15)
plt.legend(loc="upper right", bbox_to_anchor=(1.13, 1), fontsize=10)
plt.savefig("scientific_research_with_python_demo/plot/DFT/h_range.png")


# H_orig = [10]
# V_orig = np.arange(1, 171, 1) * 0.001
# data = np.load("scientific_research_with_python_demo/data_save/DFT/vPh_h_range_2/vPh_h_range_2.npy", allow_pickle=True).item()
# # print(data)
# success_rate, v_est_data, h_est_data = af_vPh.data_collect(data, [0.01], np.arange(1, 61, 1), 20, 3)
# # np.savetxt("scientific_research_with_python_demo/data_save/DFT/vPh_h_range_1/vPh_h_range_1.txt", success_rate, delimiter=",")
# np.savetxt("scientific_research_with_python_demo/data_save/DFT/vPh_h_range_2/vPh_h_range_2_v_est_data.txt", v_est_data)
# np.savetxt("scientific_research_with_python_demo/data_save/DFT/vPh_h_range_2/vPh_h_range_2_h_est_data.txt", h_est_data)

# data = np.load("scientific_research_with_python_demo/data_save/DFT/vPh_v_range_5/vPh_v_range_5.npy", allow_pickle=True).item()
# success_rate, v_est_data, h_est_data = af_vPh.data_collect(data, [10], np.arange(1, 171, 1), 17, 10)
# # np.savetxt("scientific_research_with_python_demo/data_save/DFT/vPh_v_range_1/vPh_v_range_1.txt", success_rate, delimiter=",")
# np.savetxt("scientific_research_with_python_demo/data_save/DFT/vPh_v_range_5/vPh_v_range_5_v_est_data.txt", v_est_data)
# np.savetxt("scientific_research_with_python_demo/data_save/DFT/vPh_v_range_5/vPh_v_range_5_h_est_data.txt", h_est_data)
