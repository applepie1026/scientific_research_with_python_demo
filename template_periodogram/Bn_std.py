import scientific_research_with_python_demo.utils as pf
import numpy as np
import json
import scientific_research_with_python_demo.ambiguity_fuction as af
import scientific_research_with_python_demo.data_plot as dp
import time

# from multiprocessing import Process, Manager
from joblib import Parallel, delayed

with open("template_periodogram/param.json") as f:
    param_file = json.load(f)

param_file["noise_level"] = 5
param_file["param_name"] = ["height", "velocity"]
param_file["Num_search_min"]["height"] = 600
param_file["Num_search_max"]["height"] = 600
V_orig = np.arange(1, 100, 1) * 0.001
# H_orig = [10]
param_file["param_simulation"]["velocity"] = 0.01
param_file["Nifg"] = 30


def data_collect(data_input, Bn_len):
    data = {}
    success_rate = []
    data_est_all = []
    for j in range(len(data_input)):
        data.update(data_input[j])
    # print(data)
    for i in range(Bn_len):
        data_est_all.append(data[i]["data_est"])
        success_rate.append(data[i]["success_rate"])
    # print(success_rate)
    # print(data_est_all)
    return success_rate, data_est_all


def rmse(data, data_true):
    return np.sqrt(np.mean((data - data_true) ** 2))


def sigma_collect(data_est_h, Bn_len, H_len, H):
    sigma_h_Bn = np.zeros((H_len, Bn_len))
    for j in range(Bn_len):
        for i in range(H_len):
            sigma_h_Bn[i, j] = rmse(data_est_h[j + i * Bn_len, :], H[i])
    return sigma_h_Bn


# 多进程，共享字典
T1 = time.perf_counter()
dT = [35]
Bn = np.arange(20, 1020, 20)
print(Bn, Bn.shape)
# Bn = [333, 100]

check_times = 500
H = [10, 20, 30, 40, 50]
success_rate_list = []
data_est_all_list = []
T1 = time.perf_counter()
for i, h in enumerate(H):
    print(f"height={h} start")
    param_file["param_simulation"]["height"] = h
    data_now = Parallel(n_jobs=15)(delayed(af.param_experiment_h_data_sigma)(param_file, check_times, j, dBn) for j, dBn in enumerate(Bn))
    # print(data_now)
    success_rate, data_est_all = data_collect(data_now, len(Bn))
    success_rate_list.append(success_rate)
    data_est_all_list.append(data_est_all)
    print(f"height={h} done")

# print("success_rate_list", success_rate_list)
# print("data_est_all_list", data_est_all_list)
data_h_all_success_rate = success_rate_list
# print(data_h_all_success_rate.shape)
data_h_all_est = np.concatenate(data_est_all_list, axis=0)
# print("data_h_all_est", data_h_all_est)
np.savetxt("scientific_research_with_python_demo/data_save/data_save1/Bn_std_success_rate.txt", data_h_all_success_rate, delimiter=",")
np.savetxt("scientific_research_with_python_demo/data_save/data_save1/Bn_std_h_est_data.txt", data_h_all_est, delimiter=",")
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
