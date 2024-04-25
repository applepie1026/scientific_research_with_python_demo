import numpy as np
import json
import time
import scientific_research_with_python_demo.vPh_ambiguity_function as af_vPh
from joblib import Parallel, delayed


def check_data(V_len, data, check_times):
    data_dict = {}
    for k in data:
        data_dict.update(k)
    success_rate = np.zeros(V_len)
    data_est = np.zeros((V_len, 2 * check_times))
    for i in range(V_len):
        success_rate[i] = data_dict[i]["success_rate"]
        data_est[i, :] = data_dict[i]["est_data"]
    return success_rate, data_est


with open("scientific_research_with_python_demo/template_periodogram_vPh/param_vPh.json") as f:
    param_file = json.load(f)
param_file["step_orig"]["height"] = 0.1
param_file["step_orig"]["velocity"] = 0.0001
param_file["Num_search_min"]["height"] = 600
param_file["Num_search_max"]["height"] = 600
param_file["Num_search_min"]["velocity"] = 1600
param_file["Num_search_max"]["velocity"] = 1600
param_file["flatten_range"] = [-60, 60]
param_file["flatten_num"] = 100

H_orig = np.arange(-600, 601, 1) * 0.1
V_orig = np.round(np.arange(1, 171, 1) * 0.001, 4)
# H_orig = [10, 11]
# V_orig = [0.01, 0.02]
check_times = 10
success_rate_list = []
data_est_all_list = []
# print(H_orig)
# print(V_orig)
# for v in V_orig:
for h in H_orig:
    param_file["param_simulation"]["height"] = h
    data = Parallel(n_jobs=15)(delayed(af_vPh.compute_est_data)(param_file, check_times, i, v) for i, v in enumerate(V_orig))
    # print(data)
    success_rate, data_est = check_data(len(V_orig), data, check_times)
    success_rate_list.append(success_rate)
    data_est_all_list.append(data_est)
    # print(success_rate, type(success_rate))
    # print(data_est, type(data_est))
data_suc = np.concatenate(success_rate_list, axis=0).reshape(len(H_orig), len(V_orig))
data_est = np.concatenate(data_est_all_list, axis=0)
np.savetxt("scientific_research_with_python_demo/data_save/data_vPh/vh_data.txt", data_est, delimiter=",")
np.savetxt("scientific_research_with_python_demo/data_save/data_vPh/vh_data_suc.txt", data_suc, delimiter=",")
print("data_done")
# print(data_suc)
# data_all = {}
# for k in data:
#     data_all.update(k)

# print(data_all)
