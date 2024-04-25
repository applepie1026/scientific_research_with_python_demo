import numpy as np
import json
import time
import scientific_research_with_python_demo.vPh_ambiguity_function as af_vPh
from multiprocessing import Process, Manager
import os

with open("scientific_research_with_python_demo/template_periodogram_vPh/param_vPh.json") as f:
    param_file = json.load(f)

T1 = time.perf_counter()
param_file["step_orig"]["height"] = 0.1
param_file["step_orig"]["velocity"] = 0.0001
param_file["Num_search_min"]["height"] = 600
param_file["Num_search_max"]["height"] = 600
param_file["Num_search_min"]["velocity"] = 1600
param_file["Num_search_max"]["velocity"] = 1600
param_file["flatten_range"] = [-20, 20]
param_file["flatten_num"] = 100
param_file["param_simulation"] = {"height": 15.3, "velocity": 0.0003}
# param_file["normal_baseline"] = np.random.normal(0, 333, param_file["Nifg"])
success_rate, v_est_data, h_est_data = af_vPh.compute_success_rate(param_file, 200, 1, 5)
T2 = time.perf_counter()
print("success_rate:", success_rate)
print("v_est_data_rmse:", np.mean(v_est_data), "v_est_data_accuracy:", af_vPh.accuracy(v_est_data, param_file["param_simulation"]["velocity"]))

print("h_est_data_mean", np.mean(h_est_data), "h_est_data_accuracy:", af_vPh.accuracy(h_est_data, param_file["param_simulation"]["height"]))

print("程序运行时间:%s秒" % (T2 - T1))
