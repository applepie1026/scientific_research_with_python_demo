import scientific_research_with_python_demo.utils as pf
import numpy as np
import json
import scientific_research_with_python_demo.ambiguity_fuction as af
import scientific_research_with_python_demo.data_plot as dp
import time


with open("template_periodogram/param.json") as f:
    param_file = json.load(f)

T1 = time.perf_counter()
# v,h的实验
# V = np.arange(1, 171, 1) * 0.001
V = [0.01]
H = [30]
param_file["noise_level"] = 5
param_file["param_name"] = ["height", "velocity"]
param_file["Num_search_min"]["height"] = 60
param_file["Num_search_max"]["height"] = 60
af.param_experiment_v_h("revisit_cycle", [36], V, H, param_file, "afV_H_test_03", flag=1, multiple=0)
np.savetxt("scientific_research_with_python_demo/data_save/afV_H_test_03.txt", af.data_success_rate, delimiter=",")
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
