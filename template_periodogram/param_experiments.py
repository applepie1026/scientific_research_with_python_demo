import scientific_research_with_python_demo.utils as pf
import numpy as np
import json
import scientific_research_with_python_demo.ambiguity_fuction as af
import scientific_research_with_python_demo.data_plot as dp
import time


with open("template/param2.json") as f:
    param_file = json.load(f)

# param_file["file_name"] = "Nifg_SNR70_10_100test"
param_file["file_name"] = "V_SNR70_0_0x2test"
# print(type(param_file["file_name"]))
# print(param_file["file_name"])
# Nifg 的实验
T1 = time.perf_counter()
# h_orig = np.arange(10, 101, 1)
# v_orig = np.arange(1, 201, 1) * 0.001
success_rate = af.param_experiment("h_orig", np.arange(10, 12, 1), param_file)
# success_rate = af.param_experiment("v_orig", v_orig, param_file)
# success_rate = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_saveV_SNR70_0_0.2testsuccess_rate.csv", delimiter=",")
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
# dp.line_plot(v_orig, success_rate, "SNR=70,dt=12,h=30,nifg=10", param_file["file_name"], "v[m/year]")
# print(succes_rate)
