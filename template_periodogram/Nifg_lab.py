import scientific_research_with_python_demo.utils as pf
import numpy as np
import json
import scientific_research_with_python_demo.ambiguity_fuction as af
import scientific_research_with_python_demo.data_plot as dp
import time


with open("template_periodogram/param.json") as f:
    param_file = json.load(f)

# Nifg 的实验
T1 = time.perf_counter()
# h_orig = np.arange(10, 101, 1)
v_orig = np.arange(1, 171, 1) * 0.001

# nifg,V的实验
Nifg = np.arange(10, 100, 1)
noise_level = [10]
for i in range(len(Nifg)):
    success_rate = np.zeros([1, len(v_orig)])
    param_file["Nifg"] = Nifg[i]
    print("Nifg = %s " % Nifg[i])
    for j in range(len(v_orig)):
        param_file["param_simulation"]["velocity"] = v_orig[j]
        success_rate[0][j] = af.compute_success_rate(param_file)
    with open("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/afV_Nifg_test.csv", "a") as f:
        np.savetxt(f, success_rate, delimiter=",")
        print("success_rate_save !")


T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
