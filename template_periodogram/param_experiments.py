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
# v_orig = np.arange(1, 171, 1) * 0.001
# noise_level = [1, 5, 10, 20, 30, 40]

# V的实验
# for i in range(len(noise_level)):
#     success_rate = np.zeros([1, len(v_orig)])
#     param_file["noise_level"] = noise_level[i]
#     print("noise_level = %s deg" % noise_level[i])
#     for j in range(len(v_orig)):
#         param_file["param_simulation"]["velocity"] = v_orig[j]
#         success_rate[0][j] = af.compute_success_rate(param_file)
#     with open("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/afV_test2.csv", "a") as f:
#         np.savetxt(f, success_rate, delimiter=",")
#         print("success_rate_save !")

# dt,V的实验1
# revisit_cycle = np.arange(10, 90.5, 0.5)
# noise_level = [5]
# for i in range(len(revisit_cycle)):
#     success_rate = np.zeros([1, len(v_orig)])
#     param_file["revisit_cycle"] = revisit_cycle[i]
#     print("revisit_cycle = %s days" % revisit_cycle[i])
#     for j in range(len(v_orig)):
#         param_file["param_simulation"]["velocity"] = v_orig[j]
#         success_rate[0][j] = af.compute_success_rate(param_file)
#     with open("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/afV_T_test.csv", "a") as f:
#         np.savetxt(f, success_rate, delimiter=",")
#         print("success_rate_save !")

# dt,V的实验2
revisit_cycle = np.arange(10, 90.5, 0.5)
v_orig = [0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.15]
noise_level = [5]
for i in range(len(revisit_cycle)):
    success_rate = np.zeros([1, len(v_orig)])
    param_file["revisit_cycle"] = revisit_cycle[i]
    print("revisit_cycle = %s days" % revisit_cycle[i])
    for j in range(len(v_orig)):
        param_file["param_simulation"]["velocity"] = v_orig[j]
        success_rate[0][j] = af.compute_success_rate(param_file)
    with open("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/afV_T_test2.csv", "a") as f:
        np.savetxt(f, success_rate, delimiter=",")
        print("success_rate_save !")

T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
