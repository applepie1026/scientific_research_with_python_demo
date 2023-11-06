import scientific_research_with_python_demo.utils as pf
import numpy as np
import json
import scientific_research_with_python_demo.ambiguity_fuction as af
import scientific_research_with_python_demo.data_plot as dp
import time


with open("template_periodogram/param.json") as f:
    param_file = json.load(f)

T1 = time.perf_counter()
# V的实验
success_rate = af.compute_success_rate(param_file)
# with open("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/afV_test.csv", "a") as f:
#     np.savetxt(f, success_rate, delimiter=",")
#     print("success_rate_save !")
print(success_rate)
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
