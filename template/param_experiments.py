import scientific_research_with_python_demo.utils as pf
import numpy as np
import json
import scientific_research_with_python_demo.ambiguity_fuction as af
import time


with open('template/param2.json') as f:
    param_file = json.load(f)


# print(type(param_file["file_name"]))
# print(param_file["file_name"])
# Nifg 的实验
T1 = time.perf_counter()
succes_rate=af.param_experiment("Nifg",np.arange(10,12,1), param_file)

T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
# print(succes_rate)
