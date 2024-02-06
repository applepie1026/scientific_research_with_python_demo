import numpy as np
import json
import time
import scientific_research_with_python_demo.vPh_ambiguity_function as af_vPh
import scientific_research_with_python_demo.vPh_ambiguity_demo as af_vPh_demo

with open("template_periodogram/param_vPh.json") as f:
    param_file = json.load(f)

param_file["param_simulation"]["height"] = 1
T1 = time.perf_counter()
# success_rate, v_est, h_est = af_vPh.compute_success_rate(param_file, 200, 1, 5)
success_rate, v_est, h_est = af_vPh_demo.compute_success_rate(param_file, 200)
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
print(success_rate)

# print(np.random.default_rng(1).normal(0, 10, 10))
# print(np.random.default_rng(1).normal(0, 100, 10))
# rng = np.random.default_rng(1)
# print(rng.integers(-5, 5, 10, endpoint=True))
# print(rng.integers(-10, 10, 10, endpoint=True))
