import numpy as np
import json
import time
import scientific_research_with_python_demo.vPh_ambiguity_function as af_vPh


# with open("template_periodogram/param_vPh.json") as f:
#     param_file = json.load(f)

# T1 = time.perf_counter()
# success_rate = af_vPh.compute_success_rate(param_file, 1000)
# T2 = time.perf_counter()
# print("程序运行时间:%s秒" % (T2 - T1))
# print(success_rate)

# print(np.random.default_rng(1).normal(0, 10, 10))
# print(np.random.default_rng(1).normal(0, 100, 10))
rng = np.random.default_rng()
print(rng.integers(-5, 5, 10, endpoint=True))
print(rng.integers(-5, 5, 20, endpoint=True))
