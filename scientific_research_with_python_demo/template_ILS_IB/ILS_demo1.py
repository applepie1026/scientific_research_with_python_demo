import numpy as np
import json
import scientific_research_with_python_demo.ils_estimator_oop as ils
import time
from joblib import Parallel, delayed
import os

with open("scientific_research_with_python_demo/template_ILS_IB/param_ils.json") as f:
    param_file = json.load(f)
# 计算成功率
T1 = time.time()
param_file["check_times"] = 100
param_file["param_simulation"] = {"height": 10, "velocity": 0.16}
param_file["param_pseudo"] = {"height": 8, "velocity": 0.14}

success_rate, est_h, est_v, a = ils.check_success_rate(param_file)
T2 = time.time()
print("程序运行时间:%s秒" % (T2 - T1))
