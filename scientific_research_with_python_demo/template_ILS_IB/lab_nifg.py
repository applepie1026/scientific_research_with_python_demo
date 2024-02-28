import numpy as np
import json
import scientific_research_with_python_demo.ils_estimator_oop as ils
import time
from joblib import Parallel, delayed
import os

with open("param_ils.json") as f:
    param_file = json.load(f)

file_name = "ils_Nifg_v_range_1_joblib"
main_path = "data_save/data_ils/" + file_name

param_file["check_times"] = 300
Nifg_orig = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
V_orig = np.round(np.arange(1, 161, 1) * 0.001, 3)


data_all = {}
T1 = time.perf_counter()
for k, Nifg in enumerate(Nifg_orig):
    print(f"Nifg={Nifg} start")
    data_all[k] = Parallel(n_jobs=20)(delayed(ils.lab)(param_file, "Nifg", Nifg, v, i) for i, v in enumerate(V_orig))
    print(f"Nifg={Nifg} done")
T2 = time.perf_counter()
print(f"Time: {T2 - T1} s")
if not os.path.exists(f"{main_path}"):
    os.makedirs(f"{main_path}")
np.save(f"{main_path}/{file_name}_data.npy", data_all)
