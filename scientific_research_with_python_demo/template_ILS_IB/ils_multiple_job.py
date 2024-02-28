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


def test(param_file, Nifg, V, data_id):
    # print(f"v={V}start")
    data = {}
    param_file["Nifg"] = Nifg
    param_file["velocity"] = V
    est_data_h = np.zeros(param_file["check_times"])
    est_data_v = np.zeros(param_file["check_times"])
    a_data_1st = np.zeros((param_file["check_times"], param_file["Nifg"]))
    success_time = 0
    for i in range(300):
        ils_est = ils.ILS_IB_estimator(param_file, data_id, i)
        a, x1, x2 = ils_est.ils_estimation()
        est_data_h[i] = x1[0]
        est_data_v[i] = x1[1]
        a_data_1st[i, :] = a[:, 0]
    if abs((x1[0] - param_file["param_simulation"]["height"]) < 0.5 and abs(x1[1] - param_file["param_simulation"]["velocity"]) < 0.0005) or abs(
        (x2[0] - param_file["param_simulation"]["height"]) < 0.5 and abs(x2[1] - param_file["param_simulation"]["velocity"]) < 0.0005
    ):
        success_time += 1
    success_rate = success_time / 300
    data[data_id] = {"success_rate": success_rate, "est_data_h": est_data_h, "est_data_v": est_data_v, "a_data_1st": a_data_1st}
    print(f"v={V} done")
    return data


data_all = {}
T1 = time.perf_counter()
for k, Nifg in enumerate(Nifg_orig):
    print(f"Nifg={Nifg} start")
    data_all[k] = Parallel(n_jobs=20)(delayed(test)(param_file, Nifg, v, i) for i, v in enumerate(V_orig))
    print(f"Nifg={Nifg} done")
T2 = time.perf_counter()
print(f"Time: {T2 - T1} s")
if not os.path.exists(f"{main_path}"):
    os.makedirs(f"{main_path}")
np.save(f"{main_path}/{file_name}_data.npy", data_all)
