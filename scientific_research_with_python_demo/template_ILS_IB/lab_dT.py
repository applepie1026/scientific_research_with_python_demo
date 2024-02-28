import numpy as np
import json
import scientific_research_with_python_demo.ils_estimator_oop as ils
import time
from joblib import Parallel, delayed
import os

with open("scientific_research_with_python_demo/template_ILS_IB/param_ils.json") as f:
    param_file = json.load(f)

file_name = "ils_dT_1_joblib"
main_path = "scientific_research_with_python_demo/data_save/data_ils/" + file_name

param_file["check_times"] = 300
dT = np.arange(1, 101, 1)
# dT = [35, 36]
V_orig = np.round(np.arange(1, 161, 1) * 0.001, 3)


data_all = {}
T1 = time.perf_counter()
for k, T in enumerate(dT):
    print(f"revisit_cycle={T} start")
    data = Parallel(n_jobs=25)(delayed(ils.lab)(param_file, "revisit_cycle", T, v, i) for i, v in enumerate(V_orig))
    data_all[k] = ils.dict_collect(data)
    # data_all[k] = Parallel(n_jobs=25)(delayed(ils.lab)(param_file, "revisit_cycle", T, v, i) for i, v in enumerate(V_orig))
    print(f"revisit_cycle={T} done")
T2 = time.perf_counter()
print(f"Time: {T2 - T1} s")
if not os.path.exists(f"{main_path}"):
    os.makedirs(f"{main_path}")
np.save(f"{main_path}/{file_name}_data.npy", data_all)
success_rate, v_est_data, h_est_data = ils.data_collect(data_all, len(dT), len(V_orig))
np.savetxt(f"{main_path}/{file_name}_success_rate.txt", success_rate, delimiter=",")
np.savetxt(f"{main_path}/{file_name}_v_est_data.txt", v_est_data, delimiter=",")
np.savetxt(f"{main_path}/{file_name}_h_est_data.txt", h_est_data, delimiter=",")
# print(data_all[1][120])
