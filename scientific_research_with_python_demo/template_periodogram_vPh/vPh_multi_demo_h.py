import numpy as np
import json
import time
import scientific_research_with_python_demo.vPh_ambiguity_function as af_vPh
from multiprocessing import Process, Manager
import os

with open("template_periodogram/param_vPh.json") as f:
    param_file = json.load(f)

file_name = "vPh_h_range_6"
main_path = "scientific_research_with_python_demo/data_save/DFT/" + file_name + "/"
V_orig = [0.01]
H_orig = np.arange(1, 61, 1)
param_file["noise_level"] = 5
param_file["revisit_cycle"] = 35
param_file["flatten_num"] = 100
param_file["flatten_range"] = 50
check_times = 500
data = {}
T1 = time.perf_counter()
if __name__ == "__main__":
    for k in range(len(V_orig)):
        param_file["param_simulation"]["velocity"] = V_orig[k]
        with Manager() as manager:
            shared_dict = manager.dict()
            process_list = []
            for i in range(20):
                H = H_orig[i * 3 : (i + 1) * 3]
                p = Process(target=af_vPh.compute_success_rate_multi_h, args=(param_file, check_times, H, i, len(H_orig), shared_dict))
                p.start()
                process_list.append(p)
            for p in process_list:
                p.join()
            print("All subprocesses done!")
            data[k] = shared_dict.copy()
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
# 创建文件夹
if not os.path.exists(f"{main_path}"):
    os.makedirs(f"{main_path}")
np.save(f"{main_path}/{file_name}.npy", data)
success_rate, v_est_data, h_est_data = af_vPh.data_collect(data, V_orig, H_orig, process_num_all=20, test_length=3)
np.savetxt(f"{main_path}/{file_name}.txt", success_rate, delimiter=",")
np.savetxt(f"{main_path}/{file_name}_v_est_data.txt", v_est_data)
np.savetxt(f"{main_path}/{file_name}_h_est_data.txt", h_est_data)
