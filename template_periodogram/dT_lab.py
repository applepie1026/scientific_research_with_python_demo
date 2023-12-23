import scientific_research_with_python_demo.utils as pf
import numpy as np
import json
import scientific_research_with_python_demo.ambiguity_fuction as af
import scientific_research_with_python_demo.data_plot as dp
import time
from multiprocessing import Process, Manager

# Nifg 的实验
with open("template_periodogram/param.json") as f:
    param_file = json.load(f)

param_file["noise_level"] = 5
param_file["param_name"] = ["velocity"]
param_file["Num_search_min"]["height"] = 60
param_file["Num_search_max"]["height"] = 60
V_orig = np.arange(1, 171, 1) * 0.001

# 多进程，共享字典
T1 = time.perf_counter()
dT = [12, 24, 36, 48, 60, 72]
data = {}
H = 30
process_num_all = 10
check_times = 1000
T1 = time.perf_counter()
if __name__ == "__main__":
    for k in range(len(dT)):
        # param_file["revisit_cycle"] = dT[k]
        print(f"revisit_cycle={dT[k]}days")
        with Manager() as manager:
            shared_dict = manager.dict()
            process_list = []
            for i in range(10):
                # print("进程 %s" % i)
                V = V_orig[i * 17 : (i + 1) * 17]
                # print(V)
                p = Process(target=af.param_experiment_v, args=("revisit_cycle", [dT[k]], V, param_file, "afV_H_mutiple_demo", check_times, shared_dict, 1, 1, i))
                p.start()
                process_list.append(p)
            for p in process_list:
                p.join()
            print("All subprocesses done!")
            data[f"{k}"] = shared_dict.copy()

        # with open("scientific_research_with_python_demo/data_save/afV_H_mutiple_demo.txt", "a") as f:
        #     np.savetxt(f, data_success_rate, delimiter=",")
        #     print("success_rate_save !")

success_rate = af.data_collect(data, len(V_orig), process_num_all=10, test_length=len(dT))
np.savetxt("scientific_research_with_python_demo/data_save/afV_H_mutiple_demo_v_dT1.txt", success_rate, delimiter=",")
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
