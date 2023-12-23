import scientific_research_with_python_demo.utils as pf
import numpy as np
import json
import scientific_research_with_python_demo.ambiguity_fuction as af
import scientific_research_with_python_demo.data_plot as dp
import time
from multiprocessing import Process, Manager


with open("template_periodogram/param.json") as f:
    param_file = json.load(f)

param_file["noise_level"] = 5
param_file["param_name"] = ["velocity"]
param_file["Num_search_min"]["height"] = 60
param_file["Num_search_max"]["height"] = 60
V_orig = np.arange(124, 161, 1) * 0.001
# H_orig = [10]


# 多进程，共享字典
dT = [36]
data = {}
process_num_all = 4
check_times = 1000
# v_split = [V_orig[0:12], V_orig[12:24], V_orig[24:37]]
# v_split = [np.array([0.005]), np.array([0.02]), np.array([0.1]), np.array([0.15])]
v_split = [np.array([0.05])]
# print(v_split)
T1 = time.perf_counter()
if __name__ == "__main__":
    with Manager() as manager:
        shared_dict = manager.dict()
        process_list = []
        for i in range(1):
            # print("进程 %s" % i)
            V = v_split[i]
            p = Process(target=af.param_experiment_v_data, args=("revisit_cycle", [65], V, param_file, "V_data1", check_times, shared_dict, 1, i))
            p.start()
            print(V)
            process_list.append(p)
        for p in process_list:
            p.join()
        print("All subprocesses done!")
        data["data_est"] = shared_dict.copy()
        # print(data["data_est"])
        # with open("scientific_research_with_python_demo/data_save/afV_H_mutiple_demo.txt", "a") as f:
        #     np.savetxt(f, data_success_rate, delimiter=",")
        #     print("success_rate_save !")
true_data, est_data = af.est_data_collect(data, process_num_all=1)
np.savetxt("scientific_research_with_python_demo/data_save/V_data3_5true.txt", true_data, delimiter=",")
np.savetxt("scientific_research_with_python_demo/data_save/V_data3_5error.txt", est_data, delimiter=",")
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
