import scientific_research_with_python_demo.utils as pf
import numpy as np
import json
import scientific_research_with_python_demo.ambiguity_fuction as af
import scientific_research_with_python_demo.data_plot as dp
import time
from multiprocessing import Process, Manager


with open("template_periodogram/param.json") as f:
    param_file = json.load(f)

param_file["noise_level"] = 30
param_file["param_name"] = ["velocity"]
param_file["Num_search_min"]["height"] = 60
param_file["Num_search_max"]["height"] = 60
V_orig = np.arange(1, 171, 1) * 0.001
# H_orig = [10]


def test(a, return_list):
    return return_list.append(a)


# 多进程，共享字典
T1 = time.perf_counter()
dT = [36]
search_low_bound = np.arange(-300, 1, 5) * 10
print(search_low_bound)
# Nifg = np.array([10, 11, 12])
data = {}
process_num_all = 10
check_times = 1000

T1 = time.perf_counter()
if __name__ == "__main__":
    for k in range(len(search_low_bound)):
        param_file["Num_search_min"]["velocity"] = -search_low_bound[k]
        print(f"search_min_bound={search_low_bound[k]*0.1}mm")
        with Manager() as manager:
            shared_dict = manager.dict()
            process_list = []
            for i in range(10):
                # print("进程 %s" % i)
                V = V_orig[i * 17 : (i + 1) * 17]
                # print(V)
                p = Process(target=af.param_experiment_v_data_sigma, args=("revisit_cycle", [36], V, param_file, "afV_H_mutiple_demo", check_times, shared_dict, 1, i))
                p.start()
                process_list.append(p)
            for p in process_list:
                p.join()
            print("All subprocesses done!")
            data[k] = shared_dict.copy()

est_data, success_rate = af.est_data_collect_sigma(data, process_num_all=10, test_length=17, data_length=len(search_low_bound))
np.savetxt("scientific_research_with_python_demo/data_save/data_save0/bound_est_data_1.txt", est_data, delimiter=",")
np.savetxt("scientific_research_with_python_demo/data_save/data_save0/bound_success_rate_1.txt", success_rate, delimiter=",")
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
