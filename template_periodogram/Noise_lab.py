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
dT = [35]
Noise_level = [5, 10, 20, 30, 40, 50]
data = {}
H = 30
process_num_all = 17
check_times = 1000
T1 = time.perf_counter()
if __name__ == "__main__":
    for k in range(len(Noise_level)):
        param_file["noise_level"] = Noise_level[k]
        print(f"noise_level={Noise_level[k]}deg")
        with Manager() as manager:
            shared_dict = manager.dict()
            process_list = []
            for i in range(17):
                # print("进程 %s" % i)
                V = V_orig[i * 10 : (i + 1) * 10]
                # print(V)
                p = Process(target=af.param_experiment_v_data_sigma, args=("revisit_cycle", [35], V, param_file, "afV_H_mutiple_demo", check_times, shared_dict, 1, i))
                p.start()
                process_list.append(p)
            for p in process_list:
                p.join()
            print("All subprocesses done!")
            data[k] = shared_dict.copy()

        # with open("scientific_research_with_python_demo/data_save/afV_H_mutiple_demo.txt", "a") as f:
        #     np.savetxt(f, data_success_rate, delimiter=",")
        #     print("success_rate_save !")

est_data, success_rate = af.est_data_collect_sigma(data, process_num_all=17, test_length=10, data_length=len(Noise_level))
np.savetxt("scientific_research_with_python_demo/data_save/data_save1/Noise_est_data_1.txt", est_data, delimiter=",")
np.savetxt("scientific_research_with_python_demo/data_save/data_save1/Noise_success_rate_1.txt", success_rate, delimiter=",")
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
