import scientific_research_with_python_demo.utils as pf
import numpy as np
import json
import scientific_research_with_python_demo.ambiguity_fuction as af
import scientific_research_with_python_demo.data_plot as dp
import time
from multiprocessing import Process, Manager


with open("template_periodogram/param.json") as f:
    param_file = json.load(f)

param_file["noise_level"] = 10
param_file["param_name"] = ["height", "velocity"]
param_file["Num_search_min"]["height"] = 600
param_file["Num_search_max"]["height"] = 600
param_file["step_orig"]["height"] = 0.1
V_orig = np.arange(1, 171, 1) * 0.001
# H_orig = [10]


def test(a, return_list):
    return return_list.append(a)


# 多进程，共享字典
T1 = time.perf_counter()
dT = [36]
H_orig = np.array([5, 10, 20, 30, 40, 50, 60])
data = {}
process_num_all = 10
check_times = 500
T1 = time.perf_counter()
if __name__ == "__main__":
    for k in range(len(H_orig)):
        H = H_orig[k]
        print(f"H={H}m")
        with Manager() as manager:
            shared_dict = manager.dict()
            process_list = []
            for i in range(17):
                # print("进程 %s" % i)
                V = V_orig[i * 10 : (i + 1) * 10]
                # print(V)
                p = Process(target=af.param_experiment_v_h, args=("revisit_cycle", [36], V, H, param_file, "afV_H_mutiple_demo", check_times, shared_dict, 1, 1, i))
                p.start()
                process_list.append(p)
            for p in process_list:
                p.join()
            print("All subprocesses done!")
            data[f"{k}"] = shared_dict.copy()

        # with open("scientific_research_with_python_demo/data_save/afV_H_mutiple_demo.txt", "a") as f:
        #     np.savetxt(f, data_success_rate, delimiter=",")
        #     print("success_rate_save !")

success_rate = af.data_collect(data, len(V_orig), process_num_all=17, test_length=len(H_orig))
np.savetxt("scientific_research_with_python_demo/data_save/data_save0/Bn_normal_vh_demo.txt", success_rate, delimiter=",")
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
