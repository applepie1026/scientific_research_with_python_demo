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
param_file["param_name"] = ["height", "velocity"]
param_file["Num_search_min"]["height"] = 60
param_file["Num_search_max"]["height"] = 60
V_orig = np.arange(1, 171, 1) * 0.001
# H_orig = [10]


import numpy as np
import json
import ambiguity_fuction as af
import time
from multiprocessing import Process, Manager


with open("param.json") as f:
    param_file = json.load(f)

param_file["noise_level"] = 5
param_file["param_name"] = ["height", "velocity"]
param_file["Num_search_min"]["height"] = 600
param_file["Num_search_max"]["height"] = 600
param_file["step_orig"]["height"] = 0.1
V_orig = np.arange(1, 171, 1) * 0.001
# H_orig = [10]


# 多进程，共享字典
T1 = time.perf_counter()
dT = [35]
dBn = 10
Bn_max = 500
Bn = 333
H_orig = np.array([30])

Nifg = 30
sigma_bn = 5
H_x = 780000  # satellite vertical height[m]
Incidence_angle = 23 * np.pi / 180  # the local incidence angle
R = H_x / np.cos(Incidence_angle)
Lambda = 0.056
process_num_all = 10
check_times = 500
H = 30
data = {}
T1 = time.perf_counter()
if __name__ == "__main__":
    for k in range(3):
        if k == 0:
            param_file["normal_baseline"] = np.random.normal(0, Bn, (1, Nifg))
            print(param_file["normal_baseline"].shape)
            print("normal")
        elif k == 1:
            param_file["normal_baseline"] = (np.arange(1, Nifg + 1, 1) * dBn + np.random.normal(0, sigma_bn, Nifg)).reshape(1, Nifg)
            print(param_file["normal_baseline"].shape)
            print("linear")
        elif k == 2:
            param_file["normal_baseline"] = (-50 * np.linspace(-3.16, 3.16, Nifg) ** 2 + Bn_max + np.random.normal(0, sigma_bn, Nifg)).reshape(1, Nifg)
            print(param_file["normal_baseline"].shape)
            print("parabola")
        with Manager() as manager:
            shared_dict = manager.dict()
            process_list = []
            for i in range(10):
                # print("进程 %s" % i)
                V = V_orig[i * 17 : (i + 1) * 17]
                # print(V)
                p = Process(target=af.param_experiment_v_h, args=("revisit_cycle", [35], V, H, param_file, "afV_H_mutiple_demo", check_times, shared_dict, 1, 1, i))
                p.start()
                process_list.append(p)
            for p in process_list:
                p.join()
            print("All subprocesses done!")
            data[k] = shared_dict.copy()
            print("k=%s" % k)

        # with open("scientific_research_with_python_demo/data_save/afV_H_mutiple_demo.txt", "a") as f:
        #     np.savetxt(f, data_success_rate, delimiter=",")
        #     print("success_rate_save !")
np.save("data_save/data_save1/Bn_vh_data_1.npy", data)
est_data, success_rate = af.est_data_collect_sigma(data, process_num_all=10, test_length=17, data_length=3)
np.savetxt("data_save/data_save1/Bn_vh_est_dat_1.txt", est_data, delimiter=",")
np.savetxt("data_save/data_save1/Bn_vh_suc_1.txt", success_rate, delimiter=",")
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
