import numpy as np
import json
import scientific_research_with_python_demo.ils_estimator_oop as ils
import time
from multiprocessing import Process, Manager
import scientific_research_with_python_demo.ambiguity_fuction as af

with open("scientific_research_with_python_demo/template_ILS_IB/param_ils.json") as f:
    param_file = json.load(f)

param_file["noise_level"] = 5
param_file["param_name"] = ["height", "velocity"]
V_orig = np.arange(1, 171, 1) * 0.001
Bn = 333
Nifg = 30
param_file["normal_baseline"] = np.random.normal(0, Bn, param_file["Nifg"])
revisit_cycle = [35]
data = {}
T1 = time.perf_counter()
if __name__ == "__main__":
    for k in range(len(revisit_cycle)):
        param_file["revisit_cycle"] = revisit_cycle[k]
        with Manager() as manager:
            shared_dict = manager.dict()
            process_list = []
            for i in range(10):
                # print("进程 %s" % i)
                V = V_orig[i * 17 : (i + 1) * 17]
                # print(V)
                p = Process(target=ils.main, args=(param_file, V, shared_dict, i))
                p.start()
                process_list.append(p)
            for p in process_list:
                p.join()
            data[k] = shared_dict.copy()
            print("All subprocesses done!")

# np.save("data_save/data_ils/Bn_vh_data_1.npy", data)
# est_data, success_rate = af.est_data_collect_sigma(data, process_num_all=10, test_length=17, data_length=10)
# np.savetxt("data_save/data_ils/Bn_vh_est_dat_1.txt", est_data, delimiter=",")
# np.savetxt("data_save/data_ils/Bn_vh_suc_1.txt", success_rate, delimiter=",")
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
