import scientific_research_with_python_demo.ILS_estimator as ils
import numpy as np
import time
import json

# import ambiguity_fuction as af
from multiprocessing import Process, Manager

# 计算成功率
sig = np.sqrt(2 * (np.pi * 40 / 180) ** 2)
# v = 0.016
V_orig = np.round(np.arange(1, 171, 1) * 0.001, 3)
print(V_orig)
h = 20
Nifg_orig = [30]
noise_level = 10
dT = 36
Bn = 333
h_bound = 30
v_bound = 10
check_times = 1000
data = {}
data_est = {}
T1 = time.time()
if __name__ == "__main__":
    for k in range(len(Nifg_orig)):
        Nifg = Nifg_orig[k]
        print(f"Nifg={Nifg}")
        # print(f"H={H}m")
        with Manager() as manager:
            shared_dict = manager.dict()
            est_dict = manager.dict()
            process_list = []
            for i in range(10):
                # print("进程 %s" % i)
                V = V_orig[i * 17 : (i + 1) * 17]
                # V = V_orig[0 : 17]
                print(V)
                p = Process(target=ils.check_success_rate1, args=(V, h, Nifg, noise_level, dT, Bn, h_bound, v_bound, sig, shared_dict, est_dict, check_times, 1, i))
                # p = Process(target=ils.check_success_rate2, args=(V, h, Nifg, noise_level, dT, Bn, h_bound, v_bound, sig, shared_dict, check_times, 1, i))
                p.start()
                process_list.append(p)
            for p in process_list:
                p.join()
            print("All subprocesses done!")
            data[f"{k}"] = shared_dict.copy()
            data_est[f"{k}"] = est_dict.copy()

        # with open("scientific_research_with_python_demo/data_save/afV_H_mutiple_demo.txt", "a") as f:
        #     np.savetxt(f, data_success_rate, delimiter=",")
        #     print("success_rate_save !")

# success_rate = af.data_collect(data, len(V_orig), process_num_all=10, test_length=len(Nifg_orig))
# np.savetxt("data_save/ils_demo1.txt", success_rate)
# print("success_rate save done")
# # success_rate = af.data_collect(data, 20, process_num_all=10, test_length=len(Nifg_orig))
# est_data = af.est_data_collect(data_est, check_times, len(V_orig), process_num_all=10, test_length=len(Nifg_orig))
# np.savetxt("data_save/ils_demo1_est.txt", est_data)
# print("est_data save done")
T2 = time.time()
print("程序运行时间:%s秒" % (T2 - T1))
# print(success_rate)
# param = np.array([[1], [2]])
# print(param)
# print(param.shape)
