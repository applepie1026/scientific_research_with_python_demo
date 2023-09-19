import scientific_research_with_python_demo.utils as af
from scientific_research_with_python_demo.periodogram_test import periodogram_lab, periodogram_lab2
from scientific_research_with_python_demo.periodogram_main import periodogram
import scientific_research_with_python_demo.data_plot as dp
import numpy as np
import time

# v,h双重搜索
T1 = time.perf_counter()

# ------------------------------------------------
# initial parameters
# ------------------------------------------------
WAVELENGTH = 0.056  # [unit:m]
Nifg = 30
h_orig = 30  # [m]，整数 30 循环迭代搜索结果有问题
noise_level = 10
# noise_phase = af.sim_phase_noise(noise_level, Nifg)
step_orig = np.array([1.0, 0.0001])
# std_param = np.array([40, 0.06])
param_orig = np.array([0, 0])
param_name = ["height", "velocity"]
set_param = [30, 0.05]
test_param_name = {"test_param": "velocity", "hold_param": "height"}
v_orig = np.arange(1, 171, 1) * 0.001
h = h_orig
# # calculate the number of search
# Num_search1 = af.compute_Nsearch(std_param[0], step_orig[0])
# Num_search2 = af.compute_Nsearch(std_param[1], step_orig[1])
# Num_search = np.array([Num_search1, Num_search2])
std_param = np.array([40, 0.08])
# calculate the number of search
Num_search1_max = 120
Num_search1_min = 120
Num_search2_max = 1600
Num_search2_min = 1600
Num_search = np.array([[Num_search1_max, Num_search1_min], [Num_search2_max, Num_search2_min]])
success_rate = np.zeros(len(v_orig))
time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg) * 3
v2ph = af.v_coef(time_baseline).T
therehold_v = af.param_threshold(v2ph, phase_threshold=np.pi * 0.1 / 180)
for i in range(len(v_orig)):
    v = v_orig[i]
    print("v = ", v)
    iteration = 0
    success = 0
    est = np.zeros((1000, 2))
    while iteration < 1000:
        # simulate baseline
        normal_baseline = np.random.normal(size=(1, Nifg)) * 333
        # print(normal_baseline)
        # calculate the input parameters of phase
        h2ph = af.h_coef(normal_baseline).T
        # print(h2ph)
        par2ph = [h2ph, v2ph]
        # phase_obsearvation simulate
        phase_obs, snr, phase_true, h_phase, v_phase = af.sim_arc_phase_lab(v, h_orig, v2ph, h2ph, noise_level)
        # normalize the intput parameters
        data_set = af.input_parameters(par2ph, step_orig, Num_search, param_orig, param_name, test_param_name, set_param)
        # print(data_set)
        # print(data_set["velocity"]["Num_search"])
        # ------------------------------------------------
        # main loop of searching
        # ------------------------------------------------
        count = 0
        est_param = {}
        while count <= 10 and data_set["velocity"]["step_orig"] > 1.0e-8 and data_set["height"]["step_orig"] > 1.0e-4:
            # search the parameters
            est_param, best = periodogram(data_set, phase_obs)
            # update the parameters
            for key in param_name:
                data_set[key]["param_orig"] = est_param[key]
                # update the step
                data_set[key]["step_orig"] *= 0.1
                # update the number of search
                data_set[key]["Num_search_max"] = 10
                data_set[key]["Num_search_min"] = 10

            count += 1
        if abs(est_param["velocity"] - v) < 0.0005 and abs(est_param["height"] - h_orig) < 0.5:
            success += 1
        est[iteration, 0] = est_param["height"]
        est[iteration, 1] = est_param["velocity"]
        iteration += 1
        # else:
        # print(est_param)
    with open("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/V_test_deg_1.csv", "a") as f:
        # 按列追加保存
        np.savetxt(f, est, delimiter=",")
    # success rate
    # print(success / iteration)
    print(success / iteration)
    success_rate[i] = success / iteration
np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_test_deg_1.csv", success_rate)
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
# dp.bar_plot(v_orig * 1000, success_rate, "Nifg=10,SNR=70db,dt=12", "snr_v_test5", 0.001 * 1000, "v[mm/year]")
dp.line_plot(v_orig * 1000, success_rate, "noise=10deg,dt=36,nifg=30", "V_test_deg_1", "v[mm/year]")
