import scientific_research_with_python_demo.utils as af
from scientific_research_with_python_demo.periodogram_test import periodogram_lab, periodogram_lab2
import scientific_research_with_python_demo.data_plot as dp
import numpy as np
import time

# Nifg 的实验
T1 = time.perf_counter()

# ------------------------------------------------
# initial parameters
# ------------------------------------------------
WAVELENGTH = 0.056  # [unit:m]
Nifg = 30
# h_orig = 30
v_orig = 0.05
# noise_level = [1, 5, 10, 20, 30, 40]
noise_level = 10
# bn_sigma = np.arange(100, 1001, 10)
bn_sigma = [333]
# noise_phase = af.sim_phase_noise(noise_level, Nifg)
step_orig = np.array([1.0, 0.0001])
# std_param = np.array([40, 0.06])
param_orig = np.array([0, 0])
param_name = ["height", "velocity"]
set_param = [30, 0.05]
test_param_name = {"test_param": "height", "hold_param": "velocity"}
success_rate = np.zeros((91 * 125, 3))
r = 0
for j in range(len(bn_sigma)):
    bn = bn_sigma[j]
    print("bn = ", bn_sigma[j])
    # h_orig = np.arange(1, 126, 1)
    h_orig = [30]
    std_param = np.array([40, 0.08])
    # calculate the number of search
    Num_search1_max = 120
    Num_search1_min = 120
    Num_search2_max = 1600
    Num_search2_min = 1600
    Num_search = np.array([[Num_search1_max, Num_search1_min], [Num_search2_max, Num_search2_min]])
    time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg)
    v2ph = af.v_coef2(time_baseline).T
    therehold_v = af.param_threshold(v2ph, phase_threshold=np.pi * 0.03 / 180)
    for i in range(len(h_orig)):
        h = h_orig[i]
        # print("v = ", v)
        iteration = 0
        success = 0
        # est = np.zeros((1000, 2))
        while iteration < 1000:
            # simulate baseline
            normal_baseline = np.random.normal(0, bn, size=(1, Nifg))
            # print(normal_baseline)
            # calculate the input parameters of phase
            h2ph = af.h_coef(normal_baseline).T
            # print(h2ph)
            par2ph = [h2ph, v2ph]
            # phase_obsearvation simulate
            phase_obs, snr, phase_true, h_phase, v_phase = af.sim_arc_phase_lab4(v_orig, h, v2ph, h2ph, noise_level)
            # normalize the intput parameters
            data_set = af.input_parameters(par2ph, step_orig, Num_search, param_orig, param_name, test_param_name, set_param)
            # ------------------------------------------------
            # main loop of searching
            # ------------------------------------------------
            count = 0
            est_param = {}
            while count <= 10 and data_set["velocity"]["step_orig"] > 1.0e-8 and data_set["height"]["step_orig"] > 1.0e-4:
                # search the parameters
                est_param, best = periodogram_lab2(data_set, phase_obs)
                # update the parameters
                for key in param_name:
                    data_set[key]["param_orig"] = est_param[key]
                    # update the step
                    data_set[key]["step_orig"] *= 0.1
                    # update the number of search
                    data_set[key]["Num_search_max"] = 10
                    data_set[key]["Num_search_min"] = 10

                count += 1
            if abs(est_param["height"] - h) < 0.05:
                success += 1
                print(est_param)
            # est[iteration, 0] = est_param["height"]
            # est[iteration, 1] = est_param["velocity"]
            iteration += 1
            # else:
            # print(est_param)
        # with open("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/H_test_%s_1.csv" % j, "a") as f:
        #     # 按列追加保存
        #     np.savetxt(f, est, delimiter=",")
        # success rate
        print(success / iteration)
        success_rate[r, 0] = h_orig[i]
        success_rate[r, 1] = bn_sigma[j]
        success_rate[r, 2] = success / iteration
        r += 1


# np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Bn_H_10deg_success_lab_1.csv", success_rate, delimiter=",")
print("success_rate_save !")
# dp.line_plot(v_orig * 1000, success_rate, "n=10deg,dt=%s,nifg=30" % (dt_range[j] * 12), "dT_lab%d_1" % j, "v[mm/year]")
# print("data_plot_save !")
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
