import numpy as np
import scientific_research_with_python_demo.data_plot as dp
import matplotlib.pyplot as plt
import os


def check_flex_points(success_rate_all, dv_range, dt_range):
    flex_points = np.zeros(len(dt_range))
    for j in range(len(dt_range)):
        success_rate = success_rate_all[j, :]
        for i in range(len(dv_range) - 1):
            if success_rate[i] - success_rate[i + 1] > 0.1:
                flex_points[j] = dv_range[i + 1]
                break
    return flex_points


dt_range = np.arange(32, 63 + 1, 1)

rate_all = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dT_V10deg_success_lab_1.csv", delimiter=",")
rate_all_stair = rate_all[(32 - 10) * 170 : (63 - 10 + 1) * 170, :]
print(rate_all_stair.shape)
success_rate_all = rate_all_stair[:, 2].reshape(32, 170)
print(success_rate_all.shape)
np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/flex_point.csv", success_rate_all, delimiter=",")
flex_point_all = check_flex_points(success_rate_all, np.arange(1, 171, 1), dt_range)
print(flex_point_all)
dp.line_plot_flex_point(dt_range, flex_point_all, "noise=10deg,nifg=30", "flex_point", "dt[days]")
