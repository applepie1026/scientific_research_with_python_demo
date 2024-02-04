import numpy as np
import matplotlib.pyplot as plt

# vPh demo
success_rate = np.loadtxt("scientific_research_with_python_demo/data_save/DFT/vPh_v_range_7.txt", delimiter=",")
V_orig = np.arange(1, 171, 1) * 0.001

plt.figure(figsize=(10, 7))
ax = plt.gca()
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")
ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_position(("data", 0))
plt.plot(V_orig, success_rate)
plt.xlabel("Linear Displacement Rate $\Delta{v}$ [mm/yr]", fontsize=15)
plt.ylabel("Success Rate", fontsize=15)
plt.title("Nifg=30,$\Delta{h}$=10[m],$\Delta{T}$=36days,Noise Level=5deg,flatten_num=200", fontsize=20)
plt.savefig("scientific_research_with_python_demo/plot/DFT/vPh_v_range_7.png")
# data = np.load("scientific_research_with_python_demo/data_save/DFT/vPh_v_range_test.npy", allow_pickle=True).item()
# print(list(data[0][0][i]["success_rate"] for i in range(10)))
# print(list(data[0][1][i + 10]["success_rate"] for i in range(10)))
# print(list(data[0][2][i + 20]["success_rate"] for i in range(10)))
# print(list(data[0][3][i + 30]["success_rate"] for i in range(10)))
