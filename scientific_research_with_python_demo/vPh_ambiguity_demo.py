import numpy as np
from copy import deepcopy
import json

# Constant
WAVELENGTH = 0.056  # [unit:m]
H = 780000  # satellite vertical height[m]
Incidence_angle = 23 * np.pi / 180  # the local incidence angle
R = H / np.cos(Incidence_angle)  # range to the master antenna. test
m2ph = 4 * np.pi / WAVELENGTH


# class DFT_periodogram:
#     def __init__(self, param_file):
#         param = deepcopy(param_file)
#         self.Nifg = param["Nifg"]
#         self.param_sim = param["param_simulation"]
#         self.noise_level = param["noise_level"]
#         self.step_orig = param["step_orig"]
#         self.param_name = param["param_name"]
#         self.Num_search_min = param["Num_search_min"]
#         self.Num_search_max = param["Num_search_max"]
#         self.revisit_cycle = param["revisit_cycle"]
#         self.Bn = param["Bn"]
#         # self.rng_seed = data_id
#         self.normal_baseline = np.random.normal(0, param["Bn"], param["Nifg"])
#         # print(f"{check_num},{data_id}:{self.normal_baseline}")
#         self.flatten_num = param["flatten_num"]


class dtft_af_all_array2:
    def __init__(self, wrapped_phase, p2ph):
        self.yvalues = []
        self.wrapped_phase = wrapped_phase
        self.p2ph = p2ph

    # fre:频率坐标
    def xjw(self, fre=[], N=30):
        # （式1-1）实现yvalues为X（jw）频谱值
        searched_phase = self.p2ph.reshape(N, 1) * fre
        # print(searched_phase.shape)
        coh_phase = self.wrapped_phase.reshape(N, 1) * np.ones((1, len(fre))) - searched_phase
        # print(coh_phase.shape)
        xjw_dft = np.sum(np.exp(1j * coh_phase), axis=0) / N
        coh = abs(xjw_dft)
        # print(xjw_dft.shape, coh.shape)
        # coh_phase = self.wrapped_phase.reshape(N, 1) - searched_phase
        # for i in self.yvalues:
        #     xjw_dft = np.append(xjw_dft, abs(i))
        return coh


def dtft_af_array(phase_obs, p2ph, Nifg, W):
    searched_phase = p2ph.reshape(Nifg, 1) * W
    coh_phase = phase_obs.reshape(Nifg, 1) * np.ones((1, len(W))) - searched_phase
    xjw_dft = np.sum(np.exp(1j * coh_phase), axis=0) / Nifg
    coh = abs(xjw_dft)
    return coh


def wrap_phase(phase):
    """wrap phase to [-pi,pi]

    Args:
        phase (_float_): true phase without  phase ambiguites

    Returns:
        _type_: _description_
    """
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def add_gaussian_noise1(Nifg, noise_level_set):
    noise_phase = np.random.normal(0, np.pi * noise_level_set / 180, Nifg)
    return noise_phase


def par2phase(revisit_cycle, Nifg, normal_baseline):
    # v2ph = m2ph * (revisit_cycle / 365) * np.arange(1, Nifg + 1, 1)
    v2ph = revisit_cycle * np.arange(1, Nifg + 1, 1) * 4 * np.pi / (WAVELENGTH * 365)
    # h2ph = normal_baseline * (m2ph / (R * np.sin(Incidence_angle)))
    h2ph = np.random.normal(0, 333, Nifg) * 4 * np.pi / (WAVELENGTH * R * np.sin(Incidence_angle))
    # h2ph = m2ph * normal_baseline / (R * np.sin(Incidence_angle))
    return v2ph, h2ph


def construct_searching_parameters(Num_search_min, Num_search_max, step_orig):
    # searching_v = np.arange(-Num_search_min["velocity"], Num_search_max["velocity"] + 1, 1) * step_orig["velocity"]
    # searching_h = np.arange(-Num_search_min["height"], Num_search_max["height"] + 1, 1) * step_orig["height"]
    searching_v = np.arange(-1600, 1600 + 1, 1) * 0.0001
    searching_h = np.arange(-600, 600 + 1, 1) * 0.1
    return searching_v, searching_h


def sim_observed_phase(Nifg, param_sim, noise_level, v2ph, h2ph):
    # v2ph, h2ph = par2phase(revisit_cycle, Nifg, normal_baseline)
    phase_unwrapped = v2ph * param_sim["velocity"] + h2ph * param_sim["height"] + add_gaussian_noise1(Nifg, noise_level)
    phase_obs = wrap_phase(phase_unwrapped)
    return phase_obs


def DFT_phase_flatten(phase_obs, searching_v, h2ph, v2ph, Nifg, flatten_num):
    phase_flatten = np.zeros(len(searching_v))
    # rng_flatten = np.random.default_rng()
    for h in np.random.randint(-60, 60, flatten_num):
        flatten_phase = phase_obs - h2ph * h
        # DFT_signal = dtft_af_array(flatten_phase, v2ph, Nifg, searching_v)
        DFT_signal = dtft_af_all_array2(flatten_phase, v2ph).xjw(searching_v, Nifg)
        phase_flatten += DFT_signal
    phase_flatten = phase_flatten / flatten_num
    return phase_flatten


def param_estimation(param_file):
    param_file["normal_baseline"] = np.random.normal(0, param_file["Bn"], param_file["Nifg"])
    v2ph, h2ph = par2phase(param_file["revisit_cycle"], param_file["Nifg"], param_file["normal_baseline"])
    phase_obs = sim_observed_phase(
        param_file["revisit_cycle"], param_file["Nifg"], param_file["normal_baseline"], param_file["param_simulation"], param_file["noise_level"], v2ph, h2ph
    )
    searching_v, searching_h = construct_searching_parameters(param_file["Num_search_min"], param_file["Num_search_max"], param_file["step_orig"])
    phase_flatten = DFT_phase_flatten(phase_obs, searching_v, h2ph, v2ph, param_file["Nifg"], param_file["flatten_num"])
    v_est = searching_v[np.argmax(phase_flatten)]
    phase_obs_new_v = phase_obs - v2ph * v_est
    DFT_h = dtft_af_array(phase_obs_new_v, h2ph, param_file["Nifg"], searching_h)
    h_est = searching_h[np.argmax(DFT_h)]
    phase_obs_new_h = phase_obs - h2ph * h_est
    DFT_v = dtft_af_array(phase_obs_new_h, v2ph, param_file["Nifg"], searching_v)
    v_est = np.round(searching_v[np.argmax(DFT_v)], 4)
    h_est = np.round(h_est, 1)


def param_estimation_correct(param_file):
    param_file["normal_baseline"] = np.random.normal(0, param_file["Bn"], param_file["Nifg"])
    v2ph, h2ph = par2phase(param_file["revisit_cycle"], param_file["Nifg"], param_file["normal_baseline"])
    phase_obs = sim_observed_phase(param_file["Nifg"], param_file["param_simulation"], param_file["noise_level"], v2ph, h2ph)
    searching_v, searching_h = construct_searching_parameters(param_file["Num_search_min"], param_file["Num_search_max"], param_file["step_orig"])
    phase_flatten = DFT_phase_flatten(phase_obs, searching_v, h2ph, v2ph, param_file["Nifg"], 100)
    # 根据v估计,计算h估计值,得到v_est1,h_est1
    v_est1 = searching_v[np.argmax(phase_flatten)]
    phase_obs_new_v = phase_obs - v2ph * v_est1
    # DFT_h = dtft_af_array(phase_obs_new_v, h2ph, param_file["Nifg"], searching_h)
    DFT_h = dtft_af_all_array2(phase_obs_new_v, h2ph).xjw(searching_h, param_file["Nifg"])
    h_est1 = searching_h[np.argmax(DFT_h)]
    # 根据第一次 h_est1 估计值,修正v估计值,得到 v_est2
    phase_obs_new_h = phase_obs - h2ph * h_est1
    # DFT_v = dtft_af_array(phase_obs_new_h, v2ph, param_file["Nifg"], searching_v)
    DFT_v = dtft_af_all_array2(phase_obs_new_h, v2ph).xjw(searching_v, param_file["Nifg"])
    v_est2 = np.round(searching_v[np.argmax(DFT_v)], 4)
    # 根据修正后的 v_est2 估计值修正第一次 h_est1 估计值,得到 h_est2
    phase_obs_new_v = phase_obs - v2ph * v_est2
    # DFT_h = dtft_af_array(phase_obs_new_v, h2ph, param_file["Nifg"], searching_h)
    DFT_h = dtft_af_all_array2(phase_obs_new_v, h2ph).xjw(searching_h, param_file["Nifg"])
    h_est2 = searching_h[np.argmax(DFT_h)]
    #
    h_est = np.round(h_est2, 1)
    v_est = np.round(v_est2, 4)
    return v_est, h_est


def param_estimation_correct2(param_file):
    param_file["normal_baseline"] = np.random.normal(0, param_file["Bn"], param_file["Nifg"])
    dT = param_file["revisit_cycle"]
    N = param_file["Nifg"]
    v2ph, h2ph = par2phase(param_file["revisit_cycle"], param_file["Nifg"], param_file["normal_baseline"])
    # phase_obs = sim_observed_phase(param_file["Nifg"], param_file["param_simulation"], param_file["noise_level"], v2ph, h2ph)
    # v2ph = dT * np.arange(1, N + 1, 1) * 4 * np.pi / (WAVELENGTH * 365)
    # h2ph = ((np.arange(1, N + 1, 1) * dBn + np.random.normal(0, sigma_bn, N)) * 4 * np.pi / (Lambda * R * np.sin(Incidence_angle))).reshape(N)
    # h2ph = np.random.normal(0, 333, N) * 4 * np.pi / (WAVELENGTH * R * np.sin(Incidence_angle))
    phase_unwrapped = v2ph * param_file["param_simulation"]["velocity"] + h2ph * param_file["param_simulation"]["height"] + add_gaussian_noise1(N, param_file["noise_level"])
    phase_obs = wrap_phase(phase_unwrapped)
    searching_v = np.arange(-1600, 1600 + 1, 1) * 0.0001
    searching_h = np.arange(-600, 600 + 1, 1) * 0.1

    phase_flatten = DFT_phase_flatten(phase_obs, searching_v, h2ph, v2ph, param_file["Nifg"], 100)
    # 根据v估计,计算h估计值,得到v_est1,h_est1
    v_est1 = searching_v[np.argmax(phase_flatten)]
    phase_obs_new_v = phase_obs - v2ph * v_est1
    # DFT_h = dtft_af_array(phase_obs_new_v, h2ph, param_file["Nifg"], searching_h)
    DFT_h = dtft_af_all_array2(phase_obs_new_v, h2ph).xjw(searching_h, param_file["Nifg"])
    h_est1 = searching_h[np.argmax(DFT_h)]
    # 根据第一次 h_est1 估计值,修正v估计值,得到 v_est2
    phase_obs_new_h = phase_obs - h2ph * h_est1
    # DFT_v = dtft_af_array(phase_obs_new_h, v2ph, param_file["Nifg"], searching_v)
    DFT_v = dtft_af_all_array2(phase_obs_new_h, v2ph).xjw(searching_v, param_file["Nifg"])
    v_est2 = np.round(searching_v[np.argmax(DFT_v)], 4)
    # 根据修正后的 v_est2 估计值修正第一次 h_est1 估计值,得到 h_est2
    phase_obs_new_v = phase_obs - v2ph * v_est2
    # DFT_h = dtft_af_array(phase_obs_new_v, h2ph, param_file["Nifg"], searching_h)
    DFT_h = dtft_af_all_array2(phase_obs_new_v, h2ph).xjw(searching_h, param_file["Nifg"])
    h_est2 = searching_h[np.argmax(DFT_h)]
    #
    h_est = np.round(h_est2, 1)
    v_est = np.round(v_est2, 4)
    return v_est, h_est


def compute_success_rate(param, check_times):
    success_rate = 0
    v_est_data = np.zeros(check_times)
    h_est_data = np.zeros(check_times)
    for i in range(check_times):
        # print(f"check_times: {i}")
        v_est, h_est = param_estimation_correct(param)
        if abs(h_est - param["param_simulation"]["height"]) <= 0.5 and abs(v_est - param["param_simulation"]["velocity"]) <= 0.0005:
            success_rate += 1
        v_est_data[i] = v_est
        h_est_data[i] = h_est
        # print(f"v={v_est}, h={h_est}")
    return success_rate / check_times, v_est_data, h_est_data


def compute_success_rate_multi(param, check_times, data_range, process_id, data_length, shared_dict):
    print(f"process {process_id} start!")
    data_all = {process_id: {}}
    for i in range(len(data_range)):
        data_id = i + process_id * len(data_range)
        param["param_simulation"]["velocity"] = data_range[i]
        success_rate, v_est_data, h_est_data = compute_success_rate(param, check_times, data_id, data_length)
        print(f"process {process_id} data_id {data_id} success_rate: {success_rate}")
        data_all[process_id].update({data_id: {"success_rate": success_rate, "v_est_data": v_est_data, "h_est_data": h_est_data}})
    shared_dict.update(data_all)
    print(f"process {process_id} finished!")


def data_collect(data, changed_param, V_orig, process_num_all, test_length):
    success_rate = np.zeros((len(changed_param), len(V_orig)))
    v_est_data = []
    h_est_data = []
    suc = []
    for k in range(len(changed_param)):
        for i in range(process_num_all):
            for j in range(test_length):
                data_id = i * test_length + j
                success_rate[k][data_id] = data[k][i][data_id]["success_rate"]
                v_est_data.append(data[k][i][data_id]["v_est_data"])
                h_est_data.append(data[k][i][data_id]["h_est_data"])
    v_est_data = np.concatenate(v_est_data, axis=0)
    h_est_data = np.concatenate(h_est_data, axis=0)
    return success_rate, v_est_data, h_est_data
