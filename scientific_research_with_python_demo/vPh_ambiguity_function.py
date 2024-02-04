import numpy as np
from copy import deepcopy
import json

# Constant
WAVELENGTH = 0.056  # [unit:m]
H = 780000  # satellite vertical height[m]
Incidence_angle = 23 * np.pi / 180  # the local incidence angle
R = H / np.cos(Incidence_angle)  # range to the master antenna. test
m2ph = 4 * np.pi / WAVELENGTH


class DFT_periodogram:
    def __init__(self, param_file, data_id, check_num, check_times, data_length) -> None:
        param = deepcopy(param_file)
        self.data_id = data_id
        self.check_num = check_num
        self.Nifg = param["Nifg"]
        self.param_sim = param["param_simulation"]
        self.noise_level = param["noise_level"]
        self.step_orig = param["step_orig"]
        self.param_name = param["param_name"]
        self.Num_search_min = param["Num_search_min"]
        self.Num_search_max = param["Num_search_max"]
        self.revisit_cycle = param["revisit_cycle"]
        self.Bn = param["Bn"]
        # self.rng_seed = data_id
        self.rng_seed_bn = check_num + data_id * check_times
        self.rng_seed_noise = check_num + data_length * check_times
        self.rng_flatten = check_num + 2 * data_length * check_times
        rng = np.random.default_rng(self.rng_seed_bn)
        self.normal_baseline = rng.normal(0, param["Bn"], param["Nifg"])
        # print(f"{check_num},{data_id}:{self.normal_baseline}")
        self.flatten_num = param["flatten_num"]

    @staticmethod
    def wrap_phase(phase):
        """wrap phase to [-pi,pi]

        Args:
            phase (_float_): true phase without  phase ambiguites

        Returns:
            _type_: _description_
        """
        return np.mod(phase + np.pi, 2 * np.pi) - np.pi

    @staticmethod
    def _add_gaussian_noise1(Nifg, noise_level_set, rng_seed):
        rng_noise = np.random.default_rng(rng_seed)
        noise_phase = rng_noise.normal(0, np.pi * noise_level_set / 180, Nifg)
        return noise_phase

    @staticmethod
    def dtft_af_array(phase_obs, p2ph, Nifg, W):
        searched_phase = p2ph.reshape(Nifg, 1) * W
        coh_phase = phase_obs.reshape(Nifg, 1) * np.ones((1, len(W))) - searched_phase
        xjw_dft = np.sum(np.exp(1j * coh_phase), axis=0) / Nifg
        coh = abs(xjw_dft)
        return coh

    def par2phase(self):
        self.v2ph = m2ph * (self.revisit_cycle / 365) * np.arange(1, self.Nifg + 1, 1)
        self.h2ph = (m2ph / R * np.sin(Incidence_angle)) * self.normal_baseline

    def construct_searching_parameters(self):
        self.searching_v = np.arange(-self.Num_search_min["velocity"], self.Num_search_max["velocity"] + 1, 1) * self.step_orig["velocity"]
        self.searching_h = np.arange(-self.Num_search_min["height"], self.Num_search_max["height"] + 1, 1) * self.step_orig["height"]

    def sim_observed_phase(self):
        self.par2phase()
        phase_unwrapped = self.v2ph * self.param_sim["velocity"] + self.h2ph * self.param_sim["height"]
        noise_phase = DFT_periodogram._add_gaussian_noise1(self.Nifg, self.noise_level, self.rng_seed_noise)
        # print(noise_phase)
        phase_unwrapped += noise_phase
        self.phase_obs = self.wrap_phase(phase_unwrapped)

    def DFT_phase_flatten(self):
        self.phase_flatten = np.zeros(len(self.searching_v))
        rng_flatten = np.random.default_rng(self.rng_flatten)
        for h in rng_flatten.integers(-60, 60, self.flatten_num, endpoint=True):
            flatten_phase = self.phase_obs - self.h2ph * h
            DFT_signal = self.dtft_af_array(flatten_phase, self.v2ph, self.Nifg, self.searching_v)
            self.phase_flatten += DFT_signal
        self.phase_flatten /= self.flatten_num

    def param_estimation(self):
        self.sim_observed_phase()
        self.construct_searching_parameters()
        self.DFT_phase_flatten()
        v_est = self.searching_v[np.argmax(self.phase_flatten)]
        phase_obs_new_v = self.phase_obs - self.v2ph * v_est
        self.DFT_h = self.dtft_af_array(phase_obs_new_v, self.h2ph, self.Nifg, self.searching_h)
        self.h_est = self.searching_h[np.argmax(self.DFT_h)]
        phase_obs_new_h = self.phase_obs - self.h2ph * self.h_est
        self.DFT_v = self.dtft_af_array(phase_obs_new_h, self.v2ph, self.Nifg, self.searching_v)
        self.v_est = np.round(self.searching_v[np.argmax(self.DFT_v)], 4)
        self.h_est = np.round(self.h_est, 1)

    def param_estimation_correct(self):
        self.sim_observed_phase()
        self.construct_searching_parameters()
        self.DFT_phase_flatten()
        # 根据v估计,计算h估计值,得到v_est1,h_est1
        v_est1 = self.searching_v[np.argmax(self.phase_flatten)]
        phase_obs_new_v = self.phase_obs - self.v2ph * v_est1
        DFT_h = self.dtft_af_array(phase_obs_new_v, self.h2ph, self.Nifg, self.searching_h)
        h_est1 = self.searching_h[np.argmax(DFT_h)]
        # 根据第一次 h_est1 估计值,修正v估计值,得到 v_est2
        phase_obs_new_h = self.phase_obs - self.h2ph * h_est1
        self.DFT_v = self.dtft_af_array(phase_obs_new_h, self.v2ph, self.Nifg, self.searching_v)
        v_est2 = np.round(self.searching_v[np.argmax(self.DFT_v)], 4)
        # 根据修正后的 v_est2 估计值修正第一次 h_est1 估计值,得到 h_est2
        phase_obs_new_v = self.phase_obs - self.v2ph * v_est2
        self.DFT_h = self.dtft_af_array(phase_obs_new_v, self.h2ph, self.Nifg, self.searching_h)
        h_est2 = self.searching_h[np.argmax(self.DFT_h)]

        self.h_est = np.round(h_est2, 1)
        self.v_est = np.round(v_est2, 4)


def compute_success_rate(param, check_times, data_id, data_length):
    success_rate = 0
    v_est_data = np.zeros(check_times)
    h_est_data = np.zeros(check_times)
    for i in range(check_times):
        # print(f"check_times: {i}")
        param_est = DFT_periodogram(param, data_id, i, check_times, data_length)
        param_est.param_estimation_correct()
        if abs(param_est.h_est - param["param_simulation"]["height"]) <= 0.5 and abs(param_est.v_est - param["param_simulation"]["velocity"]) <= 0.0005:
            success_rate += 1
        v_est_data[i] = param_est.v_est
        h_est_data[i] = param_est.h_est
        # print(f"v={param_est.v_est}, h={param_est.h_est}")
        del param_est
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
