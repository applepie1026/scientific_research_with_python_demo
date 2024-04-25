import numpy as np
from copy import deepcopy
import json
import scientific_research_with_python_demo.LAMBDA as lambda_method

# Constant
WAVELENGTH = 0.056  # [unit:m]
H = 780000  # satellite vertical height[m]
Incidence_angle = 23 * np.pi / 180  # the local incidence angle
R = H / np.cos(Incidence_angle)  # range to the master antenna. test
m2ph = 4 * np.pi / WAVELENGTH


class ILS_IB_estimator:
    def __init__(self, param_file, data_id, check_index):
        param = deepcopy(param_file)
        self.param_file = param_file
        self.Nifg = param_file["Nifg"]
        self.revisit_cycle = param_file["revisit_cycle"]
        self.Bn_sigma = param_file["Bn"]
        self.noise_level = param_file["noise_level"]
        rng_normal = np.random.default_rng(check_index)
        self.normal_baseline = rng_normal.normal(0, 333, param_file["Nifg"])
        self.param_sim = param["param_simulation"]
        self.param_pseudo = param["param_pseudo"]
        self.param_name = param["param_name"]
        self.h_bound = param_file["sigma_h"]
        self.v_bound = param_file["sigma_v"]
        self.check_times = param_file["check_times"]
        self.sig_y = np.sqrt(2 * (np.pi * param_file["sigma_y"] / 180) ** 2)

    def par2phase(self):
        # calculate the input parameters of phase
        v2ph = m2ph * (np.arange(1, self.Nifg + 1, 1) * self.revisit_cycle / 365)
        h2ph = m2ph * self.normal_baseline / (R * np.sin(Incidence_angle))
        par2ph = dict()
        par2ph["velocity"] = v2ph
        par2ph["height"] = h2ph
        self._par2ph = par2ph
        # print(self._par2ph)
        self.A = np.hstack((h2ph.reshape(-1, 1), v2ph.reshape(-1, 1)))

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
    def _add_gaussian_noise1(Nifg, noise_level_set):
        np.random.seed(0)
        noise_phase = np.random.normal(0, np.pi * noise_level_set / 180, Nifg)
        return noise_phase

    def sim_observed_phase(self):
        # simulate the observed phase
        # calculate the input parameters of phase
        self.par2phase()
        phase_true = np.zeros(self.Nifg)
        for key in self.param_name:
            phase_true += self._par2ph[key] * self.param_sim[key]

        # add noise
        # print(f"phase_true:{phase_true.shape}")
        phase_true = phase_true + self._add_gaussian_noise1(self.Nifg, self.noise_level)
        # print(f"phase_true:{phase_true.shape}")
        self.arc_phase = self.wrap_phase(phase_true)

    def guess_a_float(self):
        # calculate the float ambiguity
        # print("phase_obs:", self.arc_phase)
        # phase = self.arc_phase
        # print(id(phase), id(self.arc_phase))
        # for key in self.param_name:
        #     print(key)
        #     phase -= self._par2ph[key] * self.param_pseudo[key]
        phase = self.arc_phase - self._par2ph["velocity"] * self.param_pseudo["velocity"] - self._par2ph["height"] * self.param_pseudo["height"]
        # print("phase:", phase, phase.shape)
        self.a_float = (phase / (2 * np.pi)).reshape((self.Nifg))
        # print("a_hat:", self.a_float, type(self.a_float))
        # print("phase_obs:", self.arc_phase)

    # VC Estimation
    def guess_VC(self):
        self.VC_guessed = 2 * self.sig_y**2 * np.ones(self.Nifg)
        self.Qy_guessed = np.diag(self.VC_guessed)

    def guess_Q_b0(self):
        # 获得参数 vel 和 h 的协方差矩阵
        k = 1
        sig_dH = self.h_bound / k
        sig_D = (self.v_bound / 1e3) / k
        self.Q_b0 = np.diag(np.array([sig_dH, sig_D]) ** 2)

    @staticmethod
    def calculate_Q_a(Q_y, A, Q_b0):
        # 获取一个猜测的模糊度协方差矩阵,利用协方差计算性质（传递性）
        Q_a = (Q_y + A @ Q_b0 @ A.T) / (4 * np.pi**2)
        Q_a = np.tril(Q_a) + np.tril(Q_a, -1).T
        return Q_a

    @staticmethod
    def VCE(VC, A, uw_ph):
        # variance component estimation for DD InSAR phase time-series
        no_ifg = len(VC)
        Q_s = np.zeros((no_ifg, no_ifg))  # initialize cov.matrix
        # I = eye(size(y,1)); % identity matrix
        # P_A = I - A*((A'/Q_y*A)\A'/Q_y); # orthogonal projector P_A
        # e = P_A@y  # vector of least-squares residuals
        r = np.zeros(no_ifg)
        Qy = np.diag(VC)
        Qy_i = np.diag(1 / np.diag(Qy))  # inverse of diagonal matrix
        # orthogonal projector:
        P_A = np.eye(no_ifg) - A @ (np.linalg.inv(A.T @ Qy_i @ A) @ A.T @ Qy_i)
        res = P_A @ uw_ph  # vector of least-squares residuals
        Q_P_A = Qy_i @ P_A
        for i in np.arange(no_ifg):
            Q_v = Q_s.copy()
            Q_v[i, i] = 2
            # 2, no 1 -- see derivation in Freek phd
            r[i] = 0.5 * (res.T @ Qy_i @ Q_v @ Qy_i @ res)
        N = 2 * (Q_P_A * Q_P_A.T)
        VC = np.linalg.inv(N) @ r
        VC[VC < 1e-10] = (10 / 180 * np.pi) ** 2  # minimum variance factor
        # 2nd iteration:
        Qy = np.diag(VC)
        Qy_i = np.diag(1 / np.diag(Qy))  # inverse of diagonal matrix
        # orthogonal projector:
        P_A = np.eye(no_ifg) - A @ (np.linalg.inv(A.T @ Qy_i @ A) @ A.T @ Qy_i)
        res = P_A @ uw_ph  # vector of least-squares residuals
        Q_P_A = Qy_i @ P_A
        for i in np.arange(no_ifg):
            Q_v = Q_s.copy()
            Q_v[i, i] = 2
            # 2, no 1 -- see derivation in Freek phd
            r[i] = 0.5 * (res.T @ Qy_i @ Q_v @ Qy_i @ res)
        N = 2 * (Q_P_A * Q_P_A.T)
        VC = np.linalg.inv(N) @ r
        VC[VC < 1e-10] = (10 / 180 * np.pi) ** 2  # minimum variance factor
        return VC

    def guess_Q_a(self):
        self.guess_VC()
        self.guess_Q_b0()
        self.Q_a_guess = self.calculate_Q_a(self.Qy_guessed, self.A, self.Q_b0)

    def VC_upgrade(self):
        self.guess_a_float()
        self.guess_Q_a()
        a_fixed = lambda_method.main(self.a_float, self.Q_a_guess, method=1, ncands=2)[0]
        a_fixed_1st = (a_fixed[:, 0]).reshape(self.Nifg)
        # print(a_fixed_1st)
        unwrapped_phase = -2 * np.pi * a_fixed_1st + self.arc_phase
        self.VC_new = self.VCE(self.VC_guessed, self.A, unwrapped_phase)
        self.Q_a_guess_new = self.calculate_Q_a(np.diag(self.VC_new), self.A, self.Q_b0)

    @staticmethod
    def compute_parameters(a_fixed, phase_obs, A):
        phase_unwrap_new = -2 * np.pi * a_fixed + phase_obs
        # 根据phase_unwrap_new=Ax计算x
        x = np.linalg.lstsq(A, phase_unwrap_new, rcond=None)[0]
        return x

    def ils_estimation(self):
        self.sim_observed_phase()
        # VC矩阵修正，包含第一次ILS估计
        self.VC_upgrade()
        # print(self.Q_a_guess_new.shape)
        # 第二次ILS估计。基于新的模糊度协方差矩阵，以及根据先验信息得到的模糊度解
        a_fixed_group = lambda_method.main(self.a_float, self.Q_a_guess_new, method=1, ncands=2)[0]
        # 计算v,h参数
        x1 = self.compute_parameters(a_fixed_group[:, 0], self.arc_phase, self.A)
        x2 = self.compute_parameters(a_fixed_group[:, 1], self.arc_phase, self.A)
        return a_fixed_group, x1, x2

    # def check_success_rate(self):
    #     i = 0
    #     est_data = np.zeros(2 * self.check_times)
    #     a_data_1st = np.zeros((self.check_times, self.Nifg))
    #     for k in range(self.check_times):
    #         a_fixed_group, x1, x2 = self.ils_estimation()
    #         if abs((x1[0] - self.param_sim["height"]) < 0.5 and abs(x1[1] - self.param_sim["velocity"]) < 0.0005) or abs(
    #             (x2[0] - self.param_sim["height"]) < 0.5 and abs(x2[1] - self.param_sim["velocity"]) < 0.0005
    #         ):
    #             i += 1
    #         est_data[k] = x1[0]
    #         est_data[k + self.check_times] = x1[1]
    #         a_data_1st[k, :] = a_fixed_group[:, 0]
    #     success_rate = i / self.check_times
    #     return success_rate, est_data, a_data_1st


def check_success_rate(param):
    est_data_h = np.zeros(param["check_times"])
    est_data_v = np.zeros(param["check_times"])
    a_data_1st = np.zeros((param["check_times"], param["Nifg"]))
    i = 0
    for k in range(param["check_times"]):
        param["normal_baseline"] = np.random.normal(0, 333, param["Nifg"])
        ils_est = ILS_IB_estimator(param, 0, k)
        a_fixed_group, x1, x2 = ils_est.ils_estimation()
        if abs((x1[0] - param["param_simulation"]["height"]) < 0.5 and abs(x1[1] - param["param_simulation"]["velocity"]) < 0.0005) or abs(
            (x2[0] - param["param_simulation"]["height"]) < 0.5 and abs(x2[1] - param["param_simulation"]["velocity"]) < 0.0005
        ):
            i += 1
        est_data_h[k] = x1[0]
        est_data_v[k] = x1[1]
        print(x1[0], x1[1])
        a_data_1st[k, :] = a_fixed_group[:, 0]
        del ils_est
    success_rate = i / param["check_times"]
    print(success_rate)
    return success_rate, est_data_h, est_data_v, a_data_1st


# def main(param, v_range, shared_dict, process_num):
#     print("进程 %s" % process_num)
#     data_all = {process_num: {}}
#     for j in range(len(v_range)):
#         data_id=j+(len(v_range))*process_num
#         param["param_simulation"]["velocity"] = v_range[j]
#         success_rate, est_data_h,est_data_v, a_data_1st = check_success_rate(param)
#         data_all[process_num].update({data_id: {"success_rate": success_rate, "est_data_h": est_data_h,"est_data_v": est_data_v, "a_data_1st": a_data_1st}})
#     shared_dict.update(data_all)
#     print("process %s done!" % process_num)
def main(param, v_range, process_num):
    print("进程 %s" % process_num)
    data_all = {process_num: {}}
    for j in range(len(v_range)):
        data_id = j + (len(v_range)) * process_num
        param["param_simulation"]["velocity"] = v_range[j]
        success_rate, est_data_h, est_data_v, a_data_1st = check_success_rate(param)
        data_all[process_num].update({data_id: {"success_rate": success_rate, "est_data_h": est_data_h, "est_data_v": est_data_v, "a_data_1st": a_data_1st}})
    # shared_dict.update(data_all)
    print("process %s done!" % process_num)
    return data_all


def lab(param_file, change_name, change_data, V, data_id):
    # print(f"v={V}start")
    data = {}
    param_file[change_name] = change_data
    param_file["param_simulation"]["velocity"] = V
    est_data_h = np.zeros(param_file["check_times"])
    est_data_v = np.zeros(param_file["check_times"])
    a_data_1st = np.zeros((param_file["check_times"], param_file["Nifg"]))
    success_time = 0
    for i in range(param_file["check_times"]):
        ils_est = ILS_IB_estimator(param_file, data_id, i)
        a, x1, x2 = ils_est.ils_estimation()
        est_data_h[i] = x1[0]
        est_data_v[i] = x1[1]
        a_data_1st[i, :] = a[:, 0]
        if abs((x1[0] - param_file["param_simulation"]["height"]) < 0.5 and abs(x1[1] - param_file["param_simulation"]["velocity"]) < 0.0005) or abs(
            (x2[0] - param_file["param_simulation"]["height"]) < 0.5 and abs(x2[1] - param_file["param_simulation"]["velocity"]) < 0.0005
        ):
            success_time += 1
    success_rate = success_time / param_file["check_times"]
    data[data_id] = {"success_rate": success_rate, "est_data_h": est_data_h, "est_data_v": est_data_v, "a_data_1st": a_data_1st}
    print(f"v={V} done")
    return data


# def data_collect(data, changed_param, V_orig, process_num_all, test_length):
#     success_rate = np.zeros((len(changed_param), len(V_orig)))
#     v_est_data = []
#     h_est_data = []
#     for k in range(len(changed_param)):
#         for i in range(process_num_all):
#             for j in range(test_length):
#                 data_id = i * test_length + j
#                 success_rate[k][data_id] = data[k][i][data_id]["success_rate"]
#                 v_est_data.append(data[k][i][data_id]["est_data_v"].reshape(1, -1))
#                 h_est_data.append(data[k][i][data_id]["est_data_v"].reshape(1, -1))
#     v_est_data = np.concatenate(v_est_data, axis=0)
#     h_est_data = np.concatenate(h_est_data, axis=0)
#     return success_rate, v_est_data, h_est_data
def data_collect(data, changed_param_length, V_orig_length):
    success_rate = np.zeros((changed_param_length, V_orig_length))
    v_est_data = []
    h_est_data = []
    for k in range(changed_param_length):
        for i in range(V_orig_length):
            success_rate[k][i] = data[k][i]["success_rate"]
            v_est_data.append(data[k][i]["est_data_v"].reshape(1, -1))
            h_est_data.append(data[k][i]["est_data_v"].reshape(1, -1))
    v_est_data = np.concatenate(v_est_data, axis=0)
    h_est_data = np.concatenate(h_est_data, axis=0)
    return success_rate, v_est_data, h_est_data


def dict_collect(data_list):
    # 将列中的字典合并
    data = {}
    for i in range(len(data_list)):
        data.update(data_list[i])
    return data
