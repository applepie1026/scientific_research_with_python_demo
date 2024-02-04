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
    def __init__(self, param_file):
        param = deepcopy(param_file)
        self.param_file = param_file
        self.Nifg = param_file["Nifg"]
        self.revisit_cycle = param_file["revisit_cycle"]
        self.Bn_sigma = param_file["Bn"]
        self.noise_level = param_file["noise_level"]
        self.normal_baseline = param_file["normal_baseline"]
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
        phase = self.arc_phase
        # print(phase, phase.shape)
        for key in self.param_name:
            phase -= self._par2ph[key] * self.param_pseudo[key]

        self.a_float = (phase / (2 * np.pi)).reshape((self.Nifg))
        # print(self.a_float, type(self.a_float))

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
        Q_a = 1 / (4 * np.pi**2) * (Q_y + A @ Q_b0 @ A.T)
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
        VC[VC < 0] = (10 / 180 * np.pi) ** 2  # minimum variance factor
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
        VC[VC < 0] = (10 / 180 * np.pi) ** 2  # minimum variance factor
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
        # print(self.Q_a_guess_new)
        # 第二次ILS估计。基于新的模糊度协方差矩阵，以及根据先验信息得到的模糊度解
        a_fixed_group = lambda_method.main(self.a_float, self.Q_a_guess_new, method=1, ncands=2)[0]
        # 计算v,h参数
        x1 = self.compute_parameters(a_fixed_group[:, 0], self.arc_phase, self.A)
        x2 = self.compute_parameters(a_fixed_group[:, 1], self.arc_phase, self.A)
        return a_fixed_group, x1, x2


#     def check_success_rate(self):
#         i = 0
#         est_data = np.zeros(2 * self.check_times)
#         a_data_1st = np.zeros((self.check_times, self.Nifg))
#         for k in range(self.check_times):
#             a_fixed_group, x1, x2 = self.ils_estimation()
#             if abs((x1[0] - self.param_sim["height"]) < 0.5 and abs(x1[1] - self.param_sim["velocity"]) < 0.0005) or abs(
#                 (x2[0] - self.param_sim["height"]) < 0.5 and abs(x2[1] - self.param_sim["velocity"]) < 0.0005
#             ):
#                 i += 1
#             est_data[k] = x1[0]
#             est_data[k + self.check_times] = x1[1]
#             a_data_1st[k, :] = a_fixed_group[:, 0]
#         success_rate = i / self.check_times
#         return success_rate, est_data, a_data_1st


# def main(param, v_range, shared_dict, process_num):
#     print("进程 %s" % process_num)
#     success_rate_data = np.zeros(len(v_range))
#     data_all = {process_num: {}}
#     for j in range(len(v_range)):
#         param["param_simulation"]["velocity"] = v_range[j]
#         ils_est = ILS_IB_estimator(param)
#         success_rate, est_data, a_data_1st = ils_est.check_success_rate()
#         success_rate_data[j] = success_rate
#         k = j + (len(v_range) + 1) * process_num
#         data_all[process_num].update({k: est_data})
#         del ils_est
#     k = len(v_range) + (len(v_range) + 1) * process_num
#     data_all[process_num].update({k: success_rate_data})
#     shared_dict.update(data_all)
#     print("process %s done!" % process_num)
def check_success_rate(param):
    est_data = np.zeros(2 * param["check_times"])
    a_data_1st = np.zeros((param["check_times"], param["Nifg"]))
    i = 0
    for k in range(param["check_times"]):
        ils_est = ILS_IB_estimator(param)
        a_fixed_group, x1, x2 = ils_est.ils_estimation()
        if abs((x1[0] - param["param_simulation"]["height"]) < 0.5 and abs(x1[1] - param["param_simulation"]["velocity"]) < 0.0005) or abs(
            (x2[0] - param["param_simulation"]["height"]) < 0.5 and abs(x2[1] - param["param_simulation"]["velocity"]) < 0.0005
        ):
            i += 1
        est_data[k] = x1[0]
        est_data[k + param["check_times"]] = x1[1]
        a_data_1st[k, :] = a_fixed_group[:, 0]
        del ils_est
    success_rate = i / param["check_times"]
    return success_rate, est_data, a_data_1st


def main(param, v_range, shared_dict, process_num):
    print("进程 %s" % process_num)
    success_rate_data = np.zeros(len(v_range))
    data_all = {process_num: {}}
    for j in range(len(v_range)):
        param["param_simulation"]["velocity"] = v_range[j]
        success_rate, est_data, a_data_1st = check_success_rate(param)
        success_rate_data[j] = success_rate
        k = j + (len(v_range) + 1) * process_num
        data_all[process_num].update({k: est_data})

    k = len(v_range) + (len(v_range) + 1) * process_num
    data_all[process_num].update({k: success_rate_data})
    shared_dict.update(data_all)
    print("process %s done!" % process_num)
