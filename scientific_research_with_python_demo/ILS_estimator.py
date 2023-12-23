import scientific_research_with_python_demo.utils as af
import numpy as np
import scientific_research_with_python_demo.ps_vce as vc
import scientific_research_with_python_demo.LAMBDA as lambda_method

WAVELENGTH = 0.056
m2ph = 4 * np.pi / WAVELENGTH


def sim_observed_phase(v, h, Nifg, noise_level, dT=36, Bn=333):
    normal_baseline = np.random.randn(1, Nifg) * Bn
    # print(normal_baseline)
    time_baseline = (np.arange(1, Nifg + 1, 1) * dT).reshape(1, Nifg)  # 减小重访周期 dt 能明显改善结果
    # calculate the input parameters of phase
    v2ph = af.v_coef(time_baseline).T
    h2ph = af.h_coef(normal_baseline).T
    A = np.hstack((h2ph * m2ph, v2ph * m2ph))

    phase_obs, srn, phase_unwrap, h_phase, v_phase = af.sim_arc_phase(v, h, v2ph, h2ph, noise_level)
    return phase_obs.reshape(Nifg), A, phase_unwrap


def guess_Q_a(sig0, A, h_bound, v_bound, Nifg):
    VC_guessed, Q_y, Q_y_i = vc.guess_VC(sig0, Nifg)
    # 获得参数 vel 和 h 的协方差矩阵
    Q_b0 = vc.pseudoVCM(h_bound, v_bound)
    # 获取一个猜测的模糊度协方差矩阵
    Q_a_guess = vc.ambiguity_VC(Q_y, A, Q_b0)

    return Q_a_guess, Q_b0, VC_guessed


def guess_VCM(phase_obs, Q_a_guess, A, VC_guessed):
    # phase_obs维度位Nifg
    # 错略估计整数解
    a_hat = (phase_obs / (2 * np.pi)).reshape((len(phase_obs)))
    afixed_cand = lambda_method.main(a_hat, Q_a_guess, method=3, ncands=2)[0]
    # afixed = afixed_cand[:, 0]
    afixed = afixed_cand
    # 重新计算解缠相位
    phase_unwrap_new = -2 * np.pi * afixed + phase_obs
    # 根基解缠相位、猜测的 VC 参数，参数的设计矩阵，重新估计 VC 参数
    VC = vc.VCE(VC_guessed, A, phase_unwrap_new)
    # print(VC)
    return VC, a_hat


def upgrade_Q_a_guess(VC, A, Q_b0):
    # VC_mean = np.mean(VC, axis=0)
    # print(VC_mean)
    Qy_new = np.diag(VC)
    Qy_new_i = np.diag(1 / np.diag(Qy_new))
    Q_a_guess_new = vc.ambiguity_VC(Qy_new, A, Q_b0)

    return Q_a_guess_new


def lambda_ils(phase_obs, A, h_bound, v_bound, sig0):
    # 建立初始的模糊度协方差矩阵
    Q_a_guess, Q_b0, VC_guessed = guess_Q_a(sig0, A, h_bound, v_bound, len(phase_obs))
    # 猜测的VC参数
    VC, a_hat = guess_VCM(phase_obs, Q_a_guess, A, VC_guessed)
    # 重新计算模糊度协方差矩阵
    Q_a_guess_new = upgrade_Q_a_guess(VC, A, Q_b0)
    # 第二次ILS估计。基于新的模糊度协方差矩阵，以及第一次估计的模糊度解
    afixed_cand, sqnorm, Ps, Qzhat, Z, nfixed, mu = lambda_method.main(a_hat, Q_a_guess_new, method=1)
    afixed_new = afixed_cand
    return afixed_new, Ps


def main(v, h, Nifg, noise_level, dT, Bn, h_bound, v_bound, sig0):
    # simulate observed phase
    phase_obs, A, phase_unwrap = sim_observed_phase(v, h, Nifg, noise_level, dT, Bn)
    a_true = (af.wrap_phase(phase_unwrap).reshape(Nifg) - phase_unwrap.reshape(Nifg)) / (2 * np.pi)
    # print(phase_obs.shape)
    # lambda-ILS
    afixed_new, Ps = lambda_ils(phase_obs, A, h_bound, v_bound, sig0)
    x1 = compute_parameters(afixed_new[:, 0], phase_obs, A)
    x2 = compute_parameters(afixed_new[:, 1], phase_obs, A)
    return afixed_new, a_true, Ps, x1, x2


def compute_parameters(a_fixed, phase_obs, A):
    phase_unwrap_new = -2 * np.pi * a_fixed + phase_obs
    # 根据phase_unwrap_new=Ax计算x
    x = np.linalg.lstsq(A, phase_unwrap_new, rcond=None)[0]
    return x


def check_success_rate(v, h, Nifg, noise_level, dT, Bn, h_bound, v_bound, sig0, check_times=1000):
    i = 0
    success_rate = []
    for k in range(check_times):
        x = main(v, h, Nifg, noise_level, dT, Bn, h_bound, v_bound, sig0)[3]
        if abs(x[0] - h) < 0.5 and abs(x[1] - v) < 0.0005:
            i += 1
    return i / check_times