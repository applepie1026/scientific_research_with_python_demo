import numpy as np


def guess_VC(sig0, Nifg):
    # sig0 = np.sqart(2 * 0.25**2)

    VC = 2 * sig0**2 * np.ones(Nifg)
    Qy = np.diag(VC)
    Qy_i = np.diag(1 / np.diag(Qy))

    return VC, Qy, Qy_i


def neq(A, Qy_i):
    N = A.T @ Qy_i @ A
    return N


def pseudoVCM(dH, vel):
    # significance factor
    k = 2.5
    sig_dH = dH / k
    sig_D = vel / 1e3 / k
    Q_b0 = np.diag(np.array([sig_dH, sig_D]) ** 2)

    return Q_b0


def ambiguity_VC(Qy, A, Q_b0):
    Q_a = 1 / (4 * np.pi**2) * (Qy + A @ Q_b0 @ A.T)
    Q_a = np.tril(Q_a) + np.tril(Q_a, -1).T

    return Q_a


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
