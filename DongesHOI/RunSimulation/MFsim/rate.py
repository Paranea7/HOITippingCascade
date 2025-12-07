import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import math


# =========================================================
# single_phi: 使用 math.erf —— numba 完全支持
# =========================================================

@njit
def single_phi(mu_h, sigma_h):
    a = -mu_h / sigma_h
    return 0.5 * (1 - math.erf(a / math.sqrt(2)))


# =========================================================
# 截断高斯 E[x], E[x^2]
# =========================================================

@njit
def single_m_q(mu_h, sigma_h, phi):
    a = -mu_h / sigma_h
    Z = phi

    c = math.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)

    m = mu_h + sigma_h * (c / Z)
    q = (mu_h * mu_h + sigma_h * sigma_h +
         mu_h * sigma_h * (c / Z))
    return m, q


# =========================================================
# 自洽求解器（完全兼容 numba）
# =========================================================

@njit
def fixed_point_solver(mu_c, mu_d, mu_e,
                       sigma_c, sigma_d, sigma_e,
                       max_iter=2000, tol=1e-8):

    m = -0.2
    q = 0.2

    for _ in range(max_iter):

        mu_h = mu_c + mu_d * m + mu_e * q
        sigma_h = math.sqrt(
            sigma_c*sigma_c +
            sigma_d*sigma_d * q +
            sigma_e*sigma_e * q*q
        )

        phi = single_phi(mu_h, sigma_h)
        m_new, q_new = single_m_q(mu_h, sigma_h, phi)

        if abs(m_new - m) < tol and abs(q_new - q) < tol:
            return m_new, q_new, phi

        m = m_new
        q = q_new

    return m, q, phi


def cavity_phi(mu_c, mu_d, mu_e, sigma_c, sigma_d, sigma_e):
    m, q, phi = fixed_point_solver(
        mu_c, mu_d, mu_e,
        sigma_c, sigma_d, sigma_e
    )
    return phi


# =========================================================
# 二：并行二维扫描 —— sigma_d, sigma_e 平面
# =========================================================

@njit(parallel=True)
def compute_phi_grid_sigma(mu_c, mu_d, mu_e, sigma_c,
                           sigma_d_vals, sigma_e_vals):
    Sd = len(sigma_d_vals)
    Se = len(sigma_e_vals)
    phi_map = np.zeros((Sd, Se))

    for i in prange(Sd):
        for j in range(Se):
            sd = sigma_d_vals[i]
            se = sigma_e_vals[j]
            m, q, phi = fixed_point_solver(mu_c, mu_d, mu_e,
                                           sigma_c, sd, se)
            phi_map[i, j] = phi

    return phi_map


# =========================================================
# 三：并行二维扫描 —— mu_d, mu_e 平面
# =========================================================

@njit(parallel=True)
def compute_phi_grid_mu(mu_c, sigma_c, sigma_d, sigma_e,
                        mu_d_vals, mu_e_vals):
    Md = len(mu_d_vals)
    Me = len(mu_e_vals)
    phi_map = np.zeros((Md, Me))

    for i in prange(Md):
        for j in range(Me):
            md = mu_d_vals[i]
            me = mu_e_vals[j]
            m, q, phi = fixed_point_solver(mu_c, md, me,
                                           sigma_c, sigma_d, sigma_e)
            phi_map[i, j] = phi

    return phi_map


# =========================================================
# 四：三个图 —— 可直接运行
# =========================================================

def plot_phi_vs_sigma_d():
    mu_c = 0
    mu_d = 0.4
    mu_e = 0.3
    sigma_c = 0.2
    sigma_e = 0.6

    sigma_d_vals = np.linspace(0, 2, 200)
    phi_vals = [cavity_phi(mu_c, mu_d, mu_e, sigma_c, sd, sigma_e)
                for sd in sigma_d_vals]

    plt.figure()
    plt.plot(sigma_d_vals, phi_vals)
    plt.xlabel("sigma_d")
    plt.ylabel("phi")
    plt.title("phi vs sigma_d")
    plt.grid()
    plt.show()


def plot_sigma_plane():
    mu_c = 0
    mu_d = 0.2
    mu_e = 0.3
    sigma_c = 0.12

    sigma_d_vals = np.linspace(0, 2, 120)
    sigma_e_vals = np.linspace(0, 2, 120)

    phi_map = compute_phi_grid_sigma(mu_c, mu_d, mu_e, sigma_c,
                                     sigma_d_vals, sigma_e_vals)

    plt.figure()
    cs = plt.contour(sigma_d_vals, sigma_e_vals, phi_map.T,
                     levels=[0.5], colors='red')
    plt.clabel(cs)
    plt.xlabel("sigma_d")
    plt.ylabel("sigma_e")
    plt.title("phi phase boundary (sigma_d, sigma_e)")
    plt.grid()
    plt.show()


def plot_mu_plane():
    mu_c = 0
    sigma_c = 0.123
    sigma_d = 0.2
    sigma_e = 0.2

    mu_d_vals = np.linspace(-0.5, 0.5, 150)
    mu_e_vals = np.linspace(-0.5, 0.5, 150)

    phi_map = compute_phi_grid_mu(mu_c, sigma_c, sigma_d, sigma_e,
                                  mu_d_vals, mu_e_vals)

    plt.figure()
    cs = plt.contour(mu_d_vals, mu_e_vals, phi_map.T,
                     levels=[0.1], colors='blue')
    plt.clabel(cs)
    plt.xlabel("mu_d")
    plt.ylabel("mu_e")
    plt.title("phi phase boundary (mu_d, mu_e)")
    plt.grid()
    plt.show()


# =========================================================
# 主程序 —— 运行这三张图
# =========================================================

if __name__ == "__main__":
    plot_phi_vs_sigma_d()
    plot_sigma_plane()
    plot_mu_plane()