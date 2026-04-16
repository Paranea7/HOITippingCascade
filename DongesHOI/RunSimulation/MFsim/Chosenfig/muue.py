import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve

# --- 1. 样式设置 ---
def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 1.5,
        "figure.dpi": 150,
        "legend.frameon": False
    })

set_prl_style()
colors_list = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

# --- 三体项参数对齐设置 ---
MU_D = 0.0
MU_E_FIXED = 0.2
SIGMA_E_VAL = 0.1
HC = 0.3849
SIGMA_U = 0.1

# --- 2. 核心物理逻辑 ---
def get_gamma_phi_q(m, mu_u, mu_d, mu_e, sig_u, sig_d, sig_e):
    q, alpha = 1.0, 0.8
    M = mu_u + mu_d * m + mu_e * (m ** 2)
    for _ in range(50):
        gamma_sq = sig_u**2 + sig_d**2 * q + sig_e**2 * (q ** 2)
        gamma = np.sqrt(max(gamma_sq, 1e-9))
        phi = 0.5 * (1 + erf((M - HC) / (np.sqrt(2) * gamma)))
        q_target = 1 + (2 * phi - 1) * M + (M**2 + gamma_sq) / 4
        if abs(q - q_target) < 1e-8: break
        q = alpha * q_target + (1 - alpha) * q
    return gamma, phi, q, M

def equations(vars, mu_u, mu_e, sigma_u, sig_e):
    m, q, phi = vars
    M = mu_u + mu_e * (m ** 2)
    gamma_sq = sigma_u**2 + sig_e**2 * (q ** 2)
    res_phi = 0.5 * (1 + erf((M - HC) / (np.sqrt(2 * np.sqrt(max(gamma_sq, 1e-6)))))) - phi
    res_m = (2 * phi - 1) + M / 2 - m
    res_q = 1 + (2 * phi - 1) * M + (M**2 + gamma_sq) / 4 - q
    return [res_m, res_q, res_phi]

def sweep(vals, mu_e, sig_e):
    f_res, b_res = [], []
    g_f, g_b = np.array([-1.0, 1.0, 0.0]), np.array([1.5, 2.0, 1.0])
    for v in vals:
        sol, info, ier, _ = fsolve(equations, g_f, args=(v, mu_e, SIGMA_U, sig_e), full_output=True, xtol=1e-9)
        if ier == 1: g_f = sol
        f_res.append(sol)
    for v in reversed(vals):
        sol, info, ier, _ = fsolve(equations, g_b, args=(v, mu_e, SIGMA_U, sig_e), full_output=True, xtol=1e-9)
        if ier == 1: g_b = sol
        b_res.append(sol)
    return np.array(f_res), np.array(b_res)[::-1]

# --- 3. 生成数据 ---
mu_vals = np.linspace(-1.0, 2.0, 400)
mu_f, mu_b = sweep(mu_vals, MU_E_FIXED, SIGMA_E_VAL)
mu_f0, mu_b0 = sweep(mu_vals, MU_E_FIXED, 0.0)

# --- 4. 绘图 (修正后的索引方式) ---
fig, axs = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)

# (a) 自洽映射图 - 使用 axs[0]
m_range = np.linspace(-1.5, 2.5, 400)
u_tests = [-0.3, 0.0, 0.3]
for i, uu in enumerate(u_tests):
    m_outs = []
    for mi in m_range:
        _, phi, _, M = get_gamma_phi_q(mi, uu, 0.0, MU_E_FIXED, SIGMA_U, 0.0, SIGMA_E_VAL)
        m_outs.append(2 * phi - 1 + M / 2)
    axs[0].plot(m_range, m_outs, color=colors_list[i], label=rf"$\mu_u={uu}$")

axs[0].plot([-2, 3], [-2, 3], 'k--', lw=0.8, alpha=0.5)
axs[0].set_xlim([-1.5, 2.5]); axs[0].set_ylim([-1.5, 2.5])
axs[0].set_aspect('equal')
axs[0].set_xlabel("$m_{in}$"); axs[0].set_ylabel("$m_{out}$")
axs[0].legend(loc='upper left', fontsize=8)

# (b) 序参量 m 扫描 - 使用 axs[1]
axs[1].plot(mu_vals, mu_f[:, 0], color=colors_list[0], label=rf'Fwd ($\sigma_e={SIGMA_E_VAL}$)')
axs[1].plot(mu_vals, mu_b[:, 0], color=colors_list[1], linestyle='--', label=rf'Bwd ($\sigma_e={SIGMA_E_VAL}$)')
axs[1].plot(mu_vals, mu_f0[:, 0], color=colors_list[2], label=r'Fwd ($\sigma_e=0$)', alpha=0.7)
axs[1].plot(mu_vals, mu_b0[:, 0], color=colors_list[3], linestyle=':', label=r'Bwd ($\sigma_e=0$)', alpha=0.7)
diff = np.abs(mu_f[:, 0] - mu_b[:, 0])
axs[1].fill_between(mu_vals, mu_f[:, 0], mu_b[:, 0], where=(diff > 0.02), color='gray', alpha=0.15)
axs[1].set_xlabel(r'Bias $\mu_u$'); axs[1].set_ylabel(r'Order Parameter $m$')
axs[1].legend(loc='lower right', fontsize=7)

# (c) 跳变率 phi 扫描 - 使用 axs[2]
axs[2].plot(mu_vals, mu_f[:, 2], color=colors_list[0])
axs[2].plot(mu_vals, mu_b[:, 2], color=colors_list[1], linestyle='--')
axs[2].plot(mu_vals, mu_f0[:, 2], color=colors_list[2], alpha=0.7)
axs[2].plot(mu_vals, mu_b0[:, 2], color=colors_list[3], linestyle=':', alpha=0.7)
diff_phi = np.abs(mu_f[:, 2] - mu_b[:, 2])
axs[2].fill_between(mu_vals, mu_f[:, 2], mu_b[:, 2], where=(diff_phi > 0.02), color='gray', alpha=0.15)
axs[2].set_xlabel(r'Bias $\mu_u$'); axs[2].set_ylabel(r'Tipping Rate $\phi$')

# 标注 (a, b, c)
for i, tag in enumerate(['(a)', '(b)', '(c)']):
    axs[i].text(-0.15, 1.1, tag, transform=axs[i].transAxes, fontweight='bold', fontsize=12, va='top')

plt.show()
