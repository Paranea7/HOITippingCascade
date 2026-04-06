import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve


# ==========================================
# 1. Style Configuration (PRL Standard)
# ==========================================
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
        "figure.dpi": 150
    })


set_prl_style()
colors_list = ['#1f77b4', '#d62728']  # Blue for Forward, Red for Backward

# System constants
MU_D, SIGMA_D = 0., 0.4
MU_E, SIGMA_E = 0.0, 0.0
HC = 0.3849


def equations(vars, mu_u, sigma_u):
    m, q, phi = vars
    M = mu_u + MU_D * m + MU_E * m ** 2
    Gamma2 = max(sigma_u ** 2 + SIGMA_D ** 2 * q + SIGMA_E ** 2 * q ** 2, 1e-6)
    res_phi = 0.5 * (1 + erf((M - HC) / (np.sqrt(2 * Gamma2)))) - phi
    res_m = (2 * phi - 1) + M / 2 - m
    res_q = 1 + (2 * phi - 1) * M + (M ** 2 + Gamma2) / 4 - q
    return [res_m, res_q, res_phi]


def sweep(vals, const, is_mu):
    f_res, b_res = [], []
    g_f, g_b = np.array([-1.0, 1.0, 0.0]), np.array([1.5, 2.5, 1.0])

    for v in vals:
        args = (v, const) if is_mu else (const, v)
        sol, info, ier, _ = fsolve(equations, g_f, args=args, full_output=True, xtol=1e-8)
        f_res.append(sol if ier == 1 else [np.nan] * 3)
        if ier == 1: g_f = sol

    for v in vals[::-1]:
        args = (v, const) if is_mu else (const, v)
        sol, info, ier, _ = fsolve(equations, g_b, args=args, full_output=True, xtol=1e-8)
        b_res.append(sol if ier == 1 else [np.nan] * 3)
        if ier == 1: g_b = sol
    return np.array(f_res), np.array(b_res)[::-1]


# ==========================================
# 2. Data Generation
# ==========================================
mu_vals = np.linspace(-1.0, 2.0, 400)
sig_vals = np.linspace(0.1, 1.5, 400)

# 为 mu 扫频设置固定的噪声，为 sigma 扫频设置固定的偏置
mu_f, mu_b = sweep(mu_vals, 0.3, True)
sig_f, sig_b = sweep(sig_vals, 0.5, False)

# ==========================================
# 3. Plotting (1x2 Layout)
# ==========================================
fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 3.5))

plot_data = [
    (mu_vals, mu_f, mu_b, r'External Bias $\mu_u$', '(a)'),
    (sig_vals, sig_f, sig_b, r'Noise Intensity $\sigma_u$', '(b)')
]

for i, (x, data_f, data_b, xl, label) in enumerate(plot_data):
    # 绘制前向和后向曲线
    axs[i].plot(x, data_f[:, 0], color=colors_list[0], label='Forward')
    axs[i].plot(x, data_b[:, 0], color=colors_list[1], linestyle='--', label='Backward')

    # 填充滞后区域
    mask = ~np.isnan(data_f[:, 0]) & ~np.isnan(data_b[:, 0])
    axs[i].fill_between(x, data_f[:, 0], data_b[:, 0],
                        where=(np.abs(data_f[:, 0] - data_b[:, 0]) > 2e-2) & mask,
                        color='gray', alpha=0.2, label='Hysteresis')

    # 装饰
    axs[i].set_xlabel(xl)
    axs[i].set_ylabel(r'Order Parameter $m$')
    axs[i].text(0.05, 0.95, label, transform=axs[i].transAxes,
                fontsize=12, fontweight='bold', va='top', family='sans-serif')
    axs[i].legend(loc='lower right', frameon=False, fontsize=9)

plt.show()
