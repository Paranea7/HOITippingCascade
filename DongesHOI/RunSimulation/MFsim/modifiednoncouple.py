import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve

# ==========================================
# 1. Style Configuration
# ==========================================
def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.2,
        "figure.dpi": 150
    })

set_prl_style()
colors_main = ['#1f77b4', '#d62728']
colors_ctrl = ['#2ca02c', '#ff7f0e']
HC = 0.3849

# ==========================================
# 2. Calculation Functions
# ==========================================
def equations(vars, mu_u, sigma_u):
    m, q, phi = vars
    M = mu_u
    Gamma2 = max(sigma_u ** 2, 1e-6)
    res_phi = 0.5 * (1 + erf((M - HC) / (np.sqrt(2 * Gamma2)))) - phi
    res_m = (2 * phi - 1) + M / 2 - m
    res_q = 1 + (2 * phi - 1) * M + (M ** 2 + Gamma2) / 4 - q
    return [res_m, res_q, res_phi]

def sweep(vals, sigma_u):
    f_res = []
    g_f = np.array([-1.0, 1.0, 0.0])
    for v in vals:
        sol, info, ier, _ = fsolve(equations, g_f, args=(v, sigma_u), full_output=True, xtol=1e-8)
        f_res.append(sol if ier == 1 else [np.nan] * 3)
        if ier == 1: g_f = sol
    return np.array(f_res)

def solve_m_out(m_in, mu_u, mu_d=0.0, mu_e=0.0, sig_u=0.1):
    q, alpha, M = 1.0, 0.8, mu_u + mu_d * m_in + mu_e * (m_in ** 2)
    gamma = np.sqrt(max(sig_u ** 2, 1e-9))
    phi = 0.5 * (1 + erf((M - HC) / (np.sqrt(2) * gamma)))
    m_out = 2 * phi - 1 + M / 2
    return m_out

# ==========================================
# 3. Plotting (1x3 Layout)
# ==========================================
fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)

# --- (a) Self-consistency Map ---
m_in_range = np.linspace(-1.5, 2.5, 400)
u_test_vals = [-0.5, 0.0, 0.5]
u_colors = ["#1f77b4", "#d62728", "#2ca02c"]

for i, uu in enumerate(u_test_vals):
    m_outs = np.array([solve_m_out(mi, uu) for mi in m_in_range])
    axs[0].plot(m_in_range, m_outs, color=u_colors[i], label=rf"$\mu_u={uu}$")

# 绘制辅助线，但不添加 label 从而在图例中隐藏
axs[0].plot([-2, 3], [-2, 3], 'k--', lw=0.8, alpha=0.5)
axs[0].set_xlim([-1.5, 2.5])
axs[0].set_ylim([-1.5, 2.5])
axs[0].set_aspect('equal')
axs[0].set_xlabel(r'$m_{in}$')
axs[0].set_ylabel(r'$m_{out}$')
axs[0].legend(frameon=False, fontsize=10, loc='upper left')

# --- (b) Steady State m ---
mu_vals = np.linspace(-1.0, 2.0, 300)
mu_f_1 = sweep(mu_vals, 0.05)
mu_f_2 = sweep(mu_vals, 0.50)

axs[1].plot(mu_vals, mu_f_1[:, 0], color=colors_main[0], label=r'$\sigma_u=0.05$')
axs[1].plot(mu_vals, mu_f_2[:, 0], color=colors_ctrl[0], label=r'$\sigma_u=0.50$')
axs[1].set_xlabel(r'External Bias $\mu_u$')
axs[1].set_ylabel(r'Order Parameter $m$')
axs[1].legend(frameon=False, fontsize=10)

# --- (c) Tipping Rate phi ---
axs[2].plot(mu_vals, mu_f_1[:, 2], color=colors_main[0])
axs[2].plot(mu_vals, mu_f_2[:, 2], color=colors_ctrl[0])
axs[2].set_xlabel(r'External Bias $\mu_u$')
axs[2].set_ylabel(r'Tipping Rate $\phi$')

# --- Labeling (a, b, c) ---
tags = ['(a)', '(b)', '(c)']
for i, ax in enumerate(axs):
    ax.text(-0.15, 1.1, tags[i], transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

plt.show()
