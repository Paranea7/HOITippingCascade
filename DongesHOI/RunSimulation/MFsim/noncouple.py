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
        "axes.labelsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "lines.linewidth": 1.5,
        "figure.dpi": 150
    })

set_prl_style()
colors_main = ['#1f77b4', '#d62728'] # Blue/Red
colors_ctrl = ['#2ca02c', '#ff7f0e'] # Green/Orange

# System constants (Uncoupled Case)
MU_D, SIGMA_D = 0.0, 0.0
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
# 2. Generate Data
# ==========================================
mu_vals = np.linspace(-1.0, 2.5, 300)
sig_vals = np.linspace(0.1, 2.5, 300)

# Row 1: Bias sweep with different Noise levels
mu_f_main, mu_b_main = sweep(mu_vals, 0.0001, True)
mu_f_ctrl, mu_b_ctrl = sweep(mu_vals, 0.50, True)

# Row 2: Noise sweep with different Bias levels
sig_f_main, sig_b_main = sweep(sig_vals, 0.0, False)
sig_f_ctrl, sig_b_ctrl = sweep(sig_vals, 0.6, False)

# ==========================================
# 3. Plotting (2x2 Layout)
# ==========================================
fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(8, 6))
labels = [['(a)', '(b)'], ['(c)', '(d)']]

# --- ROW 1: Sweep mu_u ---
x1 = mu_vals
axs[0, 0].plot(x1, mu_f_main[:, 0], color=colors_main[0], label=r'$\sigma_u=0.0001$ (Fwd)')
axs[0, 0].plot(x1, mu_b_main[:, 0], color=colors_main[1], linestyle='--', label=r'$\sigma_u=0.0001$ (Bwd)')
axs[0, 0].plot(x1, mu_f_ctrl[:, 0], color=colors_ctrl[0], label=r'$\sigma_u=0.50$ (Fwd)')
axs[0, 0].plot(x1, mu_b_ctrl[:, 0], color=colors_ctrl[1], linestyle=':', alpha=0.8, label=r'$\sigma_u=0.50$ (Bwd)')

axs[0, 1].plot(x1, mu_f_main[:, 2], color=colors_main[0])
axs[0, 1].plot(x1, mu_b_main[:, 2], color=colors_main[1], linestyle='--')
axs[0, 1].plot(x1, mu_f_ctrl[:, 2], color=colors_ctrl[0])
axs[0, 1].plot(x1, mu_b_ctrl[:, 2], color=colors_ctrl[1], linestyle=':', alpha=0.8)

# --- ROW 2: Sweep sigma_u ---
x2 = sig_vals
axs[1, 0].plot(x2, sig_f_main[:, 0], color=colors_main[0], label=r'$\mu_u=0.0$ (Fwd)')
axs[1, 0].plot(x2, sig_b_main[:, 0], color=colors_main[1], linestyle='--', label=r'$\mu_u=0.0$ (Bwd)')
axs[1, 0].plot(x2, sig_f_ctrl[:, 0], color=colors_ctrl[0], label=r'$\mu_u=0.6$ (Fwd)')
axs[1, 0].plot(x2, sig_b_ctrl[:, 0], color=colors_ctrl[1], linestyle=':', alpha=0.8, label=r'$\mu_u=0.6$ (Bwd)')

axs[1, 1].plot(x2, sig_f_main[:, 2], color=colors_main[0])
axs[1, 1].plot(x2, sig_b_main[:, 2], color=colors_main[1], linestyle='--')
axs[1, 1].plot(x2, sig_f_ctrl[:, 2], color=colors_ctrl[0])
axs[1, 1].plot(x2, sig_b_ctrl[:, 2], color=colors_ctrl[1], linestyle=':', alpha=0.8)

# --- Formatting ---
for i in range(2):
    axs[i, 0].set_ylabel(r'Order Parameter $m$')
    axs[i, 1].set_ylabel(r'Tipping Rate $\phi$')
    for j in range(2):
        axs[i, j].set_xlabel(r'External Bias $\mu_u$' if i == 0 else r'Noise Intensity $\sigma_u$')
        axs[i, j].text(0.05, 0.95, labels[i][j], transform=axs[i, j].transAxes,
                       fontsize=11, fontweight='bold', va='top')
    # Place legends only on the left column to avoid clutter
    axs[i, 0].legend(loc='lower right', frameon=False, fontsize=8)

plt.show()
