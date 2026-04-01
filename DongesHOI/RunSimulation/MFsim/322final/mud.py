import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve

# ==========================================
# 1. 全局 PRL 風格配置
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
colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

# 系統核心參數
SIG_U = 0.12
SIG_D = 0.05
H_C = 0.3849

# ==========================================
# 2. 核心解析與動力學引擎
# ==========================================
def get_x_roots_truncated(M):
    roots = np.roots([1, 0, -1, -M])
    real_roots = np.sort(roots[np.isreal(roots)].real)
    if len(real_roots) == 3 and M <= H_C:
        return real_roots[0], real_roots[-1]
    else:
        return np.nan, real_roots[-1]

def solve_theory_unified(mu_d, sig_base=0.0):
    def equations(vars):
        m, q = vars
        M = mu_d * m
        Gamma = np.sqrt(SIG_U ** 2 + SIG_D ** 2 * q + sig_base ** 2)
        xn, xp = get_x_roots_truncated(M)
        if np.isnan(xn):
            phi = 1.0
            target_m, target_q = xp, xp**2
        else:
            phi = 0.5 * (1 + erf((M - H_C) / (np.sqrt(2) * Gamma)))
            target_m = (1 - phi) * xn + phi * xp
            target_q = (1 - phi) * (xn ** 2) + phi * (xp ** 2)
        return [m - target_m, q - target_q]

    sol = fsolve(equations, x0=[-1.0, 1.0], xtol=1e-8)
    m_f, q_f = sol
    M_f = mu_d * m_f
    phi_f = 1.0 if M_f > H_C else 0.5 * (1 + erf((M_f - H_C) / (np.sqrt(2) * np.sqrt(SIG_U**2 + SIG_D**2 * q_f + sig_base**2))))
    return m_f, phi_f

def solve_system_ensemble(mu_d, sig_base, S=1000, dt=0.05, steps=4000, x0=-1.0, n_trials=10):
    m_final_list, phi_list = [], []
    gamma_total = np.sqrt(SIG_U**2 + SIG_D**2 + sig_base**2)
    for _ in range(n_trials):
        xi_i = np.random.normal(0, gamma_total, S)
        x = np.full(S, x0, dtype=float) + np.random.normal(0, 0.01, S)
        for t in range(steps):
            m = np.mean(x)
            x += (x - x**3 + mu_d * m + xi_i) * dt
            x = np.clip(x, -2.5, 2.5)
        m_final_list.append(np.mean(x))
        phi_list.append(np.sum(x > 0) / S)
    return np.mean(m_final_list), np.mean(phi_list)

def solve_system_single_path(mu_d, sig_base, S=1000, dt=0.05, steps=4000):
    gamma_total = np.sqrt(SIG_U**2 + SIG_D**2 + sig_base**2)
    xi_i = np.random.normal(0, gamma_total, S)
    x = np.full(S, -1.0, dtype=float) + np.random.normal(0, 0.01, S)
    for t in range(steps):
        m = np.mean(x)
        x += (x - x**3 + mu_d * m + xi_i) * dt
        x = np.clip(x, -2.5, 2.5)
    return x

# ==========================================
# 3. 繪圖與分析 (僅保留 A, B, C)
# ==========================================
fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14, 4))
n_ens = 8

# --- A. Collective State m ---
mu_d_range = np.linspace(-0.6, 0.7, 12)
mu_d_theo = np.linspace(-0.6, 0.7, 100)
for i, sig in enumerate([0.1, 0.25]):
    m_th = [solve_theory_unified(md, sig)[0] for md in mu_d_theo]
    ax_a.plot(mu_d_theo, m_th, '-', color=colors[i], alpha=0.6, label=rf'Theo. $\sigma={sig}$')
    ms_sim = [solve_system_ensemble(md, sig, S=800, n_trials=n_ens)[0] for md in mu_d_range]
    ax_a.plot(mu_d_range, ms_sim, 'o', mfc='none', color=colors[i], label=rf'Sim. $\sigma={sig}$')
ax_a.set_title("A. Mean State $m$")
ax_a.set_xlabel(r"$\mu_d$")
ax_a.set_ylabel(r"$m$")
ax_a.legend(frameon=False, fontsize=8)

# --- B. Tipping Rate phi ---
mu_d_fine = np.linspace(-0.6, 0.7, 14)
for i, sig in enumerate([0.15, 0.3]):
    sim_phis = [solve_system_ensemble(md, sig, S=1000, n_trials=n_ens)[1] for md in mu_d_fine]
    ax_b.scatter(mu_d_fine, sim_phis, marker='o', s=20, edgecolors=colors[i+1], facecolors='none', label=rf'Sim. $\sigma={sig}$')
    fine_theo_phi = [solve_theory_unified(md, sig)[1] for md in mu_d_theo]
    ax_b.plot(mu_d_theo, fine_theo_phi, '-', color=colors[i+1], lw=1.2)
ax_b.set_title(r"B. Tipping Rate $\phi$")
ax_b.set_xlabel(r"$\mu_d$")
ax_b.set_ylabel(r"$\phi$")
ax_b.legend(frameon=False, fontsize=8)

# --- C. PDF Evolution ---
for i, md in enumerate([-0.6, -0.2, 0.2]):
    final_x = solve_system_single_path(md, 0.15, S=3000)
    ax_c.hist(final_x, bins=40, density=True, alpha=0.4, color=colors[i], label=rf'$\mu_d={md}$')
ax_c.set_title("C. PDF Evolution")
ax_c.set_xlabel("$x_i$")
ax_c.legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.show()
