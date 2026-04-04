import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 1. PRL Style Configuration
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

# System Constants
SIG_U = 0.12
SIG_E = 0.05
H_C = 0.3849


# ==========================================
# 2. Core Engines
# ==========================================
def get_x_roots_with_truncation(M):
    roots = np.roots([1, 0, -1, -M])
    real_roots = np.sort(roots[np.isreal(roots)].real)
    if len(real_roots) == 3 and M <= H_C:
        return real_roots[0], real_roots[-1]
    else:
        return np.nan, real_roots[-1]


def solve_theory_unified(mu_d, sig_d, mu_e=0.0, sig_e=0.0, x0_guess=[-1.0, 1.0]):
    def equations(vars):
        m, q = vars
        M = mu_d * m + mu_e * (m ** 2)
        Gamma = np.sqrt(SIG_U ** 2 + (sig_d ** 2) * q + (sig_e ** 2) * (q ** 2))
        xn, xp = get_x_roots_with_truncation(M)
        if np.isnan(xn):
            phi = 1.0
            target_m, target_q = xp, xp ** 2
        else:
            phi = 0.5 * (1 + erf((M - H_C) / (np.sqrt(2) * Gamma)))
            target_m = (1 - phi) * xn + phi * xp
            target_q = (1 - phi) * (xn ** 2) + phi * (xp ** 2)
        return [m - target_m, q - target_q]

    sol = fsolve(equations, x0=x0_guess, xtol=1e-9)
    m_f, q_f = sol
    M_f = mu_d * m_f + mu_e * m_f ** 2
    G_f = np.sqrt(SIG_U ** 2 + sig_d ** 2 * q_f + sig_e ** 2 * q_f ** 2)
    phi_f = 1.0 if M_f > H_C else 0.5 * (1 + erf((M_f - H_C) / (np.sqrt(2) * G_f)))
    return m_f, q_f, phi_f


def solve_system_ensemble(mu_d, sig_base, S=800, dt=0.05, steps=3000, n_trials=20):
    m_list, phi_list = [], []
    gamma_total = np.sqrt(SIG_U ** 2 + sig_base ** 2)
    for _ in range(n_trials):
        xi_i = np.random.normal(0, gamma_total, S)
        x = np.full(S, -1.0) + np.random.normal(0, 0.01, S)
        for t in range(steps):
            m = np.mean(x)
            x += (x - x ** 3 + mu_d * m + xi_i) * dt
            x = np.clip(x, -2.5, 2.5)
        m_list.append(np.mean(x))
        phi_list.append(np.sum(x > 0) / S)
    return np.mean(m_list), np.std(m_list), np.mean(phi_list), np.std(phi_list)


# ==========================================
# 3. Main Plotting (2x3 Layout)
# ==========================================
fig = plt.figure(figsize=(12, 7.5))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# --- A. Mean State m (Simulation + Theory) ---
ax_a = fig.add_subplot(gs[0, 0])
mu_d_range_sim = np.linspace(-0.6, 0.7, 10)
mu_d_range_th = np.linspace(-0.6, 0.7, 100)
for i, sig in enumerate([0.1, 0.25]):
    m_th = [solve_theory_unified(md, sig)[0] for md in mu_d_range_th]
    ax_a.plot(mu_d_range_th, m_th, '-', color=colors[i], alpha=0.6, label=rf'Theo. $\sigma={sig}$')

    # Simulation with error bars
    m_m, m_s, _, _ = zip(*[solve_system_ensemble(md, sig) for md in mu_d_range_sim])
    ax_a.errorbar(mu_d_range_sim, m_m, yerr=m_s, fmt='o', mfc='none', color=colors[i], capsize=3)

ax_a.set_title("A. Mean State $m$")
ax_a.set_xlabel(r"$\mu_d$");
ax_a.set_ylabel(r"$m$")
ax_a.legend(frameon=False, fontsize=8)

# --- B. Tipping Rate phi (Simulation + Theory) ---
ax_b = fig.add_subplot(gs[0, 1])
for i, sig in enumerate([0.15, 0.3]):
    # Simulation with error bars
    _, _, p_m, p_s = zip(*[solve_system_ensemble(md, sig) for md in mu_d_range_sim])
    ax_b.errorbar(mu_d_range_sim, p_m, yerr=p_s, fmt='o', mfc='none', color=colors[i + 1], capsize=3)

    p_th = [solve_theory_unified(md, sig)[2] for md in mu_d_range_th]
    ax_b.plot(mu_d_range_th, p_th, '-', color=colors[i + 1], label=rf'Theo. $\sigma={sig}$')

ax_b.set_title("B. Tipping Rate $\phi$")
ax_b.set_xlabel(r"$\mu_d$");
ax_b.set_ylabel(r"$\phi$")

# --- C, D, E panels (Theoretical scans & PDF) ---
ax_c = fig.add_subplot(gs[0, 2])
for i, md in enumerate([-0.6, -0.2, 0.2]):
    gamma = np.sqrt(SIG_U ** 2 + 0.15 ** 2)
    x = np.full(2500, -1.0)
    xi = np.random.normal(0, gamma, 2500)
    for _ in range(2500):
        x += (x - x ** 3 + md * np.mean(x) + xi) * 0.05
    ax_c.hist(x, bins=35, density=True, alpha=0.4, color=colors[i], label=rf'$\mu_d={md}$')
ax_c.set_title("C. PDF Evolution")
ax_c.legend(frameon=False, fontsize=8)

ax_d = fig.add_subplot(gs[1, 0])
mu_scan = np.linspace(-1.2, 0.6, 100)
for i, s in enumerate([0.1, 0.25, 0.35]):
    p_l = []
    guess = [-1.0, 1.0]
    for mu in mu_scan:
        m, q, p = solve_theory_unified(mu, s, sig_e=SIG_E, x0_guess=guess)
        p_l.append(p);
        guess = [m, q]
    ax_d.plot(mu_scan, p_l, color=colors[i], label=rf"$\sigma_d={s}$")
ax_d.set_title(r"D. Coupling Effect")
ax_d.set_xlabel(r"$\mu_d$");
ax_d.set_ylabel(r"$\phi$")

ax_e = fig.add_subplot(gs[1, 1:])
sig_scan = np.linspace(0.05, 1.5, 100)
for i, m_val in enumerate([-0.6, -0.2, 0.4]):
    p_l = []
    guess = [-1.0, 1.0]
    for sig in sig_scan:
        res_m, res_q, p = solve_theory_unified(m_val, sig, sig_e=SIG_E, x0_guess=guess)
        p_l.append(p);
        guess = [res_m, res_q]
    ax_e.plot(sig_scan, p_l, color=colors[i], label=rf"$\mu_d={m_val}$")
ax_e.set_title(r"E. Heterogeneity Effect")
ax_e.set_xlabel(r"$\sigma_d$")
ax_e.legend(frameon=False, ncol=3, fontsize=8)

plt.show()
