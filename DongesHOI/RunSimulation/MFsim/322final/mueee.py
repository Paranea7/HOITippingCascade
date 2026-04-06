import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve
import warnings

warnings.filterwarnings("ignore")

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
colors_list = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

SIG_U = 0.128
SIG_D = 0.00
H_C = 0.3849

# ==========================================
# 2. Theory Engine
# ==========================================
def solve_theory_m(mu_e, sig_e):
    def eq(v):
        m, q = v
        M = mu_e * (m ** 2)
        Gamma = np.sqrt(SIG_U ** 2 + SIG_D ** 2 * q + sig_e ** 2 * q ** 2)
        xp, xn = 1.0 + M / 2.0, -1.0 + M / 2.0
        phi_eff = 0.5 * (1 + erf((M - H_C) / (np.sqrt(2) * Gamma + 1e-9)))
        return [m - ((1 - phi_eff) * xn + phi_eff * xp),
                q - ((1 - phi_eff) * (xn ** 2) + phi_eff * (xp ** 2))]

    sol = fsolve(eq, x0=[-1.0, 1.0], xtol=1e-9)
    M_f = mu_e * (sol[0] ** 2)
    G_f = np.sqrt(SIG_U ** 2 + SIG_D ** 2 * sol[1] + sig_e ** 2 * sol[1] ** 2)
    phi_f = 0.5 * (1 + erf((M_f - H_C) / (np.sqrt(2) * G_f + 1e-9)))
    return sol[0], phi_f

# ==========================================
# 3. Dynamical Engine (with Batching)
# ==========================================
def solve_system_batch(mu_e, sig_e, S=1000, dt=0.02, steps=4000, n_trials=20):
    m_results, phi_results = [], []
    gamma_total = np.sqrt(SIG_U ** 2 + SIG_D ** 2 + sig_e ** 2)
    for _ in range(n_trials):
        xi = np.random.normal(0, gamma_total, S)
        x = np.full(S, -1.0) + np.random.normal(0, 0.01, S)
        for t in range(steps):
            m = np.mean(x)
            x += (x - x ** 3 + mu_e * (m ** 2) + xi) * dt
            x = np.clip(x, -2.0, 2.0)
        m_results.append(np.mean(x))
        phi_results.append(np.sum(x > 0) / S)
    return np.mean(m_results), np.std(m_results), np.mean(phi_results), np.std(phi_results)

def get_single_pdf(mu_e, sig_e=0.15, S=2500):
    gamma_total = np.sqrt(SIG_U ** 2 + SIG_D ** 2 + sig_e ** 2)
    xi = np.random.normal(0, gamma_total, S)
    x = np.full(S, -1.0) + np.random.normal(0, 0.01, S)
    for _ in range(4000):
        x += (x - x ** 3 + mu_e * (np.mean(x) ** 2) + xi) * 0.02
    return x

# ==========================================
# 4. Plotting (1x3 Layout)
# ==========================================
fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14, 4))
plt.subplots_adjust(wspace=0.25)

mu_e_range_sim = np.linspace(0.0, 1.0, 12)
mu_e_range_theo = np.linspace(0.0, 1.0, 100)
sig_e_samples = [0.05, 0.5]

for idx, se in enumerate(sig_e_samples):
    # Theory
    m_theo = [solve_theory_m(me, se)[0] for me in mu_e_range_theo]
    p_theo = [solve_theory_m(me, se)[1] for me in mu_e_range_theo]

    # Simulation with Batch/Error bars
    m_means, m_stds, p_means, p_stds = [], [], [], []
    for me in mu_e_range_sim:
        mm, ms, pm, ps = solve_system_batch(me, se, n_trials=20)
        m_means.append(mm); m_stds.append(ms)
        p_means.append(pm); p_stds.append(ps)

    # Plot Panel A (Mean State)
    ax_a.plot(mu_e_range_theo, m_theo, '-', color=colors_list[idx], alpha=0.7, label=rf"Theo. $\sigma_e={se}$")
    ax_a.errorbar(mu_e_range_sim, m_means, yerr=m_stds, fmt='o', mfc='none',
                  color=colors_list[idx], label=rf"Sim. $\sigma_e={se}$", capsize=3)

    # Plot Panel B (Tipping Rate)
    ax_b.plot(mu_e_range_theo, p_theo, '-', color=colors_list[idx], alpha=0.7, label=rf"Theo. $\sigma_e={se}$")
    ax_b.errorbar(mu_e_range_sim, p_means, yerr=p_stds, fmt='s', mfc='none',
                  color=colors_list[idx], markersize=4, label=rf"Sim. $\sigma_e={se}$", capsize=3)

# Panel C (PDF)
for i, me in enumerate([0.2, 0.5, 0.8]):
    final_x = get_single_pdf(me)
    ax_c.hist(final_x, bins=40, density=True, alpha=0.4, color=colors_list[i], label=rf'$\mu_e={me}$')

# Labels and Formatting
ax_a.set_title("A. Mean State $m$")
ax_a.set_xlabel(r"$\mu_e$"); ax_a.set_ylabel("$m$")
ax_a.legend(frameon=False, fontsize=7, ncol=2)

ax_b.set_title(r"B. Tipping Rate $\phi$")
ax_b.set_xlabel(r"$\mu_e$"); ax_b.set_ylabel(r"$\phi$")
ax_b.legend(frameon=False, fontsize=7, ncol=2)

ax_c.set_title(r"C. State PDF ($\sigma_e=0.15$)")
ax_c.set_xlabel("$x_i$")
ax_c.legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.show()
