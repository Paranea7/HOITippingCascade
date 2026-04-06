import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve
import warnings

warnings.filterwarnings('ignore')


# ==========================================
# 1. Global Style Configuration
# ==========================================
def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix", "font.size": 10,
        "axes.labelsize": 11, "xtick.direction": "in",
        "ytick.direction": "in", "lines.linewidth": 1.5,
        "figure.dpi": 150
    })


set_prl_style()
COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
SIG_U = 0.128
H_C = 0.3849


# ==========================================
# 2. Analytical Engine (High-Order Interaction)
# ==========================================
def self_consistent_equations(vars, mu_e, sigma_e):
    m, q = vars
    q_val = max(q, 1e-6)
    M = mu_e * (m ** 2)
    Gamma = np.sqrt(SIG_U ** 2 + (sigma_e ** 2 * (q_val ** 2)))
    # Branch positions (First-order perturbation)
    xp, xn = 1.0 + M / 2.0, -1.0 + M / 2.0
    phi = 0.5 * (1 + erf((M - H_C) / (np.sqrt(2) * Gamma + 1e-12)))
    return [m - ((1 - phi) * xn + phi * xp), q - ((1 - phi) * (xn ** 2) + phi * (xp ** 2))]


def solve_steady_state(mu_e, sigma_e, initial_guess=[-1.0, 1.0]):
    sol = fsolve(self_consistent_equations, x0=initial_guess, args=(mu_e, sigma_e), xtol=1e-9)
    m_f, q_f = sol
    M_f = mu_e * (m_f ** 2)
    G_f = np.sqrt(SIG_U ** 2 + (sigma_e ** 2 * (q_f ** 2)))
    phi_f = 0.5 * (1 + erf((M_f - H_C) / (np.sqrt(2) * G_f + 1e-12)))
    return sol, phi_f


# ==========================================
# 3. Dynamical Engine (Batch Processing)
# ==========================================
def solve_system_batch(mu_e, sig_e, S=1600, dt=0.02, steps=3500, n_trials=20):
    m_res, p_res = [], []
    gamma_total = np.sqrt(SIG_U ** 2 + sig_e ** 2)
    for _ in range(n_trials):
        xi = np.random.normal(0, gamma_total, S)
        x = np.full(S, -1.0) + np.random.normal(0, 0.01, S)
        for t in range(steps):
            m = np.mean(x)
            x += (x - x ** 3 + mu_e * (m ** 2) + xi) * dt
            x = np.clip(x, -2.2, 2.2)
        m_res.append(np.mean(x))
        p_res.append(np.sum(x > 0) / S)
    return np.mean(m_res), np.std(m_res), np.mean(p_res), np.std(p_res)


def get_single_pdf(mu_e, sig_e=0.15, S=2500):
    gamma = np.sqrt(SIG_U ** 2 + sig_e ** 2)
    xi = np.random.normal(0, gamma, S)
    x = np.full(S, -1.0) + np.random.normal(0, 0.01, S)
    for _ in range(4000):
        x += (x - x ** 3 + mu_e * (np.mean(x) ** 2) + xi) * 0.02
    return x


# ==========================================
# 4. Merged Plotting (2x3 Layout)
# ==========================================
fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# --- A & B: Dynamics (mu_e Transition) ---
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
mu_sim_range = np.linspace(0.0, 1.0, 10)
mu_theo_range = np.linspace(0.0, 1.0, 100)

for idx, se in enumerate([0.1, 0.5]):
    # Simulation with Error Bars (Batch=20)
    m_m, m_s, p_m, p_s = [], [], [], []
    for me in mu_sim_range:
        mm, ms, pm, ps = solve_system_batch(me, se, n_trials=20)
        m_m.append(mm);
        m_s.append(ms);
        p_m.append(pm);
        p_s.append(ps)

    # Theory for A & B
    m_t = [solve_steady_state(me, se)[0][0] for me in mu_theo_range]
    p_t = [solve_steady_state(me, se)[1] for me in mu_theo_range]

    ax_a.plot(mu_theo_range, m_t, '-', color=COLORS[idx], alpha=0.6)
    ax_a.errorbar(mu_sim_range, m_m, yerr=m_s, fmt='o', mfc='none', color=COLORS[idx], label=rf"$\sigma_e={se}$",
                  capsize=3)

    ax_b.plot(mu_theo_range, p_t, '-', color=COLORS[idx], alpha=0.6)
    ax_b.errorbar(mu_sim_range, p_m, yerr=p_s, fmt='s', mfc='none', color=COLORS[idx], markersize=4, capsize=3)

ax_a.set_title("A. Mean State $m$");
ax_a.set_xlabel(r"$\mu_e$");
ax_a.legend(frameon=False, fontsize=8)
ax_b.set_title(r"B. Tipping Rate $\phi$");
ax_b.set_xlabel(r"$\mu_e$")

# --- C: PDF Evolution ---
ax_c = fig.add_subplot(gs[0, 2])
for i, me in enumerate([0.2, 0.6, 1.0]):
    final_x = get_single_pdf(me)
    ax_c.hist(final_x, bins=35, density=True, alpha=0.4, color=COLORS[i], label=rf'$\mu_e={me}$')
ax_c.set_title(r"C. State PDF ($\sigma_e=0.15$)");
ax_c.legend(frameon=False, fontsize=8)

# --- D: Theoretical Scan mu_e ---
ax_d = fig.add_subplot(gs[1, 0])
mu_scan = np.linspace(0.0, 1.5, 80)
for i, s in enumerate([0.0, 0.4, 0.8]):
    res_p = []
    guess = np.array([-1.0, 1.0])
    for mu in mu_scan:
        guess, p_val = solve_steady_state(mu, s, guess)
        res_p.append(p_val)
    ax_d.plot(mu_scan, res_p, color=COLORS[i], label=rf"$\sigma_e={s}$")
ax_d.set_title(r"D. Analytical $\mu_e$ Scan");
ax_d.set_xlabel(r"$\mu_e$");
ax_d.set_ylabel(r"$\phi$")
ax_d.legend(frameon=False, fontsize=8)

# --- E: Theoretical Scan sigma_e ---
ax_e = fig.add_subplot(gs[1, 1:])
sig_scan = np.linspace(0.0, 1.5, 80)
for i, mu_v in enumerate([0.0, 0.4, 0.8]):
    res_p = []
    guess = np.array([-1.0, 1.0])
    for sig in sig_scan:
        guess, p_val = solve_steady_state(mu_v, sig, guess)
        res_p.append(p_val)
    ax_e.plot(sig_scan, res_p, color=COLORS[i], label=rf"$\mu_e={mu_v}$")
ax_e.set_title(r"E. Analytical $\sigma_e$ Scan");
ax_e.set_xlabel(r"$\sigma_e$")
ax_e.legend(frameon=False, ncol=3, fontsize=8)

plt.tight_layout()
plt.show()
