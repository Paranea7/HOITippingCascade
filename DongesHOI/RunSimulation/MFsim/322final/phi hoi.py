import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Global PRL Style Configuration
# ==========================================
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 10, "axes.labelsize": 11,
    "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.5, "figure.dpi": 150
})

# Physical Constants from your derivation
SIG_U = 0.128  # Base noise level
H_C = 0.3849  # Critical threshold 2/(3*sqrt(3))
COLORS = ['#1f77b4', '#d62728', '#2ca02c']


# ==========================================
# 2. Strictly Aligned Analytical Engine
# ==========================================
def self_consistent_equations(vars, mu_e, sigma_e):
    """
    Strictly follows your derived equations:
    M = mu_e * m^2
    Gamma = sqrt(SIG_U^2 + sigma_e^2 * q^2)
    x_plus = 1 + M/2, x_minus = -1 + M/2
    """
    m, q = vars
    q = max(q, 1e-6)  # Numerical stability

    # 1. Effective field and noise (Eqs 13-14)
    M = mu_e * (m ** 2)
    # Variance scaling for 2nd order coupling: sigma_e^2 * q^2
    Gamma_sq = SIG_U ** 2 + (sigma_e ** 2 * (q ** 2))
    Gamma = np.sqrt(Gamma_sq)

    # 2. Tipping probability (Eq 18)
    # Using small epsilon to prevent division by zero
    phi = 0.5 * (1 + erf((M - H_C) / (np.sqrt(2) * Gamma + 1e-12)))

    # 3. Branch positions (First-order perturbation: Eqs 15-16)
    xp = 1.0 + M / 2.0
    xn = -1.0 + M / 2.0

    # 4. Self-consistency targets for moments (Weighted average)
    target_m = (1 - phi) * xn + phi * xp
    target_q = (1 - phi) * (xn ** 2) + phi * (xp ** 2)

    return [m - target_m, q - target_q]


def solve_steady_state(mu_e, sigma_e, initial_guess):
    """Solves the system and returns the tipping rate phi."""
    sol = fsolve(self_consistent_equations, x0=initial_guess, args=(mu_e, sigma_e))
    m_f, q_f = sol

    # Re-calculate phi for the final solution
    M_f = mu_e * (m_f ** 2)
    G_f = np.sqrt(SIG_U ** 2 + (sigma_e ** 2 * (q_f ** 2)))
    phi_f = 0.5 * (1 + erf((M_f - H_C) / (np.sqrt(2) * G_f + 1e-12)))

    return sol, phi_f


# ==========================================
# 3. Scanning Logic (Hysteresis/Path Tracking)
# ==========================================
fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.8))
plt.subplots_adjust(wspace=0.3)

mu_axis = np.linspace(0.0, 1.5, 60)
sig_axis = np.linspace(0.0, 1.5, 60)
sig_samples = [0.0, 0.4, 0.8]
mu_samples = [0.0, 0.3, 0.5]

# --- Left Plot: Scan mu_e (Starting from Negative Branch) ---
for i, s in enumerate(sig_samples):
    results_mu = []
    # Start guess at the negative equilibrium m=-1, q=1
    curr_sol = np.array([-1.0, 1.0])
    for mu in mu_axis:
        curr_sol, p_val = solve_steady_state(mu, s, curr_sol)
        results_mu.append(p_val)
    axs[0].plot(mu_axis, results_mu, color=COLORS[i], label=rf"$\sigma_e={s}$")

# --- Right Plot: Scan sigma_e ---
for i, mu_v in enumerate(mu_samples):
    results_sig = []
    # Start guess at negative branch to observe noise-induced tipping
    curr_sol = np.array([-1.0, 1.0])
    for sig in sig_axis:
        curr_sol, p_val = solve_steady_state(mu_v, sig, curr_sol)
        results_sig.append(p_val)
    axs[1].plot(sig_axis, results_sig, color=COLORS[i], label=rf"$\mu_e={mu_v}$")

# Formatting
for ax in axs:
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend(frameon=False, loc='best', fontsize=9)
    ax.set_ylabel(r'Tipping Rate $\phi$')
    ax.set_ylim(-0.05, 1.05)

axs[0].set_xlabel(r'Coupling $\mu_e$')
axs[1].set_xlabel(r'Heterogeneity $\sigma_e$')

plt.tight_layout()
plt.show()
