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
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 1.5,
        "figure.dpi": 150
    })

set_prl_style()
colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

# System constants (No coupling)
MU_D, SIGMA_D = 0.0, 0.0
MU_E, SIGMA_E = 0.0, 0.0
HC = 0.3849

def equations(vars, mu_u, sigma_u):
    m, q, phi = vars
    M = mu_u + MU_D * m + MU_E * m**2
    Gamma2 = max(sigma_u**2 + SIGMA_D**2 * q + SIGMA_E**2 * q**2, 1e-6)
    res_phi = 0.5 * (1 + erf((M - HC) / (np.sqrt(2 * Gamma2)))) - phi
    res_m = (2 * phi - 1) + M/2 - m
    res_q = 1 + (2 * phi - 1) * M + (M**2 + Gamma2)/4 - q
    return [res_m, res_q, res_phi]

def solve_m(mu_range, sigma_u):
    m_list = []
    guess = [-1.0, 1.0, 0.0]
    for mu in mu_range:
        sol = fsolve(equations, guess, args=(mu, sigma_u))
        m_list.append(sol[0])
        guess = sol
    return np.array(m_list)

# ==========================================
# 2. Data Generation
# ==========================================
# Global Range
mu_global = np.linspace(-1.0, 2.0, 500)
m_global = solve_m(mu_global, 0.1)

# Zoom-in Range (Proof of Continuity)
mu_zoom = np.linspace(HC - 0.05, HC + 0.05, 800)
m_zoom = solve_m(mu_zoom, 0.00001)

# Numerical Derivative
dm_dmu = np.gradient(m_zoom, mu_zoom)

# Comparison of Noise
m_noisy = solve_m(mu_global, 0.5)

# ==========================================
# 3. Plotting (2x2 Layout)
# ==========================================
fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(8, 6))

# (a) Global Sweep - Looks like a jump
axs[0, 0].plot(mu_global, m_global, color=colors[0])
axs[0, 0].set_ylabel(r'Order Parameter $m$')
axs[0, 0].set_xlabel(r'$\mu_u$')
axs[0, 0].set_title('(a) Global Evolution')

# (b) Zoom-in - Reveals smooth S-curve
axs[0, 1].plot(mu_zoom, m_zoom, color=colors[0], marker='.', markersize=2)
axs[0, 1].set_ylabel(r'$m$')
axs[0, 1].set_xlabel(r'Local $\mu_u \approx H_c$')
axs[0, 1].set_title('(b) Zoom-in (Proof of Continuity)')
axs[0, 1].grid(alpha=0.2)

# (c) Numerical Derivative - Bell curve (Finite & Continuous)
axs[1, 0].plot(mu_zoom, dm_dmu, color=colors[1])
axs[1, 0].set_ylabel(r'$dm/d\mu_u$')
axs[1, 0].set_xlabel(r'$\mu_u$')
axs[1, 0].set_title('(c) Numerical Derivative')
axs[1, 0].fill_between(mu_zoom, dm_dmu, color=colors[1], alpha=0.1)

# (d) Noise Comparison - Smoothing the crossover
axs[1, 1].plot(mu_global, m_global, label=r'$\sigma_u=0.1$')
axs[1, 1].plot(mu_global, m_noisy, label=r'$\sigma_u=0.5$', linestyle='--')
axs[1, 1].set_ylabel(r'$m$')
axs[1, 1].set_xlabel(r'$\mu_u$')
axs[1, 1].set_title('(d) Role of Noise Intensity')
axs[1, 1].legend(frameon=False)

plt.show()
