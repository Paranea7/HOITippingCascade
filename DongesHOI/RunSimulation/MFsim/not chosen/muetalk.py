import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve
import matplotlib as mpl

# PRL 投稿級配置
mpl.rcParams.update({
    "mathtext.fontset": "stix", "font.family": "STIXGeneral", "font.size": 10,
    "axes.labelsize": 11, "xtick.direction": "in", "ytick.direction": "in", "figure.dpi": 150
})

SIG_U, SIG_E, H_C, S = 0.12, 0.25, 0.3849, 5000


def get_x_roots_exact(M):
    """精確求解 x^3 - x - M = 0 的穩定物理分支"""
    roots = np.roots([1, 0, -1, -M])
    real_roots = np.sort(roots[np.isreal(roots)].real)
    if len(real_roots) == 3:
        return real_roots[0], real_roots[-1]  # 負分支 xn, 正分支 xp
    return (real_roots[0], real_roots[0]) if M < 0 else (real_roots[-1], real_roots[-1])


def solve_theory_unified(mu_e):
    """統一精確解析引擎"""

    def equations(vars):
        m, q = vars
        M = mu_e * (m ** 2)
        Gamma = np.sqrt(SIG_U ** 2 + SIG_E ** 2 * (q ** 2))
        phi = 0.5 * (1 + erf((M - H_C) / (np.sqrt(2) * Gamma)))
        xn, xp = get_x_roots_exact(M)
        res_m = m - ((1 - phi) * xn + phi * xp)
        res_q = q - ((1 - phi) * xn ** 2 + phi * xp ** 2)
        return [res_m, res_q]

    # 從負分支啟動
    sol = fsolve(equations, x0=[-1.0, 1.0])
    m_f, q_f = sol
    M_f = mu_e * (m_f ** 2)
    G_f = np.sqrt(SIG_U ** 2 + SIG_E ** 2 * (q_f ** 2))
    phi_f = 0.5 * (1 + erf((M_f - H_C) / (np.sqrt(2) * G_f)))
    return m_f, phi_f


# 微觀模擬
mu_e_range = np.linspace(0.0, 0.8, 15)
m_sim, p_sim = [], []
for me in mu_e_range:
    # 靜態異質場 xi_i (Quenched Disorder)
    xi_i = np.random.normal(0, np.sqrt(SIG_U ** 2 + SIG_E ** 2), S)
    x, dt = np.full(S, -1.0), 0.05
    for _ in range(4000):
        m_c = np.mean(x)
        drift = x - x ** 3 + (me * m_c ** 2) + xi_i
        x += drift * dt
        x = np.clip(x, -2.5, 2.5)
    m_sim.append(np.mean(x));
    p_sim.append(np.sum(x > 0) / S)

# 理論計算
mu_e_theo = np.linspace(0.0, 0.8, 100)
m_th, p_th = zip(*[solve_theory_unified(me) for me in mu_e_theo])

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
axes[0].plot(mu_e_range, m_sim, 'ko', mfc='none', label='Simulation')
axes[0].plot(mu_e_theo, m_th, 'r-', label='Theory')
axes[1].plot(mu_e_range, p_sim, 'ks', mfc='none')
axes[1].plot(mu_e_theo, p_th, 'b-')
axes[0].set_ylabel(r'$m$');
axes[1].set_ylabel(r'$\phi$')
for ax in axes: ax.set_xlabel(r'$\mu_e$'); ax.grid(True, ls=':', alpha=0.6); ax.legend(frameon=False)
plt.tight_layout();
plt.show()
