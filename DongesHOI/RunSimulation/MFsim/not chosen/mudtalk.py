import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve
import matplotlib as mpl

# ==========================================
# 1. PRL 投稿級配置
# ==========================================
mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.linewidth": 1.0,
    "figure.dpi": 150,
})

# 系統核心參數
SIG_U, SIG_D, SIG_E = 0.12, 0.20, 0.15
H_C, S = 0.3849, 5000

# ==========================================
# 2. 核心解析引擎：精確根與動態方差
# ==========================================
def get_x_roots_exact(M):
    """求解 x^3 - x - M = 0 的精確物理根"""
    roots = np.roots([1, 0, -1, -M])
    real_roots = np.sort(roots[np.isreal(roots)].real)
    if len(real_roots) == 3:
        return real_roots[0], real_roots[-1] # 返回負分支 xn, 正分支 xp
    return (real_roots[0], real_roots[0]) if M < 0 else (real_roots[-1], real_roots[-1])

def solve_theory_unified(mu_d, mu_e=0.0):
    def equations(vars):
        m, q = vars
        M = mu_d * m + mu_e * (m ** 2)
        Gamma = np.sqrt(max(SIG_U**2 + SIG_D**2 * q + SIG_E**2 * q**2, 1e-10))
        phi = 0.5 * (1 + erf((M - H_C) / (np.sqrt(2) * Gamma)))
        xn, xp = get_x_roots_exact(M)
        # 序參量與二階矩的自洽方程
        res_m = m - ((1 - phi) * xn + phi * xp)
        res_q = q - ((1 - phi) * xn**2 + phi * xp**2)
        return [res_m, res_q]

    sol = fsolve(equations, x0=[-1.0, 1.0])
    m_f, q_f = sol
    M_f = mu_d * m_f + mu_e * (m_f ** 2)
    G_f = np.sqrt(max(SIG_U**2 + SIG_D**2 * q_f + SIG_E**2 * q_f**2, 1e-10))
    phi_f = 0.5 * (1 + erf((M_f - H_C) / (np.sqrt(2) * G_f)))
    return m_f, phi_f

# ==========================================
# 3. 微觀模擬與理論計算
# ==========================================
mu_d_range = np.linspace(-1.0, 0.5, 15)
m_sim, p_sim = [], []

for md in mu_d_range:
    # 靜態噪聲生成
    gamma_fixed = np.sqrt(SIG_U**2 + SIG_D**2 * 1.0 + SIG_E**2 * 1.0)
    xi_i = np.random.normal(0, gamma_fixed, S)
    x, dt = np.full(S, -1.0), 0.05
    for _ in range(4000):
        m_c = np.mean(x)
        drift = x - x**3 + (md * m_c) + xi_i
        x += drift * dt
        x = np.clip(x, -2.5, 2.5)
    m_sim.append(np.mean(x)); p_sim.append(np.sum(x > 0) / S)

mu_d_theo = np.linspace(-1.0, 0.5, 100)
m_th, p_th = zip(*[solve_theory_unified(md) for md in mu_d_theo])

# 繪圖
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
axes[0].plot(mu_d_range, m_sim, 'ko', mfc='none', label='Simulation')
axes[0].plot(mu_d_theo, m_th, 'r-', label='Theory')
axes[1].plot(mu_d_range, p_sim, 'ks', mfc='none')
axes[1].plot(mu_d_theo, p_th, 'b-')
for ax in axes: ax.set_xlabel(r'$\mu_d$'); ax.grid(True, ls=':', alpha=0.6); ax.legend(frameon=False)
plt.tight_layout(); plt.show()
