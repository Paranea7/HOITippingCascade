import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 物理參數與 PRL 風格
# ==========================================
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 10, "axes.labelsize": 11,
    "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.5, "figure.dpi": 150
})

SIG_U, H_C = 0.12, 0.3849
colors = ['#1f77b4', '#d62728', '#2ca02c']


# ==========================================
# 2. 核心物理邏輯：確保回傳為 float
# ==========================================
def get_roots_exact(M):
    """精確求解穩定分支，確保回傳數值而非列表"""
    roots = np.roots([1, 0, -1, -float(M)])
    real_v = np.sort(roots[np.isreal(roots)].real)

    if len(real_v) == 3:
        # 有三個實根時，取最小(xn)和最大(xp)
        return float(real_v[0]), float(real_v[-1])
    else:
        # 只有一個實根時，xn = xp = 該唯一實根
        val = float(real_v[0])
        return val, val


def self_consistent_HOI(vars, mu_e, sigma_e):
    m, q = vars
    M = mu_e * (m ** 2)
    Gamma = np.sqrt(SIG_U ** 2 + sigma_e * q ** 2)

    phi = 0.5 * (1 + erf((M - H_C) / (np.sqrt(2) * Gamma)))
    xn, xp = get_roots_exact(M)

    # 此處 xn, xp 已確保為 float，可直接運算
    res_m = m - ((1 - phi) * xn + phi * xp)
    res_q = q - ((1 - phi) * (xn ** 2) + phi * (xp ** 2))
    return [res_m, res_q]


def solve_HOI_system(mu_e, sigma_e):
    """雙起點求解確保 mu_e=0.55 穩定"""
    # 嘗試從負分支找
    sol_n, info_n, ier_n, _ = fsolve(self_consistent_HOI, x0=[-1.0, 1.0], args=(mu_e, sigma_e), full_output=True)
    # 嘗試從正分支找
    sol_p, info_p, ier_p, _ = fsolve(self_consistent_HOI, x0=[1.0, 1.0], args=(mu_e, sigma_e), full_output=True)

    # 根據 mu_e 大小選擇物理分支
    if mu_e > 0.4:
        sol = sol_p if ier_p == 1 else sol_n
    else:
        sol = sol_n if ier_n == 1 else sol_p

    m_f, q_f = sol
    M_f = mu_e * (m_f ** 2)
    G_f = np.sqrt(SIG_U ** 2 + (sigma_e * q_f) ** 2)
    phi_f = 0.5 * (1 + erf((M_f - H_C) / (np.sqrt(2) * G_f)))
    return m_f, phi_f


# ==========================================
# 3. 數據生成與 2x2 繪圖
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(8.5, 6.5))
plt.subplots_adjust(wspace=0.3, hspace=0.35)

mu_axis = np.linspace(0.0, 1.2, 60)
sig_axis = np.linspace(0.01, 1.5, 60)
sig_samples = [0.05, 0.2, 0.4]
mu_samples = [0.2, 0.4, 0.55]

# --- (a, c) Scan mu_e ---
for i, s in enumerate(sig_samples):
    ms, ps = [], []
    for mu in mu_axis:
        m, p = solve_HOI_system(mu, s)
        ms.append(m);
        ps.append(p)
    axs[0, 0].plot(mu_axis, ms, color=colors[i], label=rf"$\sigma_e={s}$")
    axs[1, 0].plot(mu_axis, ps, color=colors[i])

# --- (b, d) Scan sigma_e ---
for i, mu_v in enumerate(mu_samples):
    ms, ps = [], []
    for sig in sig_axis:
        m, p = solve_HOI_system(mu_v, sig)  # 調用穩定求解器
        ms.append(m);
        ps.append(p)
    axs[0, 1].plot(sig_axis, ms, color=colors[i], label=rf"$\mu_e={mu_v}$")
    axs[1, 1].plot(sig_axis, ps, color=colors[i])


def solve_triadic_consistent_v2(mu_e, sigma_e):
    # 針對掃描 sigma_e 進行路徑優化
    m, p = solve_HOI_system(mu_e, sigma_e)
    return m, p


# 修飾
labels = ['(a)', '(b)', '(c)', '(d)']
for i, ax in enumerate(axs.flat):
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend(frameon=False, loc='best', fontsize=8)
    ax.text(-0.15, 1.1, labels[i], transform=ax.transAxes, fontweight='bold')

axs[0, 0].set_ylabel(r'$m$')
axs[1, 0].set_ylabel(r'$\phi$')
axs[1, 0].set_xlabel(r'$\mu_e$')
axs[0, 1].set_xlabel(r'$\sigma_e$')
axs[1, 1].set_xlabel(r'$\sigma_e$')

plt.tight_layout()
plt.show()
