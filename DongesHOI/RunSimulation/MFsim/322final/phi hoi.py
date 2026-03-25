import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve
import warnings

warnings.filterwarnings('ignore')

# PRL Style
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 10, "axes.labelsize": 11,
    "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.5, "figure.dpi": 150
})

SIG_U, H_C = 0.12, 0.3849
colors = ['#1f77b4', '#d62728', '#2ca02c']


def get_roots_exact(M):
    """精确求解势能面平衡点"""
    # 限制 M 的大小防止溢出
    M = np.clip(M, -2.0, 2.0)
    roots = np.roots([1, 0, -1, -float(M)])
    real_v = np.sort(roots[np.isreal(roots)].real)
    if len(real_v) == 3:
        return float(real_v[0]), float(real_v[-1])
    return float(real_v[0]), float(real_v[0])


def self_consistent_HOI(vars, mu_e, sigma_e):
    m, q = vars
    # 物理约束：q 必须大于 0
    q = max(q, 1e-5)

    M = mu_e * q
    # 噪声由基础噪声和异质性(sigma_e)组成
    Gamma = np.sqrt(SIG_U ** 2 + (sigma_e * q) ** 2)

    # 计算跳跃概率
    phi = 0.5 * (1 + erf((M - H_C) / (np.sqrt(2) * Gamma)))

    xn, xp = get_roots_exact(M)

    # 自洽反馈：包含分布宽度 Gamma^2 修正
    target_m = (1 - phi) * xn + phi * xp
    target_q = (1 - phi) * (xn ** 2) + phi * (xp ** 2) + Gamma ** 2

    return [m - target_m, q - target_q]


def scan_phi(mu_range, sigma_e):
    """使用路径追踪法确保数值稳定性"""
    results = []
    # 初始猜测：系统处于负态
    last_sol = np.array([-0.8, 0.6])
    for mu in mu_range:
        sol, info, ier, msg = fsolve(self_consistent_HOI, x0=last_sol, args=(mu, sigma_e), full_output=True)
        if ier == 1:
            last_sol = sol

        m_f, q_f = last_sol
        M_f = mu * q_f
        G_f = np.sqrt(SIG_U ** 2 + (sigma_e * q_f) ** 2)
        phi_f = 0.5 * (1 + erf((M_f - H_C) / (np.sqrt(2) * G_f)))
        results.append(phi_f)
    return results


# ==========================================
# 3. 绘图与分析
# ==========================================
fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.8))
plt.subplots_adjust(wspace=0.3)

mu_axis = np.linspace(0.0, 1.5, 50)
sig_axis = np.linspace(0.0, 1.2, 50)
sig_samples = [0.05, 0.4, 0.8]
mu_samples = [0.4, 0.7, 1.0]

# --- 左图：Scan mu_e (观察一级相变的平滑化) ---
for i, s in enumerate(sig_samples):
    phi_vals = scan_phi(mu_axis, s)
    axs[0].plot(mu_axis, phi_vals, color=colors[i], label=rf"$\sigma_e={s}$")

# --- 右图：Scan sigma_e ---
for i, mu_v in enumerate(mu_samples):
    # 这里需要反向扫描 sigma_e 轴
    results_sig = []
    last_sol = np.array([-0.8, 0.6])
    for sig in sig_axis:
        sol, _, ier, _ = fsolve(self_consistent_HOI, x0=last_sol, args=(mu_v, sig), full_output=True)
        if ier == 1: last_sol = sol
        m_f, q_f = last_sol
        G_f = np.sqrt(SIG_U ** 2 + (sig * q_f) ** 2)
        phi_f = 0.5 * (1 + erf((mu_v * q_f - H_C) / (np.sqrt(2) * G_f)))
        results_sig.append(phi_f)
    axs[1].plot(sig_axis, results_sig, color=colors[i], label=rf"$\mu_e={mu_v}$")

# 修饰
for i, ax in enumerate(axs):
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend(frameon=False, loc='best', fontsize=9)
    ax.set_ylabel(r'Tipping Rate $\phi$')
    ax.set_ylim(-0.05, 1.05)

axs[0].set_xlabel(r'Coupling $\mu_e$')
axs[1].set_xlabel(r'Heterogeneity $\sigma_e$')
plt.tight_layout()
plt.show()
