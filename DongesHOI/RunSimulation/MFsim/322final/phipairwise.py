import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve
import warnings

warnings.filterwarnings("ignore")


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
SIG_E = 0.05  # 理论推导中包含的三体涨落
H_C = 0.3849  # 临界阈值 2/(3*sqrt(3))


# ==========================================
# 2. 核心解析工具 (包含截断逻辑)
# ==========================================
def get_x_roots_with_truncation(M):
    """
    求解 x^3 - x = M 的稳定根。
    若 M > H_c，负分支消失，返回 (NaN, xp)
    """
    roots = np.roots([1, 0, -1, -M])
    real_roots = np.sort(roots[np.isreal(roots)].real)

    # 截断机制：只有当 M <= H_c 且存在三个实根时，负分支才存在
    if len(real_roots) == 3 and M <= H_C:
        return real_roots[0], real_roots[-1]
    else:
        # 确定性跳变区：负分支消失，只剩正分支
        return np.nan, real_roots[-1]


def solve_theory_unified(mu_d, sig_d, mu_e=0.0, sig_e=0.0, x0_guess=[-1.0, 1.0]):
    def equations(vars):
        m, q = vars
        # 1. 有效场均值 (m^2 逻辑)
        M = mu_d * m + mu_e * (m ** 2)

        # 2. 有效场总方差 (修正后的自洽方程：包含 sigma_e^2 * q^2)
        Gamma2 = SIG_U ** 2 + (sig_d ** 2) * q + (sig_e ** 2) * (q ** 2)
        Gamma = np.sqrt(Gamma2)

        # 3. 根的获取与截断判断
        xn, xp = get_x_roots_with_truncation(M)

        if np.isnan(xn):
            # 确定性跳变：phi 强制为 1
            phi = 1.0
            target_m = xp
            target_q = xp ** 2
        else:
            # 噪声诱导跳变：计算 erf 概率
            phi = 0.5 * (1 + erf((M - H_C) / (np.sqrt(2) * Gamma)))
            target_m = (1 - phi) * xn + phi * xp
            target_q = (1 - phi) * (xn ** 2) + phi * (xp ** 2)

        return [m - target_m, q - target_q]

    # 解自洽方程组
    sol = fsolve(equations, x0=x0_guess, xtol=1e-9)
    m_f, q_f = sol

    # 重新计算最终的 phi 以便绘图
    M_f = mu_d * m_f + mu_e * m_f ** 2
    if M_f > H_C:
        phi_f = 1.0
    else:
        G_f = np.sqrt(SIG_U ** 2 + sig_d ** 2 * q_f + sig_e ** 2 * q_f ** 2)
        phi_f = 0.5 * (1 + erf((M_f - H_C) / (np.sqrt(2) * G_f)))

    return m_f, q_f, phi_f


# ==========================================
# 3. 绘图任务 (1x2 布局)
# ==========================================
fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.5), dpi=150)
plt.subplots_adjust(wspace=0.3)

mu_range = np.linspace(-1.2, 0.6, 100)
sig_range = np.linspace(0.05, 1.5, 100)
sig_samples = [0.1, 0.25, 0.35]
mu_samples = [-0.6, -0.2, 0.4]

# --- 左图：扫描 mu_d (均值效应) ---
for i, s in enumerate(sig_samples):
    p_l = []
    curr_guess = [-1.0, 1.0]  # 初始于负分支
    for mu in mu_range:
        m, q, p = solve_theory_unified(mu, s, sig_e=SIG_E, x0_guess=curr_guess)
        p_l.append(p)
        curr_guess = [m, q]
    axs[0].plot(mu_range, p_l, color=colors[i], label=rf"$\sigma_d={s}$")

# --- 右图：扫描 sigma_d (异构性效应) ---
for i, m_val in enumerate(mu_samples):
    p_l = []
    curr_guess = [-1.0, 1.0]
    for sig in sig_range:
        m, q, p = solve_theory_unified(m_val, sig, sig_e=SIG_E, x0_guess=curr_guess)
        p_l.append(p)
        curr_guess = [m, q]
    axs[1].plot(sig_range, p_l, color=colors[i], label=rf"$\mu_d={m_val}$")

# 图形美化
sub_labels = ['(a)', '(b)']
for i, ax in enumerate(axs):
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend(frameon=False, loc='best', fontsize=8)
    ax.text(-0.18, 1.05, sub_labels[i], transform=ax.transAxes, fontweight='bold', fontsize=12)
    ax.set_ylabel(r'Tipping Rate $\phi$')

axs[0].set_xlabel(r'Coupling $\mu_d$')
axs[1].set_xlabel(r'Heterogeneity $\sigma_d$')

plt.tight_layout()
plt.show()
