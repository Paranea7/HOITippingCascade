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

# 系統核心參數 (嚴格對齊你原始程序的設定)
SIG_U = 0.12
SIG_D = 0.05
SIG_E = 0.0
H_C = 0.3849

# ==========================================
# 2. 核心解析工具 (精確根)
# ==========================================
def get_x_roots_exact(M):
    roots = np.roots([1, 0, -1, -M])
    real_roots = np.sort(roots[np.isreal(roots)].real)
    if len(real_roots) == 3:
        return real_roots[0], real_roots[-1] # 負根 xn, 正根 xp
    return (real_roots[0], real_roots[0]) if M < 0 else (real_roots[-1], real_roots[-1])

def solve_theory_unified(mu_d, sig_d, mu_e=0.0, x0_guess=[-1.0, 1.0]):
    """
    對齊計算邏輯：
    M = mu_d * m + mu_e * q (若無三體則 mu_e=0)
    Gamma = sqrt(SIG_U**2 + sig_d**2 * q)
    """
    def equations(vars):
        m, q = vars
        M = mu_d * m + mu_e * q
        Gamma = np.sqrt(SIG_U**2 + sig_d**2 * q)
        phi = 0.5 * (1 + erf((M - H_C) / (np.sqrt(2) * Gamma)))
        xn, xp = get_x_roots_exact(M)
        res_m = m - ((1 - phi) * xn + phi * xp)
        res_q = q - ((1 - phi) * xn**2 + phi * xp**2)
        return [res_m, res_q]

    sol = fsolve(equations, x0=x0_guess, xtol=1e-9)
    m_f, q_f = sol
    M_f = mu_d * m_f + mu_e * q_f
    G_f = np.sqrt(SIG_U**2 + sig_d**2 * q_f)
    phi_f = 0.5 * (1 + erf((M_f - H_C) / (np.sqrt(2) * G_f)))
    return m_f, q_f, phi_f

# ==========================================
# 3. 2x2 繪圖任務 (依據原始 AB 圖邏輯修正)
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(8.5, 6), dpi=150)
plt.subplots_adjust(wspace=0.3, hspace=0.35)

# 參數設定
mu_range = np.linspace(-1.2, 0.6, 100)  # 對齊 AB 圖的跳變區間
sig_range = np.linspace(0.05, 1.5, 100)
sig_samples = [0.1, 0.25, 0.4]          # 對齊 AB 圖的 sigma 設置
mu_samples = [-0.6, -0.2, 0.2]          # 修正 samples 確保落在跳變敏感區

# --- 左列：Scan mu_d (a, c) ---
for i, s in enumerate(sig_samples):
    m_l, p_l = [], []
    curr_guess = [-1.0, 1.0] # 初始路徑追蹤
    for mu in mu_range:
        m, q, p = solve_theory_unified(mu, s, x0_guess=curr_guess)
        m_l.append(m)
        p_l.append(p)
        curr_guess = [m, q]
    axs[0, 0].plot(mu_range, m_l, color=colors[i], label=rf"$\sigma_d={s}$")
    axs[1, 0].plot(mu_range, p_l, color=colors[i])

# --- 右列：Scan sigma_d (b, d) ---
for i, m_val in enumerate(mu_samples):
    m_l, p_l = [], []
    curr_guess = [-1.0, 1.0]
    for sig in sig_range:
        m, q, p = solve_theory_unified(m_val, sig, x0_guess=curr_guess)
        m_l.append(m)
        p_l.append(p)
        curr_guess = [m, q]
    axs[0, 1].plot(sig_range, m_l, color=colors[i], label=rf"$\mu_d={m_val}$")
    axs[1, 1].plot(sig_range, p_l, color=colors[i])

# 圖形美化
sub_labels = ['(a)', '(b)', '(c)', '(d)']
for i, ax in enumerate(axs.flat):
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend(frameon=False, loc='best', fontsize=8)
    ax.text(-0.18, 1.05, sub_labels[i], transform=ax.transAxes, fontweight='bold', fontsize=12)

axs[0, 0].set_ylabel(r'Order Parameter $m$')
axs[1, 0].set_ylabel(r'Tipping Rate $\phi$')
axs[1, 0].set_xlabel(r'Coupling $\mu_d$')
axs[1, 1].set_xlabel(r'Heterogeneity $\sigma_d$')

plt.tight_layout()
plt.show()
