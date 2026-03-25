import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve
import warnings
import matplotlib as mpl

warnings.filterwarnings("ignore")

# ==========================================
# 1. PRL 投稿級配置 (符合 Physical Review 標準)
# ==========================================
mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 1.0,
    "figure.dpi": 150,
})

# 系統結構參數 (根據 Table 1 定義)
SIG_U = 0.1  # 外部輸入標準差
SIG_D = 0.0  # 線性耦合標準差
SIG_E = 0.0  # 二階耦合標準差 (此處設為0以專注於線性項對比)


# ==========================================
# 2. 核心解析引擎：基於 Kramers 速率理論的自洽求解
# ==========================================
def get_stable_roots(M):
    """精確求解 x - x^3 + M = 0 的穩定分支平衡點"""
    roots = np.roots([-1, 0, 1, M])
    real_roots = np.sort(roots[np.isreal(roots)].real)
    if len(real_roots) == 3:
        return real_roots[0], real_roots[-1]  # x_negative, x_positive
    return (real_roots[0], real_roots[0]) if M < 0 else (real_roots[-1], real_roots[-1])


def solve_system_consistent(mu_d, sig_noise, mu_u=0.0, mu_e=0.0):
    """
    基於 Kramers 速率理論線性近似的自洽模型
    Renormalized Hc = Hc - beta * Gamma
    """
    H_c_det = 0.3849  # 確定性臨界場強
    beta_stochastic = 0.45  # 隨機激活係數 (Stochastic activation coefficient)

    def equations(vars):
        m, q = vars
        # 1. 計算有效平均場 M (Eq. 5)
        M = mu_u + mu_d * m + mu_e * q
        # 2. 計算總有效漲落 Gamma (Eq. 6)
        Gamma = np.sqrt(SIG_U ** 2 + SIG_D ** 2 * q + SIG_E ** 2 * q ** 2 + sig_noise ** 2)

        # 3. Kramers 速率理論線性修正：噪聲誘導的勢壘降低
        H_tilde = H_c_det - beta_stochastic * Gamma

        xn, xp = get_stable_roots(M)

        # 4. 根據重整化閾值計算佔據率 phi (Eq. 9 修正版)
        phi = 0.5 * (1 + erf((M - H_tilde) / (np.sqrt(2) * Gamma)))

        # 5. 自洽殘差方程組
        res_m = m - ((1 - phi) * xn + phi * xp)
        res_q = q - ((1 - phi) * xn ** 2 + phi * xp ** 2)
        return [res_m, res_q]

    # 數值求解
    sol = fsolve(equations, x0=[-0.8, 1.0])
    m_f, q_f = sol

    # 輸出最終物理量用於繪圖
    M_f = mu_u + mu_d * m_f + mu_e * q_f
    Gamma_f = np.sqrt(SIG_U ** 2 + SIG_D ** 2 * q_f + SIG_E ** 2 * q_f ** 2 + sig_noise ** 2)
    xn, xp = get_stable_roots(M_f)
    phi_f = (m_f - xn) / (xp - xn) if abs(xp - xn) > 1e-5 else (1.0 if m_f > 0 else 0.0)

    return m_f, phi_f, xn, xp, Gamma_f


# ==========================================
# 3. 數值模擬 (方案 B：同步節點異質性)
# ==========================================
mu_d_sim_range = np.linspace(-1.2, 0.4, 11)
mu_d_theo_range = np.linspace(-1.2, 0.4, 80)
sig_val = 0.2  # 動力學噪聲強度
S = 1000

m_sim, p_sim = [], []
for md in mu_d_sim_range:
    # 模擬端：體現 Table 1 的結構漲落
    u_vec = np.random.normal(0.0, SIG_U, S)
    d_vec = np.random.normal(md, SIG_D, S)

    x = np.full(S, -1.0)
    for _ in range(7000):  # 增加步數確保臨界區穩定
        m_curr = np.mean(x)
        # 微觀動力學方程 (Eq. 1)
        drift = x - x ** 3 + u_vec + d_vec * m_curr
        x += drift * 0.02 + sig_val * np.random.normal(0, 1, S) * np.sqrt(0.02)

    m_sim.append(np.mean(x))
    p_sim.append(np.sum(x > 0) / S)

# 理論計算
m_th, p_th = [], []
for md in mu_d_theo_range:
    mt, pt, _, _, _ = solve_system_consistent(md, sig_val)
    m_th.append(mt);
    p_th.append(pt)

# ==========================================
# 4. 繪圖與結果驗證
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

# (a) 序參數 m 對比
axes[0].plot(mu_d_sim_range, m_sim, 'ko', mfc='none', markersize=5, label='Simulation')
axes[0].plot(mu_d_theo_range, m_th, 'r-', lw=1.2, label='Theory')
axes[0].set_xlabel(r'Coupling strength $\mu_d$')
axes[0].set_ylabel(r'Order parameter $m$')
axes[0].text(0.05, 0.9, '(a)', transform=axes[0].transAxes, fontweight='bold')
axes[0].legend(frameon=False, loc='lower right')

# (b) 佔據率 phi 對比
axes[1].plot(mu_d_sim_range, p_sim, 'ks', mfc='none', markersize=5)
axes[1].plot(mu_d_theo_range, p_th, 'b-', lw=1.2)
axes[1].set_xlabel(r'Coupling strength $\mu_d$')
axes[1].set_ylabel(r'Occupancy $\phi$')
axes[1].text(0.05, 0.9, '(b)', transform=axes[1].transAxes, fontweight='bold')

# (c) PDF 穩態分佈驗證
md_target = -0.4
mt, pt, xn, xp, Gt = solve_system_consistent(md_target, sig_val)

# 執行 PDF 專項模擬
S_pdf = 5000
u_p, d_p = np.random.normal(0.0, SIG_U, S_pdf), np.random.normal(md_target, SIG_D, S_pdf)
x_p = np.full(S_pdf, -1.0)
for _ in range(8000):
    x_p += (x_p - x_p ** 3 + u_p + d_p * np.mean(x_p)) * 0.02 + sig_val * np.random.normal(0, 1, S_pdf) * np.sqrt(0.02)

axes[2].hist(x_p, bins=50, density=True, color='lightgray', edgecolor='gray', alpha=0.5)
xr = np.linspace(-2.2, 2.2, 300)
vn, vp = Gt ** 2 / (2 * np.abs(1 - 3 * xn ** 2)), Gt ** 2 / (2 * np.abs(1 - 3 * xp ** 2))
pdf_th = (1 - pt) * (1 / np.sqrt(2 * np.pi * vn)) * np.exp(-(xr - xn) ** 2 / (2 * vn)) + pt * (
            1 / np.sqrt(2 * np.pi * vp)) * np.exp(-(xr - xp) ** 2 / (2 * vp))
axes[2].plot(xr, pdf_th, 'r-', lw=1.5)
axes[2].set_xlabel(r'State $x$')
axes[2].set_ylabel(r'$P(x)$')
axes[2].text(0.05, 0.9, '(c)', transform=axes[2].transAxes, fontweight='bold')
axes[2].set_title(rf'$\mu_d = {md_target}$', fontsize=9)

plt.tight_layout()
plt.show()
