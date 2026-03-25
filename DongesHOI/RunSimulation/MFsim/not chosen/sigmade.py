import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


# ==========================================
# 1. 风格配置 (PRL Style)
# ==========================================
def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 130,
        "lines.linewidth": 1.2
    })


set_prl_style()
colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']


# ==========================================
# 2. 动力学引擎 (M = mu_e * m^2, 初值 -1.0)
# ==========================================
def solve_system_advanced(mu_u, mu_d, mu_e, sigma_d, sigma_e, S=1000, dt=0.02, steps=6000, x0=-1.0):
    x = np.full(S, x0, dtype=float)
    m_h = np.zeros(steps)

    for t in range(steps):
        m = np.mean(x)
        q = np.mean(x ** 2)

        # 核心逻辑修正：M = mu_u + mu_d * m + mu_e * m^2
        # 注意：m^2 会抵消负值，增强系统的不稳定性
        drift = x - x ** 3 + mu_u + mu_d * m + mu_e * (m ** 2)

        # 乘性噪声 Gamma 计算逻辑 (Multiplicative Noise)
        # 噪声强度 sqrt(dt) * (sigma_d * m * eta + sigma_e * q * eta)
        noise = (sigma_d * m * np.random.normal(0, 1, size=S) +
                 sigma_e * q * np.random.normal(0, 1, size=S))

        x += drift * dt + noise * np.sqrt(dt)
        m_h[t] = m
    return m_h


# ==========================================
# 3. 解析 Phi 计算 (包含 Gamma 平方逻辑与 q 迭代)
# ==========================================
def calculate_phi_analytical(sd, se, mu_u, mu_d, mu_e, m_val, H_c=0.3849):
    # 1. 均值场 M 修正
    M = mu_u + mu_d * m_val + mu_e * (m_val ** 2)

    # 2. q 的迭代逻辑：在稳态附近 q = m^2 + Var(x)
    # 对于乘性噪声系统，Var(x) 随噪声强度增加。近似公式：
    q_approx = m_val ** 2 + (sd ** 2 + se ** 2) * 0.1

    # 3. Gamma 计算逻辑：基于乘性噪声的方差叠加原则
    # Gamma^2 = (sigma_d * m)^2 + (sigma_e * q)^2
    Gamma = np.sqrt(max((sd * m_val) ** 2 + (se * q_approx) ** 2, 1e-9))

    # 4. Phi 计算
    phi = 0.5 * (1 + erf((M - H_c) / (np.sqrt(2) * Gamma)))
    return phi


# ==========================================
# 4. 绘图任务：2x2 矩阵布局
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# 公共物理参数
mu_u, mu_d, mu_e = 0.0, 0.4, 0.5
H_c = 0.3849
m_start = -1.0
sigma_axis = np.linspace(0.01, 1.5, 100)

# --- 第一列：两体相互作用涨落 (Pairwise, sigma_d) ---
for i, sd in enumerate([0.2, 0.6, 1.2]):
    m_h = solve_system_advanced(mu_u, mu_d, mu_e, sd, 0.05, x0=m_start)
    axes[0, 0].plot(m_h, color=colors[i], label=rf'$\sigma_d={sd}$')
axes[0, 0].set_title(r"Evolution: Pairwise ($\sigma_d$) with $M=\mu_e m^2$")
axes[0, 0].set_ylabel(r"Order Parameter $m$")
axes[0, 0].legend(frameon=False, fontsize=8)

phi_d = [calculate_phi_analytical(sd, 0.05, mu_u, mu_d, mu_e, m_start, H_c) for sd in sigma_axis]
axes[1, 0].plot(sigma_axis, phi_d, color='teal', lw=2)
axes[1, 0].set_ylabel(r"Analytical Tipping Rate $\phi$")
axes[1, 0].set_xlabel(r"Pairwise Noise $\sigma_d$")

# --- 第二列：三体相互作用涨落 (Triadic, sigma_e) ---
for i, se in enumerate([0.2, 0.6, 1.2]):
    m_h = solve_system_advanced(mu_u, mu_d, mu_e, 0.05, se, x0=m_start)
    axes[0, 1].plot(m_h, color=colors[i], label=rf'$\sigma_e={se}$')
axes[0, 1].set_title(r"Evolution: Triadic ($\sigma_e$) with $M=\mu_e m^2$")
axes[0, 1].legend(frameon=False, fontsize=8)

phi_e = [calculate_phi_analytical(0.05, se, mu_u, mu_d, mu_e, m_start, H_c) for se in sigma_axis]
axes[1, 1].plot(sigma_axis, phi_e, color='indianred', lw=2)
axes[1, 1].set_xlabel(r"Triadic Noise $\sigma_e$")

# 统一细节美化
for ax in axes.flat:
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.axhline(-1.0, ls=':', color='black', alpha=0.3)
for ax in axes[1, :]:
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, ls='--', color='gray', alpha=0.5)

plt.tight_layout()
plt.show()
