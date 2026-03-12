import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# ==========================================
# 1. 风格配置
# ==========================================
def set_prl_style():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 130
    })

set_prl_style()
colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

# ==========================================
# 2. 动力学引擎 (区分两体与三体涨落)
# ==========================================
def solve_system_advanced(mu_u, mu_d, mu_e, sigma_d, sigma_e, S=1000, dt=0.02, steps=6000, x0=-0.6):
    x = np.full(S, x0)
    m_h = np.zeros(steps)
    for t in range(steps):
        m = np.mean(x)
        q = np.mean(x ** 2)
        drift = x - x ** 3 + mu_u + mu_d * m + mu_e * q
        # 乘性噪声项
        noise = (sigma_d * m * np.random.normal(0, 1, size=S) +
                 sigma_e * q * np.random.normal(0, 1, size=S))
        x += drift * dt + noise * np.sqrt(dt)
        m_h[t] = m
    return m_h

# ==========================================
# 3. 解析 Phi 计算函数 (保留定义但未使用)
# ==========================================
def calculate_phi_analytical(sd, se, mu_u, mu_d, mu_e, m, q, H_c=0.3849):
    # M 场方程
    M = mu_u + mu_d * m + mu_e * q
    # Gamma 场方程 (乘性噪声叠加)
    Gamma = np.sqrt(max((sd * m)**2 + (se * q)**2, 1e-9))
    # Phi 公式
    return 0.5 * (1 + erf((M - H_c) / (np.sqrt(2) * Gamma)))

# ==========================================
# 4. 绘图：只绘制第一行 (1行2列)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))  # 调整为一行两列

# 公共物理参数
mu_u, mu_d, mu_e = 0.0, 0.5, 0.5
H_c = 0.3849
m_meta, q_meta = -0.6, 0.36  # 亚稳态参考点 (未使用)

# --- 左图：两体相互作用涨落 (Pairwise, sigma_d) ---
for i, sd in enumerate([0.1, 0.5, 1.2]):
    m_h = solve_system_advanced(mu_u, mu_d, mu_e, sd, 0.1)  # 固定弱三体噪声
    axes[0].plot(m_h, color=colors[i], label=rf'$\sigma_d={sd}$')
axes[0].set_title(r"Evolution: Pairwise Fluctuation ($\sigma_d$)")
axes[0].set_ylabel(r"Order Parameter $m$")
axes[0].set_xlabel("Time step")
axes[0].legend(frameon=False)

# --- 右图：三体相互作用涨落 (Triadic, sigma_e) ---
for i, se in enumerate([0.1, 0.5, 1.2]):
    m_h = solve_system_advanced(mu_u, mu_d, mu_e, 0.1, se)  # 固定弱两体噪声
    axes[1].plot(m_h, color=colors[i], label=rf'$\sigma_e={se}$')
axes[1].set_title(r"Evolution: Triadic Fluctuation ($\sigma_e$)")
axes[1].set_xlabel("Time step")
axes[1].legend(frameon=False)

# 可选：添加网格辅助线
for ax in axes:
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()