import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# ==========================================
# 1. 全局 PRL 风格配置
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
        "lines.linewidth": 1.2,
        "figure.dpi": 150
    })

set_prl_style()
colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

# ==========================================
# 2. 核心动力学引擎 (Euler-Maruyama)
# ==========================================
def solve_system(mu_u, mu_d, mu_e, sigma, S=1000, dt=0.02, steps=3000, x0=-0.6):
    # 模拟群体动力学
    x = np.full(S, x0) + np.random.normal(0, 0.05, S)
    m_h = np.zeros(steps)
    q_h = np.zeros(steps)
    for t in range(steps):
        m = np.mean(x)
        q = np.mean(x ** 2)
        # 漂移项包含线性耦合 mu_d 和三体协同 mu_e
        drift = x - x ** 3 + mu_u + mu_d * m + mu_e * q
        x += drift * dt + sigma * np.random.normal(0, 1, size=S) * np.sqrt(dt)
        m_h[t] = m
        q_h[t] = q
    return m_h, q_h, x

# ==========================================
# 3. 解析函数定义 (对齐自洽场理论)
# ==========================================

def calculate_analytical_phi(mu_u, mu_d, mu_e, m, q, sigma_base, H_c=0.3849):
    """
    计算解析跳变率 phi
    M: 有效平均场
    Gamma: 总有效涨落强度
    """
    # 场方程涨落系数 (设定为模型固有参数)
    sigma_u, sigma_d, sigma_e = 0.1, 0.1, 0.1

    # Eq 2: Effective Field M (必须包含 mu_d 和 mu_e)
    M = mu_u + mu_d * m + mu_e * q

    # Eq 3: Fluctuation Strength Gamma
    # 采用能量叠加叠加原理对齐模拟中的 sigma
    G2 = (sigma_u**2 + sigma_d**2 * q + sigma_e**2 * q**2) + sigma_base**2
    Gamma = np.sqrt(G2)

    # Eq 1: Tipping Probability
    phi = 0.5 * (1 + erf((M - H_c) / (np.sqrt(2) * Gamma)))
    return phi

# ==========================================
# 4. 绘图任务 A, B, C, D & E
# ==========================================
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 3)

# --- A. 准静态参数扫描 (mu_e Sweep) ---
ax_a = fig.add_subplot(gs[0, 0])
mu_e_range = np.linspace(0.0, 1.8, 20)
mu_d_fixed = 0.5
for i, sig in enumerate([0.1, 0.35]):
    # 增加 steps 确保达到稳态
    ms = [np.mean(solve_system(0.0, mu_d_fixed, me, sig, steps=4000)[0][-800:]) for me in mu_e_range]
    ax_a.plot(mu_e_range, ms, 'o-', mfc='none', color=colors[i], label=rf'$\sigma={sig}$')
ax_a.set_xlabel(r'$\mu_e$')
ax_a.set_ylabel(r'$m$')
ax_a.set_title(r"A. Simulation $\mu_e$ Sweep")
ax_a.legend(frameon=False)

# --- B. 临界跳变率分析 (与模拟对齐版) ---

ax_b = fig.add_subplot(gs[0, 1])
mu_e_fine = np.linspace(0.0, 1.8, 29)
for i, sig in enumerate([0.15, 0.3]):
    phi_vals = []
    for me in mu_e_fine:
        # 获取当前模拟状态下的统计量
        m_h, q_h, _ = solve_system(0.0, mu_d_fixed, me, sig, steps=3500)
        m_ss, q_ss = np.mean(m_h[-500:]), np.mean(q_h[-500:])
        # 计算对齐后的解析 phi
        phi = calculate_analytical_phi(0.0, mu_d_fixed, me, m_ss, q_ss, sig)
        phi_vals.append(phi)
    ax_b.plot(mu_e_fine, phi_vals, 's-', mfc='none', markersize=4, color=colors[i + 1], label=rf'$\sigma={sig}$')

ax_b.set_xlabel(r'$\mu_e$')
ax_b.set_ylabel(r' $\phi$')
ax_b.set_title(r"B. Tipping Rate $\phi$")
ax_b.legend(frameon=False)

# --- C. PDF 演化 ---
ax_c = fig.add_subplot(gs[0, 2])
for i, me in enumerate([0.2, 0.6, 1.2]):
    _, _, final_x = solve_system(0.0, mu_d_fixed, me, 0.2, S=3000)
    ax_c.hist(final_x, bins=35, density=True, alpha=0.5, color=colors[i], label=rf'$\mu_e={me}$')
ax_c.set_title("C. PDF Evolution")
ax_c.legend(frameon=False)

# --- D. 有限尺寸效应 ---
ax_d = fig.add_subplot(gs[1, 0])
for i, S in enumerate([200, 1000, 5000]):
    me_range = np.linspace(0.8, 1.4, 9)
    ms = [np.mean(solve_system(-0.1, mu_d_fixed, me, 0.12, S=S)[0][-800:]) for me in me_range]
    ax_d.plot(me_range, ms, 'v-', color=colors[i], label=rf'$S={S}$')
ax_d.set_title("D. Finite-size Scaling")
ax_d.legend(frameon=False)

# --- E. 时间演化图 ---

ax_e = fig.add_subplot(gs[1, 1:])
times = np.arange(8000) * 0.02
for i, me in enumerate([0.6, 1.0, 1.4]):
    m_history, _, _ = solve_system(0.0, mu_d_fixed, me, 0.2, steps=8000)
    ax_e.plot(times, m_history, color=colors[i], label=rf'$\mu_e={me}$')
ax_e.set_xlabel('Time $t$'); ax_e.set_ylabel(r'$m(t)$')
ax_e.set_title("E. Temporal Evolution")
ax_e.axhline(-0.6, ls=':', color='black', alpha=0.5, label='$m_0$')
ax_e.legend(frameon=False, ncol=4, fontsize=8)

plt.tight_layout()
plt.show()