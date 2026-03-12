import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve


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
# 2. 核心动力学引擎
# ==========================================
def solve_system(mu_u, mu_d, mu_e, sigma, S=1000, dt=0.02, steps=4000, x0=-0.6):
    x = np.full(S, x0) + np.random.normal(0, 0.05, S)  # 加入微小扰动打破对称
    m_h = np.zeros(steps)
    q_h = np.zeros(steps)
    for t in range(steps):
        m = np.mean(x)
        q = np.mean(x ** 2)
        drift = x - x ** 3 + mu_u + mu_d * m + mu_e * q
        # 标准 SDE 更新
        x += drift * dt + sigma * np.random.normal(0, 1, size=S) * np.sqrt(dt)
        m_h[t] = m
        q_h[t] = q
    return m_h, q_h, x


# ==========================================
# 3. 自洽解析函数 (针对图 B 对齐)
# ==========================================
def get_self_consistent_phi(md, mu_u, mu_e, sigma_base, h_c=0.3849):
    # 内部定义的自洽参数
    sigma_u, sigma_d, sigma_e = 0.1, 0.05, 0.02

    def residual(m_val):
        # 稳态假设：q = m^2 + sigma^2/2 (在势阱底部附近的近似)
        q_val = m_val ** 2 + (sigma_base ** 2 / 2.0)
        M = mu_u + md * m_val + mu_e * q_val
        G2 = (sigma_u ** 2 + sigma_d ** 2 * q_val + sigma_e ** 2 * q_val ** 2) + sigma_base ** 2
        Gamma = np.sqrt(G2)
        phi = 0.5 * (1 + erf((M - h_c) / (np.sqrt(2) * Gamma)))
        # 自洽映射：m 应该指向 2*phi - 1 (从 -1 跳到 1)
        return (2 * phi - 1) - m_val

    # 寻找稳定解
    m_sol = fsolve(residual, x0=-0.6)[0]

    # 返回基于该稳定解计算的 phi
    q_sol = m_sol ** 2 + (sigma_base ** 2 / 2.0)
    M_sol = mu_u + md * m_sol + mu_e * q_sol
    G2_sol = (sigma_u ** 2 + sigma_d ** 2 * q_sol + sigma_e ** 2 * q_sol ** 2) + sigma_base ** 2
    phi_final = 0.5 * (1 + erf((M_sol - h_c) / (np.sqrt(2) * np.sqrt(G2_sol))))
    return phi_final


# ==========================================
# 4. 绘图任务
# ==========================================
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 3)

# --- A. 准静态参数扫描 ---
ax_a = fig.add_subplot(gs[0, 0])
mu_d_range = np.linspace(-1.5, 1.5, 21)
for i, sig in enumerate([0.15, 0.4]):  # 调高低噪声值以获得更清晰的跳变
    ms = [np.mean(solve_system(-0.1, md, 0.5, sig)[0][-800:]) for md in mu_d_range]
    ax_a.plot(mu_d_range, ms, 'o-', mfc='none', color=colors[i], label=rf'$\sigma={sig}$')
ax_a.set_xlabel(r'Pairwise Coupling $\mu_d$')
ax_a.set_ylabel(r'Order Parameter $m$')
ax_a.set_title(r"A. Quasi-static $\mu_d$ Sweep")
ax_a.legend(frameon=False)

# --- B. 临界跳变率分析 (纯自洽解析对齐) ---

ax_b = fig.add_subplot(gs[0, 1])
mu_d_fine = np.linspace(-1.5, 1.5, 100)
mu_u_const, mu_e_const = 0.0, 0.5

for i, sig_base in enumerate([0.1, 0.3]):
    # 注意：这里不再运行 solve_system，而是直接计算自洽解
    phi_list = [get_self_consistent_phi(md, mu_u_const, mu_e_const, sig_base) for md in mu_d_fine]
    ax_b.plot(mu_d_fine, phi_list, '-', color=colors[i + 1], label=rf'$\sigma={sig_base}$')

ax_b.set_xlabel(r'$\mu_d$')
ax_b.set_ylabel(r'Tipping Rate $\phi$')
ax_b.set_title(r"B. Self-consistent $\phi$")
ax_b.legend(frameon=False)
ax_b.grid(True, linestyle=':', alpha=0.5)

# --- C. PDF 演化 ---
ax_c = fig.add_subplot(gs[0, 2])
mu_d_points = [0.2, 0.6, 1.2]
for i, md in enumerate(mu_d_points):
    _, _, final_x = solve_system(0.0, md, 0.5, 0.2, S=2000)
    ax_c.hist(final_x, bins=35, density=True, alpha=0.5, color=colors[i], label=rf'$\mu_d={md}$')
ax_c.set_title(r"C. PDF Evolution")
ax_c.legend(frameon=False)

# --- D. 有限尺寸效应 ---
ax_d = fig.add_subplot(gs[1, 0])
sizes = [200, 1000, 5000]
for i, S in enumerate(sizes):
    mu_range = np.linspace(0.8, 1.6, 10)
    ms = [np.mean(solve_system(0.0, md, 0.5, 0.15, S=S)[0][-800:]) for md in mu_range]
    ax_d.plot(mu_range, ms, 'v-', color=colors[i], label=rf'$S={S}$')
ax_d.set_title("D. Finite-size Scaling")
ax_d.legend(frameon=False)

# --- E. 时间演化图 ---

ax_e = fig.add_subplot(gs[1, 1:])
times = np.arange(8000) * 0.02
for i, md in enumerate([0.4, 0.9, 1.4]):
    m_history, _, _ = solve_system(0.0, md, 0.5, 0.25, steps=8000)
    ax_e.plot(times, m_history, color=colors[i], label=rf'$\mu_d={md}$')
ax_e.set_xlabel('Time $t$')
ax_e.set_ylabel(r'$m(t)$')
ax_e.set_title("E. Temporal Evolution")
ax_e.axhline(-0.6, ls=':', color='black', alpha=0.5, label='$m_0$')
ax_e.legend(frameon=False, ncol=4, fontsize=8)

plt.tight_layout()
plt.show()