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
# 2. 核心动力学引擎
# ==========================================
def solve_system(mu_u, mu_d, mu_e, sigma, S=1000, dt=0.02, steps=4000, x0=-0.6):
    x = np.full(S, x0)
    m_h = np.zeros(steps)
    q_h = np.zeros(steps)
    for t in range(steps):
        m = np.mean(x)
        q = np.mean(x ** 2)
        drift = x - x ** 3 + mu_u + mu_d * m + mu_e * q
        diffusion = sigma * np.random.normal(0, 1, size=S)
        x += drift * dt + diffusion * np.sqrt(dt)
        m_h[t] = m
        q_h[t] = q
    return m_h, q_h, x


# ==========================================
# 3. 绘图任务：Sigma 分析
# ==========================================
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 3)

# --- A. 准静态 Sigma 扫描 ---
ax_a = fig.add_subplot(gs[0, 0])
sigma_range = np.linspace(0.01, 1.0, 21)
for i, mu_val in enumerate([0.4, 0.8]):
    ms = [np.mean(solve_system(-0.1, mu_val, 0.5, sig)[0][-500:]) for sig in sigma_range]
    ax_a.plot(sigma_range, ms, 'o-', color=colors[i], label=rf"$\mu_d={mu_val}$")
ax_a.set_xlabel(r'$\sigma$')
ax_a.set_ylabel(r'$m$')
ax_a.set_title(r"A. Equilibrium $m$ vs $\sigma$")
ax_a.legend(frameon=False)

# --- B. 临界跳变率分析 (解析平滑版) ---
ax_b = fig.add_subplot(gs[0, 1])
sigma_fine = np.linspace(0.01, 1.2, 50)
H_c = 0.3849
mu_u_const, mu_d_const = -0.1, 0.3
sigma_u, sigma_d, sigma_e = 0.1, 0.05, 0.02

# 物理预设：计算跳变前的“逃逸概率”
# 我们假设系统处于亚稳态附近 x ~ -0.6
m_fixed = -0.6
q_fixed = 0.36  # (-0.6)^2

for i, mu_e_val in enumerate([0.6, 0.9]):
    phi_list = []
    for sig in sigma_fine:
        # 考虑 q 随噪声 sigma 增大的物理响应 (近似 q = x0^2 + sigma^2)
        q_dynamic = m_fixed ** 2 + sig ** 2

        # 重新计算有效场 M 和 涨落 Gamma
        M = mu_u_const + mu_d_const * m_fixed + mu_e_val * q_dynamic
        G2 = (sigma_u ** 2 + sigma_d ** 2 * q_dynamic + sigma_e ** 2 * q_dynamic ** 2) * (sig ** 2)
        Gamma = np.sqrt(max(G2, 1e-9))

        # 计算 phi，并强制约束在 [0, 1] 区间
        res = 0.5 * (1 + erf((M - H_c) / (np.sqrt(2) * Gamma)))
        phi_list.append(np.clip(res, 0, 1))  # 保证物理意义

    ax_b.plot(sigma_fine, phi_list, '-', color=colors[i + 2], label=rf"$\mu_e={mu_e_val}$")

ax_b.set_xlabel(r'Noise $\sigma$')
ax_b.set_ylabel(r'Tipping Rate $\phi$')
ax_b.set_title(r"B. Tipping rate $\phi$")
ax_b.legend(frameon=False)
ax_b.grid(True, linestyle=':', alpha=0.5)

# --- C. PDF 演化 ---
ax_c = fig.add_subplot(gs[0, 2])
sigmas = [0.1, 0.3, 0.6]
for i, sig in enumerate(sigmas):
    _, _, final_x = solve_system(-0.1, 0.3, 0.5, sig, S=2000)
    ax_c.hist(final_x, bins=35, density=True, alpha=0.4, color=colors[i], label=rf"$\sigma={sig}$")
ax_c.set_xlabel(r'$x_i$');
ax_c.set_ylabel('PDF')
ax_c.set_title(r"C. PDF vs Sigma")
ax_c.legend(frameon=False)

# --- D. 有限尺寸效应 ---
ax_d = fig.add_subplot(gs[1, 0])
sizes = [100, 500, 2000]
sigma_range_d = np.linspace(0.1, 0.6, 10)
for i, S in enumerate(sizes):
    ms = [np.mean(solve_system(-0.1, 1.0, 0.5, sig, S=S)[0][-500:]) for sig in sigma_range_d]
    ax_d.plot(sigma_range_d, ms, 'v-', color=colors[i], label=rf"$S={S}$")
ax_d.set_xlabel(r'$\sigma$');
ax_d.set_ylabel(r'$m$')
ax_d.set_title("D. Finite-size Scaling")

# --- E. m 随 t 的演化图 ---
ax_e = fig.add_subplot(gs[1, 1:])
times = np.arange(8000) * 0.02
for i, sig in enumerate([0.15, 0.35, 0.6]):
    m_history, _, _ = solve_system(-0.1, 0.5, 0.5, sig, steps=8000)
    ax_e.plot(times, m_history, color=colors[i], label=rf"$\sigma={sig}$", alpha=0.8)
ax_e.set_xlabel('Time $t$');
ax_e.set_ylabel(r'$m(t)$')
ax_e.set_title("E. Noise-induced Escape")
ax_e.legend(frameon=False, ncol=3)

plt.tight_layout()
plt.show()