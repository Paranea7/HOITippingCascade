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
    x = np.full(S, x0)
    m_h = np.zeros(steps)
    q_h = np.zeros(steps)
    for t in range(steps):
        m = np.mean(x)
        q = np.mean(x ** 2)
        # 漂移项包含 m 和 q 的反馈
        drift = x - x ** 3 + mu_u + mu_d * m + mu_e * q
        diffusion = sigma * np.random.normal(0, 1, size=S)
        x += drift * dt + diffusion * np.sqrt(dt)
        m_h[t] = m
        q_h[t] = q
    return m_h, q_h, x


# ==========================================
# 3. 绘图任务
# ==========================================
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 3)

# 统一全局参数
mu_u_const, mu_e_const = 0.0, 0.5
H_c = 0.3849

# --- A. Simulation mu_d Sweep ---
ax_a = fig.add_subplot(gs[0, 0])
mu_d_range = np.linspace(-1.5, 1.5, 21)
for i, sig in enumerate([0.1, 0.4]):
    ms = [np.mean(solve_system(mu_u_const, md, mu_e_const, sig)[0][-500:]) for md in mu_d_range]
    ax_a.plot(mu_d_range, ms, 'o-', mfc='none', color=colors[i], label=rf'$\sigma={sig}$')
ax_a.set_xlabel(r'$\mu_d$');
ax_a.set_ylabel(r'$m$')
ax_a.set_title(r"A. Simulation $\mu_d$ Sweep")
ax_a.legend(frameon=False)

# --- B. Tipping rate phi (统一二阶矩计算) ---
ax_b = fig.add_subplot(gs[0, 1])
mu_d_fine = np.linspace(-1.5, 1.5, 200)

for i, sig_base in enumerate([0.1, 0.4]):
    phi_list = []
    m_curr = -0.6  # 初始猜测值

    for md in mu_d_fine:
        def self_consistent_equations(m):
            # 核心修正：q = m^2 + Var(x)，稳态方差近似为 sigma^2/2
            q_val = m ** 2 + (sig_base ** 2 / 2.0)
            M = mu_u_const + md * m + mu_e_const * q_val
            # 统一有效噪声 Gamma
            Gamma = sig_base * np.sqrt(1 + 0.1 * q_val)
            phi = 0.5 * (1 + erf((M - H_c) / (np.sqrt(2) * Gamma)))
            # 求解 m 的平衡点
            rhs = (4 * phi - 2 + mu_u_const + mu_e_const * q_val) / (2 - md)
            return m - rhs


        m_sol = fsolve(self_consistent_equations, x0=m_curr)[0]
        m_curr = m_sol  # 路径追踪

        # 计算该解对应的 phi
        q_sol = m_sol ** 2 + (sig_base ** 2 / 2.0)
        M_sol = mu_u_const + md * m_sol + mu_e_const * q_sol
        Gamma_sol = sig_base * np.sqrt(1 + 0.1 * q_sol)
        phi_final = 0.5 * (1 + erf((M_sol - H_c) / (np.sqrt(2) * Gamma_sol)))
        phi_list.append(phi_final)

    ax_b.plot(mu_d_fine, phi_list, '-', color=colors[i], label=rf'$\sigma={sig_base}$')

ax_b.set_xlabel(r'$\mu_d$');
ax_b.set_ylabel(r'$\phi$')
ax_b.set_title(r"B. Tipping rate $\phi$ ")
ax_b.legend(frameon=False)

# --- C. PDF Evolution (还原) ---
ax_c = fig.add_subplot(gs[0, 2])
for i, md in enumerate([0.2, 0.8, 1.4]):
    _, _, final_x = solve_system(mu_u_const, md, mu_e_const, 0.2, S=2000)
    ax_c.hist(final_x, bins=30, density=True, alpha=0.5, color=colors[i], label=rf'$\mu_d={md}$')
ax_c.set_title(r"C. PDF Evolution")
ax_c.legend(frameon=False)

# --- D. Finite-size Scaling (还原) ---
ax_d = fig.add_subplot(gs[1, 0])
for i, S in enumerate([100, 500, 2000]):
    md_vals = np.linspace(0.8, 1.8, 10)
    ms = [np.mean(solve_system(mu_u_const, md, mu_e_const, 0.1, S=S)[0][-500:]) for md in md_vals]
    ax_d.plot(md_vals, ms, 'v-', color=colors[i + 1], label=rf'$S={S}$')
ax_d.set_title("D. Finite-size Scaling")
ax_d.legend(frameon=False)

# --- E. Temporal Evolution ---
ax_e = fig.add_subplot(gs[1, 1:])
times = np.arange(8000) * 0.02
for i, md in enumerate([0.2, 0.8, 1.4]):
    m_history, _, _ = solve_system(mu_u_const, md, mu_e_const, 0.2, steps=8000)
    ax_e.plot(times, m_history, color=colors[i], label=rf'$\mu_d={md}$')
ax_e.set_title("E. Temporal Evolution")
ax_e.set_xlabel("Time");
ax_e.set_ylabel(r"$m$")
ax_e.legend(frameon=False, ncol=3)

plt.tight_layout()
plt.show()