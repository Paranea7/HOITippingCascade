import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import fsolve
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 1. Style Configuration
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
colors_list = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

SIG_U = 0.128
SIG_D = 0.00
H_C = 0.3849  # 2/(3*sqrt(3))


def solve_theory_m(mu_e, sig_e):
    def eq(v):
        m, q = v
        # 1. 计算有效场与波动强度
        M = mu_e * (m ** 2)
        # 注意：这里保留 sig_u, sig_d 以匹配你的全局设置
        Gamma = np.sqrt(SIG_U ** 2 + SIG_D ** 2 * q + sig_e ** 2 * q ** 2)

        # 2. 计算分支位置 (一阶微扰)
        # 即使 M 较大，微扰公式在解析连续性上表现更好
        xp = 1.0 + M / 2.0
        xn = -1.0 + M / 2.0

        # 3. 计算翻转概率 (纯 erf 逻辑，不再硬截断)
        # 当 M 接近或超过 H_C 且 Gamma 很小时，erf 会自然趋向 1.0
        # 为防止分母为 0，添加一个极小的 epsilon
        eps = 1e-9
        phi_eff = 0.5 * (1 + erf((M - H_C) / (np.sqrt(2) * Gamma + eps)))

        # 4. 计算宏观统计量
        target_m = (1 - phi_eff) * xn + phi_eff * xp
        target_q = (1 - phi_eff) * (xn ** 2) + phi_eff * (xp ** 2)

        return [m - target_m, q - target_q]

    # 初始猜测：从全负状态开始迭代
    sol = fsolve(eq, x0=[-1.0, 1.0], xtol=1e-9)
    return sol[0], sol[1]


def calculate_phi_final(mu_e, m, q, sig_e):
    """移除截断，完全由 erf 决定的翻转率"""
    M = mu_e * (m ** 2)
    Gamma = np.sqrt(SIG_U ** 2 + SIG_D ** 2 * q + sig_e ** 2 * q ** 2)
    eps = 1e-9
    return 0.5 * (1 + erf((M - H_C) / (np.sqrt(2) * Gamma + eps)))


# ==========================================
# 3. Dynamical Engine
# ==========================================
def solve_system(mu_e, sig_e, S=1500, dt=0.02, steps=5000, x0=-1.0):
    gamma_total = np.sqrt(SIG_U ** 2 + SIG_D ** 2 + sig_e ** 2)
    xi = np.random.normal(0, gamma_total, S)
    x = np.full(S, x0, dtype=float) + np.random.normal(0, 0.01, S)

    m_h = np.zeros(steps)
    for t in range(steps):
        m = np.mean(x)
        # Higher-order interaction field: M = mu_e * m^2
        drift = x - x ** 3 + mu_e * (m ** 2) + xi
        x += drift * dt
        x = np.clip(x, -2.0, 2.0)
        m_h[t] = m
    return m_h, x


# ==========================================
# 4. Plotting
# ==========================================
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 3)

sig_e_samples = [0.05, 0.5]
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

mu_e_range_sim = np.linspace(0.0, 1.2, 26)
mu_e_range_theo = np.linspace(0.0, 1.2, 100)

for idx, se in enumerate(sig_e_samples):
    m_theo, phi_theo = [], []
    for me in mu_e_range_theo:
        mt, qt = solve_theory_m(me, se)
        m_theo.append(mt)
        phi_theo.append(calculate_phi_final(me, mt, qt, se))

    ms_sim, phi_sim = [], []
    for me in mu_e_range_sim:
        mh, fx = solve_system(me, se, S=1000)
        ms_sim.append(mh[-500:].mean())
        phi_sim.append(np.sum(fx > 0) / len(fx))

    ax_a.plot(mu_e_range_theo, m_theo, '-', color=colors_list[idx], alpha=0.7)
    ax_a.plot(mu_e_range_sim, ms_sim, 'o', mfc='none', color=colors_list[idx], label=rf"$\sigma_e={se}$")

    ax_b.plot(mu_e_range_theo, phi_theo, '-', color=colors_list[idx], alpha=0.7)
    ax_b.plot(mu_e_range_sim, phi_sim, 's', mfc='none', markersize=4, color=colors_list[idx])

ax_a.set_title("A. Mean State $m$ (Truncated)")
ax_a.set_xlabel(r"$\mu_e$");
ax_a.set_ylabel("$m$")
ax_a.legend(frameon=False, fontsize=7)

ax_b.set_title(r"B. Tipping Rate $\phi$")
ax_b.set_xlabel(r"$\mu_e$")

# --- PDF, Size Effect, and Temporal plots remain similar ---
# [Code for C, D, E panels follows the same logic as your original script]

# --- C. PDF Evolution ---
ax_c = fig.add_subplot(gs[0, 2])
for i, me in enumerate([0.2, 0.5, 0.8]):
    _, final_x = solve_system(me, sig_e=0.15)
    ax_c.hist(final_x, bins=40, density=True, alpha=0.4, color=colors_list[i], label=rf'$\mu_e={me}$')
ax_c.set_title(r"C. State PDF ($\sigma_e=0.15$)");
ax_c.legend(frameon=False, fontsize=8)

# --- D. Finite Size Effect ---
ax_d = fig.add_subplot(gs[1, 0])
for i, S_val in enumerate([200, 500, 2500]):
    me_range_d = np.linspace(0.4, 0.9, 6)
    ms_d = [solve_system(me, sig_e=0.1, S=S_val)[0][-500:].mean() for me in me_range_d]
    ax_d.plot(me_range_d, ms_d, 'v-', color=colors_list[i], markersize=4, label=rf'$S={S_val}$')
ax_d.set_title("D. Size Effect Scaling");
ax_d.legend(frameon=False, fontsize=8)

# --- E. Temporal Evolution ---
ax_e = fig.add_subplot(gs[1, 1:])
for i, me in enumerate([0.3, 0.6, 0.9]):
    mh, _ = solve_system(me, sig_e=0.15, steps=5000)
    ax_e.plot(mh, color=colors_list[i], label=rf'$\mu_e={me}$')
ax_e.set_title("E. Temporal Paths ")
ax_e.set_xlabel('Steps');
ax_e.legend(frameon=False, ncol=3)

plt.tight_layout()
plt.show()
