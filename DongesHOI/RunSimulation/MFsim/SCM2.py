import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# --- 1. 风格配置 (PRL 规范) ---
plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "stix", "font.family": "STIXGeneral",
    "axes.labelsize": 10, "font.size": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.5,
    "axes.linewidth": 0.8, "xtick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.titlesize": 10
})


def solve_system(m_in, mu_u, mu_d, mu_e, sig_u, sig_d, sig_e, h_c=0.3849):
    q = 0.36
    alpha = 0.8  # 稍微调高阻尼，因为 m^2 的反馈更强

    # 1. 统一 M 的定义：这里我们研究 m^2 对有效场的贡献
    # 注意：在自洽体系中，通常认为 M = mu_u + mu_d*m + mu_e*m^2
    M = mu_u + mu_d * m_in + mu_e * (m_in ** 2)

    for _ in range(30):
        # 2. 计算 Gamma (涨落仍然取决于 q)
        gamma_sq = sig_u ** 2 + sig_d ** 2 * q + sig_e ** 2 * q ** 2
        gamma = np.sqrt(max(gamma_sq, 1e-9))

        # 3. 计算 phi
        phi = 0.5 * (1 + erf((M - h_c) / (np.sqrt(2) * gamma)))

        # 4. 更新 q (q 依然是二阶矩，受 M 调制)
        M_limited = np.clip(M, -5, 5)
        q_target = 1 + (2 * phi - 1) * M_limited + (M_limited ** 2) / 4
        q = alpha * q_target + (1 - alpha) * q

    # 5. 最终计算 m_out 时，必须使用与迭代一致的 M
    phi_final = 0.5 * (1 + erf((M - h_c) / (np.sqrt(2) * gamma)))

    # 注意：m_out 的定义基于分支位置：m = 2*phi - 1 + M/2
    # 由于此时 M 已经完全由输入的 m_in 确定，所以直接计算即可
    m_out = 2 * phi_final - 1 + M / 2

    return m_out


# --- 3. 绘图参数设置 ---
m_range = np.linspace(-1.5, 2.5, 400)
H_c = 0.3849

# 基础方差设置 (sigma_u, sigma_d, sigma_e)
base_sigmas = [0.12, 0.03, 0.03]

fig, axes = plt.subplots(2, 3, figsize=(11, 7), dpi=150)
plt.subplots_adjust(wspace=0.3, hspace=0.35)

column_configs = [
    {"mu_d": 0.4, "mu_e": 0.0, "title": "Two-body Only ($\mu_e=0$)"},
    {"mu_d": 0.0, "mu_e": 0.48, "title": "Three-body Only ($\mu_d=0$)"},
    {"mu_d": 0.2, "mu_e": 0.35, "title": "Combined Interaction"}
]

# 第一行变化的均值增量
mu_variants = [-0.3, 0.0, 0.3]
colors_mean = ['#1f77b4', '#d62728', '#2ca02c']

# 第二行变化的涨落 (主要变 sigma_u)
sig_variants = [0.0, 0.3, 0.6]
colors_fluc = ['#7f7f7f', '#e377c2', '#8c564b']

# --- 4. 循环绘图 ---
for col in range(3):
    cfg = column_configs[col]

    # --- Row 0: Mean Effect (调整 mu_d 或 mu_e) ---
    ax = axes[0, col]
    for i, var in enumerate(mu_variants):
        curr_mu_d = cfg["mu_d"] + (var if col != 1 else 0)
        curr_mu_e = cfg["mu_e"] + (var if col == 1 else 0)

        m_vals = [solve_system(mi, 0.0, curr_mu_d, curr_mu_e, *base_sigmas, h_c=H_c) for mi in m_range]

        label = rf"$\mu_d={curr_mu_d:.1f}$" if col != 1 else rf"$\mu_e={curr_mu_e:.1f}$"
        ax.plot(m_range, m_vals, color=colors_mean[i], label=label)

    # --- Row 1: Fluctuation Effect (sigma_u) ---
    ax = axes[1, col]
    for i, sv in enumerate(sig_variants):
        # 保持均值参数为各列的基准值
        m_vals = [solve_system(mi, 0.0, cfg["mu_d"], cfg["mu_e"], sv, 0.1, 0.1, h_c=H_c) for mi in m_range]
        ax.plot(m_range, m_vals, color=colors_fluc[i], label=rf"$\sigma={sv}$")

# --- 5. 装饰美化 ---
for row in range(2):
    for col in range(3):
        ax = axes[row, col]
        ax.plot([-2, 3], [-2, 3], 'k--', lw=0.8, alpha=0.5, label="$m_{out}=m_{in}$")
        ax.set_xlim([-1.5, 2.5])
        ax.set_ylim([-1.5, 2.5])
        ax.set_aspect('equal')
        ax.grid(True, ls=':', alpha=0.3)
        ax.set_xlabel("$m_{in}$")

        if col == 0:
            row_title = "Mean Field Effect" if row == 0 else "Fluctuation Effect"
            ax.set_ylabel(f"{row_title}\n\n$m_{{out}}$")
        else:
            ax.set_ylabel("$m_{out}$")

        if row == 0:
            ax.set_title(column_configs[col]["title"], pad=10, fontweight='bold')

        ax.legend(frameon=False, loc='upper left', fontsize=7)

plt.tight_layout()
plt.show()