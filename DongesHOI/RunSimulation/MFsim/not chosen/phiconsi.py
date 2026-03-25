import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import matplotlib as mpl

# --- 1. 沿用您的绘图配置 ---
mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.5,
    "figure.dpi": 150,
    "figure.figsize": (8.5, 6),  # 调整比例以适应 2x3 布局
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
})


# --- 2. 核心计算函数 (修改为返回 phi) ---
def solve_phi_system(phi_in, mu_u, mu_d, mu_e, sig_u, sig_d, sig_e, h_c=0.3849):
    # 根据 phi_in 估算对应的 m_in (利用一阶微扰关系: m = 2phi - 1)
    # 注意：在自洽迭代中，M 依赖于 m，而 m 依赖于 phi
    m_in = 2 * phi_in - 1

    q = 1
    alpha = 0.8
    # 计算有效场 M
    M = mu_u + mu_d * m_in + mu_e * (m_in ** 2)

    # 内部循环用于平衡 q 的响应 (模拟原程序逻辑)
    for _ in range(30):
        gamma_sq = sig_u ** 2 + (sig_d ** 2) * q + (sig_e ** 2) * (q ** 2)
        gamma =gamma_sq
        phi_current = 0.5 * (1 + erf((M - h_c) / (np.sqrt(2) * gamma)))

        M_limited = np.clip(M, -5, 5)
        q_target = 1 + (2 * phi_current - 1) * M_limited + (M_limited ** 2) / 4
        q = alpha * q_target + (1 - alpha) * q

    # 最终输出的 phi_out
    phi_out = 0.5 * (1 + erf((M - h_c) / (np.sqrt(2) * gamma)))
    return phi_out


# --- 3. 参数配置区 (保持与原程序一致) ---
H_C = 0.3849
PHI_RANGE = np.linspace(0, 1, 400)  # phi 的范围是 0 到 1
colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']

ROW_0_PARAMS = [
    ([-0.4, 0.4, 0.6], [0.0, 0.0, 0.0]),  # Vary mu_d
    ([0.0, 0.0, 0.0], [0.1, 0.3, 0.8]),  # Vary mu_e
    ([0.3, 0.3, 0.3], [0.1, 0.3, 0.8])  # Combined
]

ROW_1_CONFIGS = [
    (1, r"\sigma_d", [0.12, None, 0.2]),
    (2, r"\sigma_e", [0.12, 0.2, None]),
    (0, r"\sigma_u", [None, 0.2, 0.2])
]
ROW_1_BASE_MU = [(0.7, 0.0), (0.0, 0.6), (0.2, 0.2)]
SIG_VARIANTS = [0.0, 0.3, 0.6]

# --- 4. 绘图逻辑 ---
fig, axes = plt.subplots(2, 3, figsize=(11, 7), dpi=150)
plt.subplots_adjust(wspace=0.3, hspace=0.35)

titles = [r"Two-body Only ($\mu_e=0$)", r"Three-body Only ($\mu_d=0$)", "Combined Interaction"]

for col in range(3):
    # --- Row 0: Mean Field Effect ---
    ax_top = axes[0, col]
    mu_d_list, mu_e_list = ROW_0_PARAMS[col]
    for i in range(len(mu_d_list)):
        md, me = mu_d_list[i], mu_e_list[i]
        phi_vals = [solve_phi_system(pi, 0.0, md, me, 0.12, 0.1, 0.1, h_c=H_C) for pi in PHI_RANGE]
        ax_top.plot(PHI_RANGE, phi_vals, color=colors[i], label=rf"$\mu_d={md:.1f}, \mu_e={me:.1f}$")

    # --- Row 1: Fluctuation Effect ---
    ax_bot = axes[1, col]
    target_idx, sig_label, current_sigs = ROW_1_CONFIGS[col]
    base_md, base_me = ROW_1_BASE_MU[col]

    for i, sv in enumerate(SIG_VARIANTS):
        test_sigs = list(current_sigs)
        test_sigs[target_idx] = sv
        phi_vals = [solve_phi_system(pi, 0.0, base_md, base_me, *test_sigs, h_c=H_C) for pi in PHI_RANGE]
        ax_bot.plot(PHI_RANGE, phi_vals, color=colors[i], label=rf"${sig_label}={sv}$")

# --- 5. 装饰美化 ---
for row in range(2):
    for col in range(3):
        ax = axes[row, col]
        ax.plot([0, 1], [0, 1], 'k--', lw=0.7, alpha=0.5, label=r"$\phi_{out}=\phi_{in}$")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
        ax.grid(True, ls=':', alpha=0.3)
        ax.set_xlabel(r"$\phi_{in}$")

        if col == 0:
            ylabel = "Mean Field Effect" if row == 0 else "Fluctuation Effect"
            ax.set_ylabel(rf"{ylabel}\n\n$\phi_{{out}}$")
        else:
            ax.set_ylabel(r"$\phi_{out}$")

        if row == 0:
            ax.set_title(titles[col], pad=10, fontweight='bold')
        ax.legend(frameon=False, loc='upper left', fontsize=7)

plt.tight_layout()
plt.show()
