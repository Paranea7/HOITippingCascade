import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# --- 1. 风格配置 ---
plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "stix", "font.family": "STIXGeneral",
    "axes.labelsize": 10, "font.size": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.5,
    "axes.linewidth": 0.8, "xtick.direction": "in", "xtick.top": True, "ytick.right": True
})


def solve_system(m_in, mu_u, mu_d, mu_e, sig_u, sig_d, sig_e, h_c=0.3849):
    q = 0.36
    alpha = 0.8
    M = mu_u + mu_d * m_in + mu_e * (m_in ** 2)
    for _ in range(30):
        gamma_sq = sig_u ** 2 + (sig_d ** 2) * q + (sig_e ** 2) * (q ** 2)
        gamma = np.sqrt(max(gamma_sq, 1e-9))
        phi = 0.5 * (1 + erf((M - h_c) / (np.sqrt(2) * gamma)))
        M_limited = np.clip(M, -5, 5)
        q_target = 1 + (2 * phi - 1) * M_limited + (M_limited ** 2) / 4
        q = alpha * q_target + (1 - alpha) * q
    phi_final = 0.5 * (1 + erf((M - h_c) / (np.sqrt(2) * gamma)))
    return 2 * phi_final - 1 + M / 2


# --- 2. 参数留白区 (核心修改点) ---

# 第一行：均值效应 (每列对应的 mu_d 序列, mu_e 序列)
ROW_0_PARAMS = [
    ([0.0, 0.3, 0.6], [0.0, 0.0, 0.0]),  # Col 0: Only mu_d varies
    ([0.0, 0.0, 0.0], [0.0, 0.3, 0.6]),  # Col 1: Only mu_e varies
    ([0.2, 0.2, 0.2], [0.1, 0.3, 0.5])  # Col 2: Combined
]

# 第二行：涨落效应 (每列对应的 sig_u, sig_d, sig_e 序列)
# 格式: (sig_u_list, sig_d_list, sig_e_list)
ROW_1_PARAMS = [
    ([0.1], [0.0, 0.3, 0.6], [0.1]),  # Col 0: 重点看 sigma_d (二体涨落)
    ([0.1], [0.1], [0.0, 0.3, 0.6]),  # Col 1: 重点看 sigma_e (三体涨落)
    ([0.0, 0.4, 0.8], [0.1], [0.1])  # Col 2: 重点看 sigma_u (全局背景噪声)
]

# 共有常量
H_C = 0.3849
M_RANGE = np.linspace(-1.5, 2.5, 400)
COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']

# --- 3. 绘图循环 ---
fig, axes = plt.subplots(2, 3, figsize=(11, 7.5), dpi=150)
plt.subplots_adjust(wspace=0.3, hspace=0.35)

titles = [r"Two-body Interaction", r"Three-body Interaction", "Global System Noise"]

for col in range(3):
    # --- Row 0: Mean Effect ---
    ax_top = axes[0, col]
    md_l, me_l = ROW_0_PARAMS[col]
    for i in range(len(md_l)):
        m_vals = [solve_system(mi, 0.0, md_l[i], me_l[i], 0.15, 0.1, 0.1, H_C) for mi in M_RANGE]
        ax_top.plot(M_RANGE, m_vals, color=COLORS[i], label=rf"$\mu_d={md_l[i]}, \mu_e={me_l[i]}$")

    # --- Row 1: Fluctuation Effect ---
    ax_bot = axes[1, col]
    su_l, sd_l, se_l = ROW_1_PARAMS[col]

    # 确定谁在变，动态生成 Label
    max_len = max(len(su_l), len(sd_l), len(se_l))
    for i in range(max_len):
        # 自动补齐列表长度（如果某个列表只有一个值，则一直用那个值）
        curr_su = su_l[i] if i < len(su_l) else su_l[0]
        curr_sd = sd_l[i] if i < len(sd_l) else sd_l[0]
        curr_se = se_l[i] if i < len(se_l) else se_l[0]

        # 这里的均值参数可以根据需要微调，以保证曲线在相变点附近
        m_vals = [solve_system(mi, 0.0, 0.4, 0.3, curr_su, curr_sd, curr_se, H_C) for mi in M_RANGE]

        # 动态显示当前列变化的那个参数
        if len(sd_l) > 1:
            label = rf"$\sigma_d={curr_sd}$"
        elif len(se_l) > 1:
            label = rf"$\sigma_e={curr_se}$"
        else:
            label = rf"$\sigma_u={curr_su}$"

        ax_bot.plot(M_RANGE, m_vals, color=COLORS[i], label=label)

# --- 4. 装饰 ---
for r in range(2):
    for c in range(3):
        ax = axes[r, c]
        ax.plot([-2, 3], [-2, 3], 'k--', lw=0.8, alpha=0.4, label="$m_{out}=m_{in}$")
        ax.set_xlim([-1.2, 2.2]);
        ax.set_ylim([-1.2, 2.2])
        ax.set_aspect('equal');
        ax.grid(True, ls=':', alpha=0.3)
        ax.set_xlabel("$m_{in}$")
        ax.set_ylabel("$m_{out}$")
        if r == 0: ax.set_title(titles[c], pad=10, fontweight='bold')
        ax.legend(frameon=False, loc='upper left', fontsize=7)

plt.tight_layout()
plt.show()