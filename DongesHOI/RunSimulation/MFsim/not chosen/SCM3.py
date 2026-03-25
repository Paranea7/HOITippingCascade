import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import matplotlib as mpl
mpl.rcParams.update({

    # --- 基本字体与文本 ---
    "text.usetex": False,               # 如果你有 LaTeX 环境，改为 True
    "mathtext.fontset": "stix",         # 数学字体接近 Times
    "font.family": "STIXGeneral",       # 正文字体接近 Times
    "font.size": 10,                    # 正文字体大小
    "axes.labelsize": 11,               # 坐标轴标签
    "axes.titlesize": 11,               # 子图标题
    "legend.fontsize": 8,               # 图例字号
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    # --- 线条与标记 ---
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "lines.markeredgewidth": 0.5,

    # --- 画布与子图 ---
    "figure.dpi": 150,
    "figure.figsize": (3.4, 2.5),       # 单栏图接近 PRL 单栏宽度 ~3.4 in
    "figure.autolayout": False,         # 通常配合 plt.tight_layout() 使用

    # --- 坐标轴与边框 ---
    "axes.linewidth": 1.0,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.grid": False,
    "axes.formatter.use_mathtext": True,

    # --- 刻度样式（PRL 常用内刻度，四边都有）---
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.major.size": 4,
    "ytick.minor.size": 2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,

    # --- 图例 ---
    "legend.frameon": False,            # PRL 常用无边框 legend
    "legend.framealpha": 1.0,
    "legend.fancybox": False,
    "legend.borderpad": 0.2,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.4,

    # --- 保存图像 ---
    "savefig.dpi": 300,                 # 投稿时导出 300 dpi 光栅图
    "savefig.transparent": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    # --- 颜色循环（可根据期刊黑白打印需求调整） ---
    "axes.prop_cycle": mpl.cycler(
        color=["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    ),
})

# --- 2. 核心计算函数 ---
def solve_system(m_in, mu_u, mu_d, mu_e, sig_u, sig_d, sig_e, h_c=0.3849):
    q = 1
    alpha = 0.8
    M = mu_u + mu_d * m_in + mu_e * (m_in ** 2)

    for _ in range(30):
        # 涨落效应公式：gamma^2 = sigma_u^2 + sigma_d^2 * q + sigma_e^2 * q^2
        gamma_sq = sig_u ** 2 + (sig_d ** 2) * q + (sig_e ** 2) * (q ** 2)
        gamma = np.sqrt(max(gamma_sq, 1e-9))
        phi = 0.5 * (1 + erf((M - h_c) / (np.sqrt(2) * gamma)))

        M_limited = np.clip(M, -5, 5)
        q_target = 1 + (2 * phi - 1) * M_limited + (M_limited ** 2) / 4
        q = alpha * q_target + (1 - alpha) * q

    phi_final = 0.5 * (1 + erf((M - h_c) / (np.sqrt(2) * gamma)))
    m_out = 2 * phi_final - 1 + M / 2
    return m_out


# --- 3. 参数配置区 ---
H_C = 0.3849
M_RANGE = np.linspace(-1.5, 2.5, 400)
colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']

# 第一行：均值效应实验参数
ROW_0_PARAMS = [
    ([0.1, 0.4, 0.7], [0.0, 0.0, 0.0]),  # Column 1: Vary mu_d
    ([0.0, 0.0, 0.0], [0.1, 0.3, 0.6]),  # Column 2: Vary mu_e
    ([0.3, 0.3, 0.3], [0.1, 0.3, 0.6])  # Column 3: Combined
]

# 第二行：涨落效应实验配置
# (针对变量索引, 标签, 固定sigma值[sig_u, sig_d, sig_e])
# 索引：0->sig_u, 1->sig_d, 2->sig_e
ROW_1_CONFIGS = [
    (1, r"\sigma_d", [0.12, None, 0.1]),  # 第一列：变 sig_d，固定 u=0.12, e=0.1
    (2, r"\sigma_e", [0.12, 0.1, None]),  # 第二列：变 sig_e，固定 u=0.12, d=0.1
    (0, r"\sigma_u", [None, 0.1, 0.1])  # 第三列：变 sig_u，固定 d=0.1, e=0.1
]
ROW_1_BASE_MU = [(0.7, 0.0), (0.0, 0.6), (0.2, 0.2)]  # 对应的基础均值 (mu_d, mu_e)
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
        # 第一行统一使用 [0.12, 0.1, 0.1] 作为基准噪声
        m_vals = [solve_system(mi, 0.0, md, me, 0.12, 0.1, 0.1, h_c=H_C) for mi in M_RANGE]
        ax_top.plot(M_RANGE, m_vals, color=colors[i], label=rf"$\mu_d={md:.1f}, \mu_e={me:.1f}$")

    # --- Row 1: Fluctuation Effect (更新后的逻辑) ---
    ax_bot = axes[1, col]
    target_idx, sig_label, current_sigs = ROW_1_CONFIGS[col]
    base_md, base_me = ROW_1_BASE_MU[col]

    for i, sv in enumerate(SIG_VARIANTS):
        test_sigs = list(current_sigs)
        test_sigs[target_idx] = sv  # 替换需要观察的变量

        m_vals = [solve_system(mi, 0.0, base_md, base_me, *test_sigs, h_c=H_C) for mi in M_RANGE]
        ax_bot.plot(M_RANGE, m_vals, color=colors[i], label=rf"${sig_label}={sv}$")

# --- 5. 装饰美化 ---
for row in range(2):
    for col in range(3):
        ax = axes[row, col]
        ax.plot([-2, 3], [-2, 3], 'k--', lw=0.7, alpha=0.5, label="$m_{out}=m_{in}$")
        ax.set_xlim([-1.5, 2.5])
        ax.set_ylim([-1.5, 2.5])
        ax.set_aspect('equal')
        ax.grid(True, ls=':', alpha=0.3)
        ax.set_xlabel("$m_{in}$")

        if col == 0:
            ylabel = "Mean Field Effect" if row == 0 else "Fluctuation Effect"
            ax.set_ylabel(f"{ylabel}\n\n$m_{{out}}$")
        else:
            ax.set_ylabel("$m_{out}$")

        if row == 0:
            ax.set_title(titles[col], pad=10, fontweight='bold')
        ax.legend(frameon=False, loc='upper left', fontsize=7)

plt.tight_layout()
plt.show()