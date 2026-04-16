import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import matplotlib as mpl

# --- 1. 样式配置 ---
mpl.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "font.size": 10,
    "figure.dpi": 150,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
})


# --- 2. 核心计算函数 ---
def solve_system(m_in, mu_u, mu_d, mu_e, sig_u, sig_d, sig_e, h_c=0.3849):
    q = 1.0
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
    m_out = 2 * phi_final - 1 + M / 2
    return m_out


def plot_with_slope(params_list, title, param_labels):
    """ 通用绘图函数：主图 + 斜率图 """
    M_RANGE = np.linspace(-1.5, 2.5, 400)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    for i, p in enumerate(params_list):
        # p 结构: (mu_u, mu_d, mu_e, sig_u, sig_d, sig_e)
        m_vals = np.array([solve_system(mi, *p) for mi in M_RANGE])
        # 计算斜率 dm_out / dm_in
        slope = np.gradient(m_vals, M_RANGE)

        # 主图
        ax1.plot(M_RANGE, m_vals, color=colors[i], label=param_labels[i])
        # 斜率图
        ax2.plot(M_RANGE, slope, color=colors[i], label=param_labels[i])

    # 主图美化
    ax1.plot([-2, 3], [-2, 3], 'k--', lw=0.8, alpha=0.5, label="$m_{out}=m_{in}$")
    ax1.set_xlim([-1.5, 2.5]);
    ax1.set_ylim([-1.5, 2.5])
    ax1.set_aspect('equal')
    ax1.set_xlabel("$m_{in}$");
    ax1.set_ylabel("$m_{out}$")
    ax1.set_title(title + " (Self-consistency)")
    ax1.legend(loc='upper left', fontsize=7)

    # 斜率图美化
    ax2.axhline(1.0, color='gray', lw=0.8, ls='--')  # 阈值线
    ax2.set_xlim([-1.5, 2.5]);
    ax2.set_ylim([0, 5])
    ax2.set_xlabel("$m_{in}$");
    ax2.set_ylabel("$dm_{out}/dm_{in}$")
    ax2.set_title("Response Slope (Stability)")
    ax2.text(-1.4, 1.1, ">1: Unstable/Tipping", fontsize=7, color='gray')

    plt.show()


# --- 3. 分别展示四种情况 ---

# 1. 外部输入项 (External Bias mu_u)
# 固定 mu_d=0.5, 其他噪声 sig=0.1
u_params = [(uu, 0.5, 0.0, 0.1, 0.0, 0.0) for uu in [-0.3, 0.0, 0.3]]
u_labels = [rf"$\mu_u={uu}$" for uu in [-0.3, 0.0, 0.3]]
plot_with_slope(u_params, "External Bias Effect", u_labels)

# 2. 两体作用 (Two-body mu_d)
# 固定 mu_u=0, 其他噪声 sig=0.1
d_params = [(0.0, md, 0.0, 0.1, 0.1, 0.0) for md in [0.2, 0.5, 0.8]]
d_labels = [rf"$\mu_d={md}$" for md in [0.2, 0.5, 0.8]]
plot_with_slope(d_params, "Two-body Interaction", d_labels)

# 3. 三体作用 (Three-body mu_e)
# 固定 mu_u=0, 其他噪声 sig=0.1
e_params = [(0.0, 0.0, me, 0.1, 0.1, 0.1) for me in [0.2, 0.4, 0.7]]
e_labels = [rf"$\mu_e={me}$" for me in [0.2, 0.4, 0.7]]
plot_with_slope(e_params, "Three-body Interaction", e_labels)

# 4. 混合相互作用 (Combined)
# 综合 mu_u, mu_d, mu_e
c_params = [
    (0.0, 0.4, 0.0, 0.1, 0.1, 0.1),
    (0.0, 0.4, 0.4, 0.1, 0.1, 0.1),
    (0.2, 0.4, 0.4, 0.1, 0.1, 0.1)
]
c_labels = ["$d=0.4$", "$d=0.4, e=0.4$", "$u=0.2, d=0.4, e=0.4$"]
plot_with_slope(c_params, "Combined Interaction", c_labels)