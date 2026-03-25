import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# --- PRL 风格配置 ---
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.8,
    "figure.figsize": (3.4, 2.8)
})


def solve_plots():
    # --- 物理参数定义 (与 LaTeX 推导对应) ---

    # h_c: 临界阈值 (Critical Threshold)
    # 对应推导中节点状态翻转的边界点，即 x 发生分支切换的场强位置
    hc = 0.385

    # sigma_h: 局部场分布的宽标准差 (Noise Intensity / Distribution Width)
    # 对应 P(h) 分布的宽度，决定了相变边缘的平滑程度（erfc 的斜率）
    sigma_h = 0.2

    # mu_d: 二体交互强度 (Dyadic Interaction Strength)
    # 对应推导公式 \mu_h = H_ext + mu_d * m + ... 中的线性反馈项
    mu_d = 0.4

    # mu_e: 三体交互强度 (Triadic/Higher-order Interaction Strength)
    # 对应推导中非线性项 0.5 * mu_e * m^2，是诱导一级相变的关键
    mu_e = 0.2

    # H_ext: 外部控制参数 (External Field / Driving Force)
    # 对应实验中的自变量，改变 H_ext 会平移 F(phi) 曲线，导致交点个数变化
    H_ext_list = [-0.1, 0.05, 0.2]

    # --- 状态变量定义 ---

    # x_minus & x_plus: 分支平均值 (Branch Expectations)
    # 对应推导中的 \bar{x}_- 和 \bar{x}_+。这里简化为常数，
    # 严格推导中它们应是局部场 h 的积分函数
    x_minus = -1.0
    x_plus = 1.0

    # phi_range: 序参量空间 (Order Parameter Space)
    # 对应 \phi \in [0, 1]，即系统中处于“正分支”状态的节点比例
    phi_range = np.linspace(0, 1, 500)

    fig, ax = plt.subplots(dpi=300)

    # 绘制自洽恒等线 y = phi (Self-consistency condition: \phi = F(\phi))
    ax.plot(phi_range, phi_range, 'k--', label=r'$\phi$', zorder=1)

    colors = ['#1f77b4', '#d62728', '#2ca02c']

    for i, H_ext in enumerate(H_ext_list):
        # 1. 计算宏观序参量 m (与 phi 的线性耦合)
        # 对应公式: m = (1-phi)*x_- + phi*x_+
        m = (1 - phi_range) * x_minus + phi_range * x_plus

        # 2. 计算平均局部场 mu_h (耦合了二体和三体作用)
        # 对应公式: \mu_h = H_ext + \mu_d * m + 0.5 * \mu_e * m^2
        mu_h = H_ext + mu_d * m + 0.5 * mu_e * (m ** 2)

        # 3. 计算自洽映射函数 F(phi)
        # 对应公式: \phi = 0.5 * erfc( (h_c - \mu_h) / (\sqrt{2}\sigma_h) )
        # 这是基于正态分布假设下的累积分布函数(CDF)的补函数
        F_phi = 0.5 * erfc((hc - mu_h) / (np.sqrt(2) * sigma_h))

        ax.plot(phi_range, F_phi, color=colors[i],
                label=r'$H_{ext} = ' + f'{H_ext}$')

    # --- 图形修饰 ---
    ax.set_xlabel(r'Input $\phi$')
    ax.set_ylabel(r'Response $\mathcal{F}(\phi)$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=False, loc='upper left')

    # 标注双稳态区域 (曲线与虚线有三个交点的情况)
    ax.annotate('Bistability', xy=(0.5, 0.4), xytext=(0.6, 0.2),
                arrowprops=dict(arrowstyle='->', lw=0.8), fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    solve_plots()