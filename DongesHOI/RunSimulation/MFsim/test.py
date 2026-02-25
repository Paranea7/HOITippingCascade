import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# --- 1. 风格配置 (PRL 规范) ---
plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "stix", "font.family": "STIXGeneral",
    "axes.labelsize": 11, "font.size": 10, "legend.fontsize": 8,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "lines.linewidth": 1.5,
    "axes.linewidth": 1.0, "xtick.direction": "in", "xtick.top": True, "ytick.right": True
})


# --- 2. 核心物理函数 ---
def theory_m_out(m_in, mu_d, mu_e, sigma, h_c=0.3849):
    """基于推导的自洽方程: m = erf((M - H_c) / (sqrt(2)*Gamma))"""
    # 假设 q ≈ 1
    M = mu_d + mu_e * m_in
    gamma = sigma  # 简化模型中 Gamma 主要由噪声项贡献
    return erf((M - h_c) / (np.sqrt(2) * gamma))


def get_simulation_stats(m_in, mu_d, mu_e, sigma, samples, h_c=0.3849):
    """数值采样验证"""
    # 总驱动力 H = mu_d + mu_e * m_in + xi
    h_dist = mu_d + mu_e * m_in + sigma * samples
    # 计算 Tipping Rate phi: 超过临界阈值的比例
    phi = np.mean(h_dist > h_c)
    # 映射到序参量 m = 2*phi - 1
    m_out = 2 * phi - 1
    return m_out, phi


# --- 3. 参数设置 ---
np.random.seed(42)
samples = np.random.normal(0, 1, 20000)
m_range = np.linspace(-1.2, 1.2, 100)
h_c = 0.3849

# 基准参数
d_b, e_b, s_b = 0.5, 0.5, 0.8

# 扫描范围
scan_d = np.linspace(-0.5, 1.0, 50)
scan_e = np.linspace(0.0, 1.5, 50)
scan_s = np.linspace(0.01, 0.8, 50)

# --- 4. 绘图 ---
fig, axes = plt.subplots(2, 3, figsize=(11, 6), dpi=150, gridspec_kw={'height_ratios': [1.2, 1]})
colors = ['#1f77b4', '#d62728', '#2ca02c']

# --- Panel 1: Varying mu_d (均值场平移) ---
for i, v in enumerate([-0.2, 0.2, 0.6]):
    # 理论线
    m_theo = theory_m_out(m_range, v, e_b, s_b, h_c)
    axes[0, 0].plot(m_range, m_theo, color=colors[i], label=rf'$\mu_d={v}$')
    # 采样点 (可选，用于验证)
    if i == 1:
        m_sim = [get_simulation_stats(m, v, e_b, s_b, samples)[0] for m in m_range[::10]]
        axes[0, 0].plot(m_range[::10], m_sim, 'o', color=colors[i], markersize=3, alpha=0.5)

phi_d = [get_simulation_stats(0.0, d, e_b, s_b, samples)[1] for d in scan_d]
axes[1, 0].plot(scan_d, phi_d, 'k-', lw=2)
axes[1, 0].axvline(h_c, color='gray', ls='--', alpha=0.6, label='$H_c$')
axes[1, 0].set_xlabel(r'Field $\mu_d$')

# --- Panel 2: Varying mu_e (非线性反馈/斜率) ---
m_test_e = 0.5
for i, v in enumerate([0.2, 0.8, 1.4]):
    m_theo = theory_m_out(m_range, d_b, v, s_b, h_c)
    axes[0, 1].plot(m_range, m_theo, color=colors[i], label=rf'$\mu_e={v}$')

phi_e = [get_simulation_stats(m_test_e, d_b, e, s_b, samples)[1] for e in scan_e]
axes[1, 1].plot(scan_e, phi_e, 'k-')
axes[1, 1].set_xlabel(r'Coupling $\mu_e$ (at $m_{in}=0.5$)')

# --- Panel 3: Varying sigma (噪声激活) ---
m_test_s = 0.4
for i, v in enumerate([0.1, 0.25, 0.5]):
    m_theo = theory_m_out(m_range, d_b, e_b, v, h_c)
    axes[0, 2].plot(m_range, m_theo, color=colors[i], label=rf'$\sigma={v}$')

# 展示不同偏置下的噪声激活曲线
for i, d_val in enumerate([0.1, 0.3]):
    phi_s = [get_simulation_stats(m_test_s, d_val, e_b, s, samples)[1] for s in scan_s]
    axes[1, 2].plot(scan_s, phi_s, label=rf'$\mu_d={d_val}$', alpha=0.8)

axes[1, 2].set_xlabel(r'Noise $\sigma$ (at $m_{in}=0.4$)')
axes[1, 2].legend(frameon=False, loc='lower right')

# --- 5. 统一修饰 ---
titles = [r'Varying $\mu_d$', r'Varying $\mu_e$', r'Varying $\sigma$']
for j in range(3):
    # 自洽辅助线 y = x
    axes[0, j].plot([-1.5, 1.5], [-1.5, 1.5], 'k--', lw=0.8, alpha=0.4)
    axes[0, j].set_title(titles[j], fontweight='bold', pad=10)
    axes[0, j].set_ylabel(r'$m_{out}$')
    axes[0, j].set_xlabel(r'$m_{in}$')
    axes[0, j].legend(frameon=False, loc='upper left')
    axes[0, j].set_xlim([-1.2, 1.2]);
    axes[0, j].set_ylim([-1.1, 1.1])

    axes[1, j].set_ylabel(r'Tipping Rate $\phi$')
    axes[1, j].set_ylim([-0.05, 1.05])
    axes[1, j].grid(True, ls=':', alpha=0.4)

plt.tight_layout()
# plt.savefig("Self_Consistent_Analysis.pdf") # 建议保存为矢量图
plt.show()