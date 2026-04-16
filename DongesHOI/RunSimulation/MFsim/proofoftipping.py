import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# ==========================================
# 1. Style Configuration
# ==========================================
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 11,
    "xtick.direction": "in", "ytick.direction": "in", "figure.dpi": 150
})
colors = ['#1f77b4', '#d62728']

# ==========================================
# 2. Corrected Mapping Logic
# ==========================================
def get_m_out(m_in, mu_u, mu_d, mu_e, sig_u):
    """ 计算单步自洽映射: m_out = f(m_in) """
    HC = 0.3849
    # 有效场 M 的正确定义
    M = mu_u + mu_d * m_in + mu_e * (m_in**2)
    q = 1.0 + (m_in * M) / 2.0
    gamma = np.sqrt(max(sig_u**2 + 0.01 * q, 1e-8))
    phi = 0.5 * (1 + erf((M - HC) / (np.sqrt(2) * gamma)))
    return 2 * phi - 1 + M / 2

# ==========================================
# 3. Data Generation (Using Mapping residuals)
# ==========================================
mu_zoom = np.linspace(0.2, 0.5, 500) # 聚焦在跳变发生的关键区域
m_range = np.linspace(-1.5, 2.0, 1000)

def find_stable_roots(mu_val, mu_d, mu_e, sig_u):
    """ 寻找 m_out = m_in 的所有根 """
    res = []
    m_test = np.linspace(-1.5, 2.0, 2000)
    # 计算映射残差: f(m) - m
    diff = [get_m_out(m, mu_val, mu_d, mu_e, sig_u) - m for m in m_test]
    # 寻找过零点
    idx = np.where(np.diff(np.sign(diff)))[0]
    return m_test[idx]

# 生成数据
def get_b_column_data(mu_axis, mu_d, mu_e, sig_u):
    upper, lower = [], []
    for mu in mu_axis:
        roots = find_stable_roots(mu, mu_d, mu_e, sig_u)
        if len(roots) > 0:
            lower.append(np.min(roots))
            upper.append(np.max(roots))
        else:
            lower.append(np.nan); upper.append(np.nan)
    return np.array(lower), np.array(upper)

# --- 参数设定 ---
sig_test = 0.13
# 两体组
low2, up2 = get_b_column_data(mu_zoom, 0.2, 0.0, sig_test)
# 三体组
low3, up3 = get_b_column_data(mu_zoom, 0.0, 0.2, sig_test)

# ==========================================
# 4. Plotting (Corrected 2x2)
# ==========================================
fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(8, 7))

# --- Row 1: Two-body ---
axs[0, 0].plot(mu_zoom, low2, color=colors[0], label='Lower')
axs[0, 0].plot(mu_zoom, up2, color=colors[0], linestyle='--', label='Upper')
axs[0, 0].set_title(r'Two-body ($\mu_d=0.2$): Hysteresis View')
axs[0, 0].set_ylabel(r'Order Parameter $m$')

dm2 = np.gradient(up2, mu_zoom)
axs[0, 1].plot(mu_zoom, dm2, color=colors[0])
axs[0, 1].set_title(r'Susceptibility $\chi$')

# --- Row 2: Three-body ---
axs[1, 0].plot(mu_zoom, low3, color=colors[1])
axs[1, 0].plot(mu_zoom, up3, color=colors[1], linestyle='--')
axs[1, 0].set_title(r'Three-body ($\mu_e=0.2$): Hysteresis View')
axs[1, 0].set_ylabel(r'$m$')
axs[1, 0].set_xlabel(r'External Bias $\mu_u$')

dm3 = np.gradient(up3, mu_zoom)
axs[1, 1].plot(mu_zoom, dm3, color=colors[1])
axs[1, 1].set_title(r'Susceptibility $\chi$')
axs[1, 1].set_xlabel(r'External Bias $\mu_u$')

for row in range(2):
    axs[row, 0].text(0.05, 0.9, "(b)", transform=axs[row, 0].transAxes, fontweight='bold')
    axs[row, 1].text(0.05, 0.9, "(c)", transform=axs[row, 1].transAxes, fontweight='bold')
    for col in range(2): axs[row, col].grid(alpha=0.2)

plt.show()