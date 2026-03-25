import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import cumtrapz

# --- 1. 基础配置与物理模型 ---
plt.rcParams.update({"text.usetex": False, "font.family": "STIXGeneral", "axes.labelsize": 10})


def f_m(m, mu_d, mu_e, sig_u, sig_d, sig_e, h_c=0.3849):
    """计算迭代函数 m_out = f(m_in)"""
    # 简化版 q 达到稳态的近似逻辑 (用于演示核心动力学)
    q = 0.4
    M = 0.0 + mu_d * m + mu_e * (m ** 2)
    gamma = np.sqrt(sig_u ** 2 + sig_d ** 2 * q + sig_e ** 2 * q ** 2)
    phi = 0.5 * (1 + np.vectorize(lambda x: np.math.erf(x))((M - h_c) / (np.sqrt(2) * gamma)))
    return 2 * phi - 1 + M / 2


# --- 2. 分析核心：不动点与稳定性 ---
def analyze_dynamics(mu_e_val):
    m_range = np.linspace(-2, 3, 500)
    diff = f_m(m_range, 0.4, mu_e_val, 0.12, 0.1, 0.1) - m_range

    # 寻找不动点 (diff=0 的点)
    fixed_points = []
    for i in range(len(m_range) - 1):
        if diff[i] * diff[i + 1] < 0:
            root = fsolve(lambda m: f_m(m, 0.4, mu_e_val, 0.12, 0.1, 0.1) - m, m_range[i])[0]
            # 计算稳定性 (斜率 lambda)
            eps = 1e-5
            slope = (f_m(root + eps, 0.4, mu_e_val, 0.12, 0.1, 0.1) -
                     f_m(root - eps, 0.4, mu_e_val, 0.12, 0.1, 0.1)) / (2 * eps)
            fixed_points.append((root, slope))
    return fixed_points


# --- 3. 绘图与分析 ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# (A) 势能函数重建 (Potential Landscape)
ax1 = axes[0, 0]
mu_e_test = 0.6
m_axis = np.linspace(-1.5, 2.5, 200)
# 驱动力 F(m) = f(m) - m
force = f_m(m_axis, 0.4, mu_e_test, 0.12, 0.1, 0.1) - m_axis
potential = -cumtrapz(force, m_axis, initial=0)
ax1.plot(m_axis, potential, 'b-', lw=2)
ax1.set_title("1. Potential Landscape (Double Well)")
ax1.set_xlabel("m");
ax1.set_ylabel("V(m)")
ax1.grid(True, ls=':')

# (B) 分叉分析 (Bifurcation Diagram)
ax2 = axes[0, 1]
mu_e_range = np.linspace(0, 1.2, 50)
for me in mu_e_range:
    pts = analyze_dynamics(me)
    for p, s in pts:
        color = 'red' if s > 1 else 'blue'  # 红色为不稳定点，蓝色为稳定点
        ax2.plot(me, p, '.', color=color, markersize=3)
ax2.set_title("2. Bifurcation (Blue: Stable, Red: Unstable)")
ax2.set_xlabel(r"$\mu_e$");
ax2.set_ylabel(r"$m^*$")

# (C) 磁滞回线分析 (Hysteresis Loop)
ax3 = axes[1, 0]
mu_e_forward = np.linspace(0, 1.5, 100)
mu_e_backward = mu_e_forward[::-1]


def get_tracked_m(mu_range, m_init):
    res = []
    curr_m = m_init
    for me in mu_range:
        # 模拟动力学演化直到收敛
        for _ in range(20):
            curr_m = f_m(curr_m, 0.4, me, 0.12, 0.1, 0.1)
        res.append(curr_m)
    return res


m_up = get_tracked_m(mu_e_forward, -1.0)
m_down = get_tracked_m(mu_e_backward, 2.5)
ax3.plot(mu_e_forward, m_up, 'r', label="Increasing $\mu_e$")
ax3.plot(mu_e_backward, m_down, 'b--', label="Decreasing $\mu_e$")
ax3.set_title("3. Hysteresis Loop")
ax3.legend()

# (D) 稳定性斜率分析 (Stability Criterion)
ax4 = axes[1, 1]
pts_sample = analyze_dynamics(0.8)
m_fine = np.linspace(-1, 2, 100)
ax4.plot(m_fine, f_m(m_fine, 0.4, 0.8, 0.12, 0.1, 0.1), 'g-')
ax4.plot([-1, 2], [-1, 2], 'k--', alpha=0.5)
for p, s in pts_sample:
    ax4.plot(p, p, 'ko')
    ax4.annotate(f"slope={s:.2f}", (p, p + 0.2), fontsize=8)
ax4.set_title("4. Stability Check ($|f'(m)| < 1$)")

plt.show()