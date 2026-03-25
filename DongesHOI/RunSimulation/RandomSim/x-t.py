import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib  # 导入 matplotlib 主包以访问 colormaps 注册表


# ==========================================
# 1. PNAS 绘图风格配置
# ==========================================
def set_pnas_style():
    """
    配置符合 PNAS 期刊标准的绘图风格
    """
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 1.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        "lines.linewidth": 1.0,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.prop_cycle": plt.cycler(color=['#0072B2', '#D55E00', '#009E73', '#CC79A7'])
    })


set_pnas_style()


# ==========================================
# 2. 核心计算逻辑 (保持不变)
# ==========================================
def generate_parameters(s, mu_d, sigma_d, mu_e, sigma_e):
    d = np.random.normal(mu_d / s, sigma_d / np.sqrt(s), (s, s))
    np.fill_diagonal(d, 0)
    e = np.random.normal(mu_e / s ** 2, sigma_e / s, (s, s, s))
    for i in range(s):
        e[i, i, :] = 0;
        e[i, :, i] = 0;
        e[:, i, i] = 0
    return d, e


def sample_c(s):
    sigma = 2 * np.sqrt(3) / 27
    return np.random.normal(0, sigma, size=s)


@njit
def dxdt_numba(x, d, e, c):
    s = len(x)
    out = -x ** 3 + x + c + d @ x
    for i in range(s):
        acc = 0.0
        for j in range(s):
            for k in range(s):
                acc += e[i, j, k] * x[j] * x[k]
        out[i] += acc
    return out


@njit
def rk4_step_numba(x, d, e, c, dt):
    k1 = dxdt_numba(x, d, e, c)
    k2 = dxdt_numba(x + 0.5 * dt * k1, d, e, c)
    k3 = dxdt_numba(x + 0.5 * dt * k2, d, e, c)
    k4 = dxdt_numba(x + dt * k3, d, e, c)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def simulate_trajectory(s, t_steps, dt, d, e, c):
    x = np.full(s, -1.0)
    traj = np.zeros((t_steps, s))
    for t in range(t_steps):
        traj[t] = x
        x = rk4_step_numba(x, d, e, c, dt)
    return traj


# ==========================================
# 3. 主程序：PNAS 风格绘图逻辑 (已修复警告)
# ==========================================

if __name__ == "__main__":
    # --- 参数设置 ---
    s = 50
    t_steps = 6000
    dt = 0.01
    mu_d, sigma_d = -0.2, 0.3
    mu_e, sigma_e = 0.3, 0.3

    print(f"Generating data...")
    d_mat, e_mat = generate_parameters(s, mu_d, sigma_d, mu_e, sigma_e)
    c_vec = sample_c(s)
    traj = simulate_trajectory(s, t_steps, dt, d_mat, e_mat, c_vec)
    time_array = np.arange(t_steps) * dt

    print("Plotting in PNAS style...")

    fig, ax = plt.subplots(figsize=(3.42, 2.6))

    # --- 颜色策略 ---
    final_values = traj[-1, :]
    norm = mcolors.Normalize(vmin=np.min(final_values), vmax=np.max(final_values))

    # [修复点] 使用 matplotlib.colormaps['name'] 替代 get_cmap
    cmap = matplotlib.colormaps['coolwarm']

    # --- 排序绘制 ---
    sorted_indices = np.argsort(final_values)

    for idx in sorted_indices:
        ax.plot(time_array, traj[:, idx],
                linewidth=0.6,
                alpha=0.4,
                color=cmap(norm(final_values[idx])),
                rasterized=True
                )



    # --- 装饰 ---
    ax.set_xlabel('Time ($t$)', fontweight='normal')
    ax.set_ylabel('State variable $x_i(t)$')
    ax.set_xlim(0, t_steps * dt)

    y_min, y_max = np.min(traj), np.max(traj)
    ax.set_ylim(y_min - 0.1, y_max + 0.1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- 保存 ---
    filename_base = f"PNAS_Style_md{mu_d:.1f}_sd{sigma_d:.1f}_me{mu_e:.1f}_se{sigma_e:.1f}_XT"

    plt.savefig(f"{filename_base}.png", dpi=400)
    plt.savefig(f"{filename_base}.pdf")

    print(f"Done. Saved PNAS-style plot as '{filename_base}.pdf'")
    plt.show()