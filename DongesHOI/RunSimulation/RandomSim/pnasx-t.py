import numpy as np
from numba import njit
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib as mpl


# ==========================================
# 1. PNAS 绘图风格配置 (无需 LaTeX)
# ==========================================
def set_pnas_style_no_latex():
    """
    配置 Matplotlib 以符合 PNAS (Proceedings of the National Academy of Sciences) 风格。
    特点：无衬线字体(Arial)、刻度朝外、简约边框。
    """
    plt.rcParams.update({
        # --- 核心字体设置 ---
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",  # 数学公式也使用无衬线风格

        # --- 字体大小 ---
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,

        # --- 线条和刻度 ---
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.5,
        "xtick.direction": "out",  # PNAS 刻度朝外
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.top": False,  # 顶部无刻度
        "ytick.right": False,  # 右侧无刻度

        # --- 布局 ---
        "figure.constrained_layout.use": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",

        # --- 颜色 (色盲友好/现代配色) ---
        "axes.prop_cycle": plt.cycler(color=[
            "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"
        ])
    })


def simple_axis(ax):
    """辅助函数：移除顶部和右侧边框 (PNAS 常见风格)"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# 应用样式
set_pnas_style_no_latex()


# ==========================================
# 2. 参数生成与计算核心 (保持不变)
# ==========================================

def generate_parameters(s, mu_d, sigma_d, mu_e, sigma_e):
    d = np.random.normal(mu_d / s, sigma_d / np.sqrt(s), (s, s))
    np.fill_diagonal(d, 0)
    e = np.random.normal(mu_e / s ** 2, sigma_e / s, (s, s, s))
    for i in range(s):
        e[i, i, :] = 0
        e[i, :, i] = 0
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


def simulate_once(args):
    s, t_steps, dt, mu_d, sigma_d, mu_e, sigma_e = args
    d, e = generate_parameters(s, mu_d, sigma_d, mu_e, sigma_e)
    c = sample_c(s)
    x = np.full(s, -0.6)
    for _ in range(t_steps):
        x = rk4_step_numba(x, d, e, c, dt)
    return x


def run_parallel(batch, s, t_steps, dt, mu_d, sigma_d, mu_e, sigma_e, n_jobs=4):
    args_list = [(s, t_steps, dt, mu_d, sigma_d, mu_e, sigma_e) for _ in range(batch)]
    with Pool(n_jobs) as pool:
        results = pool.map(simulate_once, args_list)
    return np.array(results)


# ==========================================
# 3. 主程序与绘图
# ==========================================

if __name__ == "__main__":
    # --- 系统参数 ---
    s = 50
    t_steps = 4000
    dt = 0.01

    mu_d = 0.2
    sigma_d = 0.4
    mu_e = 0.0
    sigma_e = 0.0

    print("Step 1: Generating parameters...")
    d, e = generate_parameters(s, mu_d, sigma_d, mu_e, sigma_e)
    c = sample_c(s)
    traj = simulate_trajectory(s, t_steps, dt, d, e, c)
    time_array = np.arange(t_steps) * dt

    print("Step 2: Running batch simulations...")
    xs_final = run_parallel(20, s, t_steps, dt, mu_d, sigma_d, mu_e, sigma_e, n_jobs=4)

    # --- 数据处理 ---
    probe_indices = [200, 400, 600, 800, 1000, 1200, 1500, 1800, 2000, 2500, 3000, 3500]
    probe_times = np.array(probe_indices) * dt
    ratios = []
    for pt in probe_indices:
        if pt < t_steps:
            ratios.append(np.mean(traj[pt] > 0))
        else:
            ratios.append(np.nan)

    print("Step 3: Plotting in PNAS style...")

    # PNAS 单栏宽度约 3.42 英寸，双栏约 7 英寸
    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.25)

    # --- 子图 (a): 时间序列 ---
    ax1 = fig.add_subplot(gs[0, :])
    simple_axis(ax1)  # 应用去边框样式

    # 使用 PNAS 风格颜色
    for i in range(s):
        ax1.plot(time_array, traj[:, i], linewidth=0.8, alpha=0.5, color='#0072B2')

    for pt_time in probe_times:
        ax1.axvline(x=pt_time, color='gray', linestyle='--', linewidth=0.8, alpha=0.8)

    ax1.set_xlabel('Time (t)')  # PNAS 通常不用斜体变量名作为轴标签，除非特指
    ax1.set_ylabel('State x')
    ax1.set_xlim(0, t_steps * dt)
    # PNAS 标签通常是粗体小写或大写，放在左上角外部
    #ax1.text(-0.06, 1.05, 'A', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

    # --- 子图 (b): 最终状态分布 ---
    ax2 = fig.add_subplot(gs[1, 0])
    simple_axis(ax2)
    flat = xs_final.flatten()
    ax2.hist(flat, bins=40, density=True, color='gray', alpha=0.7, edgecolor='none')  # PNAS 常用无边框直方图
    ax2.set_xlabel('State x')
    ax2.set_ylabel('Probability Density')
    ax2.text(-0.15, 1.05, 'B', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

    # --- 子图 (c): 比例演化 ---
    ax3 = fig.add_subplot(gs[1, 1])
    simple_axis(ax3)
    ax3.plot(probe_times, ratios, marker='o', markersize=5, linestyle='-',
             color='#D55E00', linewidth=1.5, markerfacecolor='white', markeredgewidth=1.5)
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('Ratio (x > 0)')
    ax3.set_ylim(0.0, 0.6)
    ax3.text(-0.15, 1.05, 'C', transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

    plt.savefig("pnas_results.pdf")
    plt.savefig("pnas_results.png", dpi=300)

    print("Figures saved as 'pnas_results.pdf' (vector) and 'pnas_results.png'.")
    plt.show()