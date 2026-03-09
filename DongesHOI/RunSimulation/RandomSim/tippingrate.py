import numpy as np
from numba import njit
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import time


# ==========================================
# 1. PNAS 绘图风格配置
# ==========================================
def set_pnas_style_no_latex():
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        "figure.constrained_layout.use": True,
        "axes.prop_cycle": plt.cycler(color=["#D55E00", "#0072B2", "#009E73"])
    })


def simple_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


set_pnas_style_no_latex()


# ==========================================
# 2. 核心计算逻辑 (Numba 加速)
# ==========================================

def generate_parameters(s, mu_d, sigma_d, mu_e, sigma_e):
    # 生成两体相互作用矩阵 D
    d = np.random.normal(mu_d / s, sigma_d / np.sqrt(s), (s, s))
    np.fill_diagonal(d, 0)

    # 生成三体相互作用张量 E
    if mu_e == 0 and sigma_e == 0:
        e = np.zeros((s, s, s))
    else:
        e = np.random.normal(mu_e / s ** 2, sigma_e / s, (s, s, s))
        # 移除自相互作用 (简化处理，实际上全零矩阵不需要这一步，但为了通用性保留)
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
    # 基础部分：-x^3 + x + c + D*x
    out = -x ** 3 + x + c + d @ x

    # 高阶部分：E*x*x
    # 只有当 e 不全为 0 时才需要计算这个昂贵的三重循环
    # 简单的检查方法是看第一个元素是否为0 (假设全零矩阵) 或者传入标志位
    # 这里为了代码简洁，直接运算。如果 e 是全零，Numba 优化后开销也不大。
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
    x = np.full(s, -0.6)  # 初始状态
    traj = np.zeros((t_steps, s))
    for t in range(t_steps):
        traj[t] = x
        x = rk4_step_numba(x, d, e, c, dt)
    return traj


# ==========================================
# 3. 单次实验控制逻辑 (供并行调用)
# ==========================================

def run_single_experiment(args):
    """
    包装函数，用于 Multiprocessing。
    args 是一个元组，包含所有需要的参数。
    """
    s, t_steps, dt, mu_d, sigma_d, mu_e, sigma_e, probe_indices = args

    # 重新生成随机种子，确保并行进程中的随机性不同
    # (在较新的 numpy 版本中，多进程通常会自动处理，但显式调用更安全)
    np.random.seed()

    # 1. 生成参数
    d, e = generate_parameters(s, mu_d, sigma_d, mu_e, sigma_e)
    c = sample_c(s)

    # 2. 运行模拟
    traj = simulate_trajectory(s, t_steps, dt, d, e, c)

    # 3. 计算比率
    ratios = []
    for pt in probe_indices:
        if pt < t_steps:
            ratios.append(np.mean(traj[pt] > 0))
        else:
            ratios.append(np.nan)

    return np.array(ratios)


# ==========================================
# 4. 批量实验管理器
# ==========================================

def run_batch_experiments(num_repeats, s, t_steps, dt, mu_d, sigma_d, mu_e, sigma_e, probe_indices):
    """
    运行 num_repeats 次实验，并返回所有结果的矩阵
    """
    # 准备参数列表
    param_tuple = (s, t_steps, dt, mu_d, sigma_d, mu_e, sigma_e, probe_indices)
    args_list = [param_tuple] * num_repeats

    # 使用多进程并行计算
    # 自动检测 CPU 核心数，留一个核心备用以免卡死系统
    n_cores = max(1, cpu_count() - 1)
    print(f"  > Starting batch simulation with {num_repeats} repeats on {n_cores} cores...")

    with Pool(processes=n_cores) as pool:
        results = pool.map(run_single_experiment, args_list)

    # results 是一个 list of arrays, 转换为 2D array: (num_repeats, num_probe_points)
    return np.array(results)


# ==========================================
# 5. 主程序
# ==========================================

if __name__ == "__main__":
    # --- 通用参数 ---
    s = 100
    t_steps = 6000
    dt = 0.01
    num_repeats = 100  # 重复次数

    # 探测点
    probe_indices = np.arange(0, t_steps, 50)  # 稍微增加采样密度
    probe_times = probe_indices * dt

    # --- 实验 A: 包含高阶相互作用 (HOI) ---
    mu_d_A = 0.2
    sigma_d_A = 0.5
    mu_e_A = 0.2
    sigma_e_A = 0.5

    t0 = time.time()
    print(f"--- Experiment A (With HOI) ---")
    data_A = run_batch_experiments(num_repeats, s, t_steps, dt, mu_d_A, sigma_d_A, mu_e_A, sigma_e_A, probe_indices)

    # 计算统计量
    mean_A = np.mean(data_A, axis=0)
    std_A = np.std(data_A, axis=0)
    # 如果你想用标准误 (SEM) 而不是标准差，可以使用: std_A = np.std(data_A, axis=0) / np.sqrt(num_repeats)

    # --- 实验 B: 仅两体相互作用 (Baseline) ---
    mu_e_B = 0.0
    sigma_e_B = 0.0

    print(f"--- Experiment B (Baseline) ---")
    data_B = run_batch_experiments(num_repeats, s, t_steps, dt, mu_d_A, sigma_d_A, mu_e_B, sigma_e_B, probe_indices)

    # 计算统计量
    mean_B = np.mean(data_B, axis=0)
    std_B = np.std(data_B, axis=0)

    print(f"Total simulation time: {time.time() - t0:.2f} seconds")

    # --- 绘图 ---
    print("Plotting with error bands...")

    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    simple_axis(ax)

    # 绘制曲线 A (HOI)
    ax.plot(probe_times, mean_A,
            label=f'With HOI',
            color='#D55E00', linewidth=2)
    # 绘制误差带 (Mean +/- Std)
    ax.fill_between(probe_times, mean_A - std_A, mean_A + std_A,
                    color='#D55E00', alpha=0.2, edgecolor=None)

    # 绘制曲线 B (Baseline)
    ax.plot(probe_times, mean_B,
            label=f'Pairwise Only',
            color='#0072B2', linewidth=2, linestyle='--')
    # 绘制误差带
    ax.fill_between(probe_times, mean_B - std_B, mean_B + std_B,
                    color='#0072B2', alpha=0.2, edgecolor=None)

    # 装饰
    ax.set_xlabel('Time (t)')
    ax.set_ylabel(r'Tipping Rate $\phi$')
    ax.set_ylim(0, 0.7)
    ax.set_xlim(0, t_steps * dt)

    ax.legend(frameon=False, loc='upper left')
    #ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.savefig("tipping_rate_comparison_with_errorbars.png", dpi=300)
    plt.show()
    print("Done.")