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
        "axes.prop_cycle": plt.cycler(color=["#0072B2", "#D55E00", "#009E73"])  # 蓝, 橙, 绿
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
        # 如果参数为0，生成全零张量，节省计算资源
        e = np.zeros((s, s, s))
    else:
        e = np.random.normal(mu_e / s ** 2, sigma_e / s, (s, s, s))
        # 移除自相互作用
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
    # 检查张量是否全零的简单方法是看第一个元素 (假设全零矩阵)
    # 但为了稳健性，我们依赖 Numba 的优化。如果 E 全是 0，虽然循环会跑，但乘法结果为 0。
    # 为了极致性能，可以传入一个 flag，但这里为了代码简洁直接计算。
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


@njit
def simulate_and_check_backtipping(s, t_steps, dt, d, e, c):
    """
    专门为 Back-tipping 优化的模拟函数。
    """
    x = np.full(s, -0.6)  # 初始状态

    # 记录每个节点是否曾经 > 0
    has_tipped_once = np.zeros(s, dtype=np.bool_)

    for t in range(t_steps):
        x = rk4_step_numba(x, d, e, c, dt)

        # 检查当前步是否有节点 > 0
        for i in range(s):
            if x[i] > 0:
                has_tipped_once[i] = True

    return x, has_tipped_once


# ==========================================
# 3. 单次实验控制逻辑
# ==========================================

def run_single_backtipping_experiment(args):
    """
    计算单次实验的 Back-tipping rate
    """
    s, t_steps, dt, mu_d, sigma_d, mu_e, sigma_e = args
    np.random.seed()  # 确保并行进程的随机性不同

    d, e = generate_parameters(s, mu_d, sigma_d, mu_e, sigma_e)
    c = sample_c(s)

    # 运行模拟
    final_x, has_tipped_once = simulate_and_check_backtipping(s, t_steps, dt, d, e, c)

    # === 核心定义 ===
    # Back-tipping: 曾经 > 0 (has_tipped_once) 并且 最终 < 0 (final_x < 0)
    is_back_tipped = np.logical_and(has_tipped_once, final_x < 0)

    # 计算比例
    back_tipping_rate = np.mean(is_back_tipped)

    return back_tipping_rate


# ==========================================
# 4. 批量实验管理器
# ==========================================

def run_parameter_scan(num_repeats, s, t_steps, dt, scan_param_name, scan_values, fixed_params, label_desc):
    """
    扫描参数并计算 Back-tipping rate
    """
    results_mean = []
    results_std = []

    n_cores = max(1, cpu_count() - 1)

    print(f"Starting scan for scenario: [{label_desc}]")

    for val in scan_values:
        # 构建当前参数集
        current_params = fixed_params.copy()
        current_params[scan_param_name] = val

        # 解包参数
        mu_d = current_params['mu_d']
        sigma_d = current_params['sigma_d']
        mu_e = current_params['mu_e']
        sigma_e = current_params['sigma_e']

        # 准备多进程参数
        args_tuple = (s, t_steps, dt, mu_d, sigma_d, mu_e, sigma_e)
        args_list = [args_tuple] * num_repeats

        with Pool(processes=n_cores) as pool:
            batch_results = pool.map(run_single_backtipping_experiment, args_list)

        batch_results = np.array(batch_results)
        mean_val = np.mean(batch_results)
        std_val = np.std(batch_results)  # 也可以用 sem: std / sqrt(N)

        results_mean.append(mean_val)
        results_std.append(std_val)
        # print(f"    {scan_param_name}={val:.2f} -> Rate={mean_val:.3f}")

    return np.array(results_mean), np.array(results_std)


# ==========================================
# 5. 主程序
# ==========================================

if __name__ == "__main__":
    # --- 通用参数 ---
    s = 50  # 节点数
    t_steps = 6000  # 模拟步数
    dt = 0.01
    num_repeats = 200  # 每个点的重复实验次数 (增大此值曲线更平滑)

    # 扫描变量: 两体相互作用强度 sigma_d
    scan_values = np.linspace(0.1, 1.2, 12)
    scan_param_name = 'sigma_d'

    # --- 定义三组实验场景 ---
    scenarios = [
        {
            "label": "No HOI",
            "color": "#0072B2",  # 蓝色
            "params": {'mu_d': 0.3, 'sigma_d': 0.0, 'mu_e': 0.0, 'sigma_e': 0.0}
        },
        {
            "label": "HOI (Mean Only)",
            "color": "#D55E00",  # 橙色
            "params": {'mu_d': 0.3, 'sigma_d': 0.0, 'mu_e': 0.2, 'sigma_e': 0.0}
        },
        {
            "label": "HOI (Mean + Var)",
            "color": "#009E73",  # 绿色
            "params": {'mu_d': 0.3, 'sigma_d': 0.0, 'mu_e': 0.2, 'sigma_e': 0.5}
        }
    ]

    t0 = time.time()
    results_storage = []

    # --- 循环运行三组实验 ---
    for scenario in scenarios:
        fixed_params = scenario["params"]
        label = scenario["label"]

        means, stds = run_parameter_scan(
            num_repeats, s, t_steps, dt,
            scan_param_name=scan_param_name,
            scan_values=scan_values,
            fixed_params=fixed_params,
            label_desc=label
        )

        # 存储结果以便绘图
        results_storage.append({
            "label": label,
            "color": scenario["color"],
            "means": means,
            "stds": stds
        })

    print(f"Total simulation time: {time.time() - t0:.2f} seconds")

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    simple_axis(ax)

    for res in results_storage:
        # 绘制主曲线
        ax.plot(scan_values, res["means"],
                marker='o', markersize=4,
                label=res["label"],
                color=res["color"], linewidth=1.5)

        # 绘制误差带
        ax.fill_between(scan_values,
                        res["means"] - res["stds"],
                        res["means"] + res["stds"],
                        color=res["color"], alpha=0.2, edgecolor=None)

    ax.set_xlabel(r'Pairwise Interaction Strength $\sigma_d$')
    ax.set_ylabel(r'Back-tipping Rate')

    # 自动调整 Y 轴范围，留一点余量
    # ax.set_ylim(bottom=0)

    ax.legend(frameon=False, loc='best')
    plt.tight_layout()

    filename = "back_tipping_comparison_3groups.png"
    plt.savefig(filename, dpi=300)
    print(f"Plot saved to {filename}")
    plt.show()