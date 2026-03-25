import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import jit
from joblib import Parallel, delayed
import time


# ==========================================================
# 0. PNAS 绘图风格配置 (学术发表级)
# ==========================================================
def set_pnas_style():
    """配置 Matplotlib 以符合 PNAS 风格"""
    plt.style.use('default')

    # PNAS 单栏宽度约为 3.42 英寸 (8.7cm)
    fig_width = 4.0
    fig_height = 3.2

    params = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'SimHei'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'mathtext.fontset': 'stixsans',
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.top': True,
        'ytick.right': True,
        'figure.figsize': [fig_width, fig_height],
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'axes.grid': False,
        'figure.autolayout': False,
    }
    mpl.rcParams.update(params)


set_pnas_style()


# ==========================================================
# 1. 动力学核心 (Numba 加速)
# ==========================================================

@jit(nopython=True, cache=True)
def compute_rhs(x, c, d_matrix, e_tensor):
    """计算朗之万方程右侧项"""
    S = x.shape[0]
    nonlinear_term = -x ** 3 + x
    d_term = d_matrix @ x
    e_term = np.zeros(S)
    if e_tensor.shape[0] > 0:
        for i in range(S):
            val = 0.0
            for j in range(S):
                for k in range(S):
                    val += e_tensor[i, j, k] * x[j] * x[k]
            e_term[i] = val
    return nonlinear_term + c + d_term + e_term


@jit(nopython=True, cache=True)
def rk4_step(x, dt, c, d_matrix, e_tensor):
    """四阶龙格-库塔积分步"""
    k1 = compute_rhs(x, c, d_matrix, e_tensor)
    k2 = compute_rhs(x + 0.5 * dt * k1, c, d_matrix, e_tensor)
    k3 = compute_rhs(x + 0.5 * dt * k2, c, d_matrix, e_tensor)
    k4 = compute_rhs(x + dt * k3, c, d_matrix, e_tensor)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@jit(nopython=True, cache=True)
def simulate_dynamics_jit(x_init, dt, t_steps, c, d_matrix, e_tensor):
    """执行完整的动力学演化"""
    x = x_init.copy()
    for t in range(t_steps):
        x = rk4_step(x, dt, c, d_matrix, e_tensor)
    return x


# ==========================================================
# 2. 模拟运行器 (处理随机矩阵与并行)
# ==========================================================

def run_single_simulation(seed, params, initial_x_mean, initial_x_std):
    """运行单次实验"""
    np.random.seed(seed)
    S = params['S']

    c = np.random.normal(params['mu_c'], params['sigma_c'], size=S)
    d_matrix = np.random.normal(params['mu_d'] / S, params['sigma_d'] / np.sqrt(S), size=(S, S))
    np.fill_diagonal(d_matrix, 0.0)

    if params['sigma_e'] == 0 and params['mu_e'] == 0:
        e_tensor = np.zeros((S, S, S))
    else:
        e_tensor = np.random.normal(params['mu_e'] / S ** 2, params['sigma_e'] / S, size=(S, S, S))
        idx = np.indices((S, S, S), sparse=True)
        mask = (idx[0] == idx[1]) | (idx[1] == idx[2]) | (idx[0] == idx[2])
        e_tensor[mask] = 0.0

    x_init = np.random.normal(initial_x_mean, initial_x_std, size=S)
    final_x = simulate_dynamics_jit(x_init, params['dt'], params['t_steps'], c, d_matrix, e_tensor)

    m = np.mean(final_x)
    q = np.mean(final_x ** 2)
    phi = np.mean(final_x > 0)
    return m, q, phi


def run_parallel_simulations(params, num_batches, initial_mean=-0.6, initial_std=0.01):
    """并行运行多次实验并取平均"""
    base_seed = int(time.time())
    results = Parallel(n_jobs=-1)(
        delayed(run_single_simulation)(i + base_seed, params, initial_mean, initial_std)
        for i in range(num_batches)
    )
    results = np.array(results)
    m_avg = np.mean(results[:, 0])
    q_avg = np.mean(results[:, 1])
    phi_avg = np.mean(results[:, 2])
    phi_std = np.std(results[:, 2], ddof=1) / np.sqrt(num_batches)
    return m_avg, q_avg, phi_avg, phi_std


# ==========================================================
# 3. 理论求解器 (自洽方程迭代 - 已修正平滑度问题)
# ==========================================================

def solve_cubic_root_hysteresis(h, current_m):
    """求解 x^3 - x - h = 0，选择离 current_m 最近的根"""
    coeffs = [1, 0, -1, -h]
    roots = np.roots(coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-6].real
    real_roots = np.sort(real_roots)

    if len(real_roots) == 0: return 0.0
    if len(real_roots) == 1: return real_roots[0]
    distances = np.abs(real_roots - current_m)
    return real_roots[np.argmin(distances)]


def theoretical_solver(params, z_samples, initial_guess_m=-0.6, tol=1e-5, max_iter=100):
    """
    利用空穴法/平均场理论计算稳态序参量。

    关键修正：z_samples (标准正态随机数) 现在作为参数传入。
    这保证了在扫描参数时，积分样本保持不变，从而消除曲线抖动。
    """
    m = initial_guess_m
    q = initial_guess_m ** 2
    damping = 0.2

    for it in range(max_iter):
        # 1. 计算局部场 h 的统计特征
        mu_h = params['mu_c'] + params['mu_d'] * m + params['mu_e'] * (m ** 2)/2.
        var_h = (params['sigma_c'] ** 2 +
                 params['sigma_d'] ** 2 * q +
                 params['sigma_e'] ** 2 * (q ** 2)/2.)

        # 防止方差为负（数值误差）
        if var_h < 0: var_h = 0
        sigma_h = np.sqrt(var_h)

        # 2. 构建局部场样本 (使用固定的 z_samples)
        h_eff = mu_h + sigma_h * z_samples

        # 3. 对每个 h 求解定点 x
        # 注意：这里列表推导式可能较慢，向量化求解会更快，但 roots 函数难以向量化。
        # 考虑到 num_samples 较大，这里是性能瓶颈，但对于平滑曲线是必要的。
        x_new_samples = np.array([solve_cubic_root_hysteresis(h, m) for h in h_eff])

        # 4. 更新序参量
        m_new = np.mean(x_new_samples)
        q_new = np.mean(x_new_samples ** 2)

        # 5. 检查收敛
        err = abs(m_new - m) + abs(q_new - q)
        if err < tol:
            phi = np.mean(x_new_samples > 0)
            return m, q, phi

        # 6. 阻尼更新
        m = (1 - damping) * m + damping * m_new
        q = (1 - damping) * q + damping * q_new

    phi = np.mean(x_new_samples > 0)
    return m, q, phi


# ==========================================================
# 4. 主程序：扫描与绘图 (PNAS 风格)
# ==========================================================

def run_experiment_sigma_d_scan():
    # 基础参数
    base_params = {
        'S': 200,
        'dt': 0.01,
        't_steps': 5000,
        'mu_c': 0.0, 'sigma_c': 0.124,
        'mu_d': 0.1,
        'mu_e': 0.1, 'sigma_e': 0.0
    }

    # 扫描 sigma_d
    sigma_d_values = np.linspace(0.0, 1.5, 16)  # 增加点数使曲线更平滑

    sim_phi_list = []
    sim_err_list = []
    theory_phi_list = []

    # ==========================================
    # 关键步骤：预生成固定的随机样本 (Frozen Noise)
    # ==========================================
    print("预生成理论计算用的固定样本...")
    num_theory_samples = 50000
    # 固定种子，确保每次运行程序结果一致
    rng = np.random.default_rng(seed=42)
    frozen_z_samples = rng.standard_normal(num_theory_samples)

    print(f"开始扫描 sigma_d (点数: {len(sigma_d_values)})...")
    start_time = time.time()

    for val in sigma_d_values:
        current_params = base_params.copy()
        current_params['sigma_d'] = val

        # 1. 运行模拟 (多次取平均)
        # 注意：模拟部分不需要冻结噪声，反而需要不同的种子来体现真实的统计误差
        _, _, phi_sim, phi_err = run_parallel_simulations(current_params, num_batches=12)
        sim_phi_list.append(phi_sim)
        sim_err_list.append(phi_err)

        # 2. 运行理论计算 (传入固定的 frozen_z_samples)
        _, _, phi_th = theoretical_solver(current_params, frozen_z_samples)
        theory_phi_list.append(phi_th)

        print(f"Sigma_d={val:.2f} | Sim_Phi={phi_sim:.3f} | Thy_Phi={phi_th:.3f}")

    print(f"计算完成，耗时: {time.time() - start_time:.2f} 秒")

    # ==========================================
    # 绘图逻辑
    # ==========================================
    color_sim = '#004488'  # 模拟数据：深蓝色
    color_thy = '#BB5566'  # 理论曲线：深红色

    fig, ax = plt.subplots()

    # 1. 绘制理论线
    ax.plot(sigma_d_values, theory_phi_list,
            linestyle='-',
            linewidth=2.0,
            color=color_thy,
            label='Theory (Mean Field)',
            zorder=1)

    # 2. 绘制模拟点 (带误差棒)
    ax.errorbar(sigma_d_values, sim_phi_list, yerr=sim_err_list,
                fmt='o',
                markersize=5,
                markeredgewidth=1.0,
                color=color_sim,
                ecolor=color_sim,
                capsize=3,
                elinewidth=1.0,
                label='Simulation',
                alpha=0.9,
                zorder=2)

    # 3. 标签与图例
    ax.set_xlabel(r'Std. Dev. of 2-body Interaction $\sigma_d$')
    ax.set_ylabel(r'Fraction of Positive States $\phi$')
    ax.legend(frameon=False, loc='best', handlelength=1.5)

    # 4. 坐标轴微调
    ax.set_xlim(left=0)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    # plt.savefig('pnas_figure_smooth.pdf', format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    run_experiment_sigma_d_scan()