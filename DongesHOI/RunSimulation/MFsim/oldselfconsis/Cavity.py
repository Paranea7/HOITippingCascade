import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from joblib import Parallel, delayed
import time
from scipy.stats import norm


# ==========================================
# 1. Numba 加速的核心计算函数 (动力学模拟)
# ==========================================

@jit(nopython=True, cache=True)
def compute_rhs(x, c, d_matrix, e_tensor):
    """
    计算 dx/dt 的右侧项。
    方程: dx/dt = -x^3 + x + c + D@x + E@x@x
    """
    # 1. 非线性项: -x^3 + x
    nonlinear_term = -x ** 3 + x

    # 2. 二体相互作用: D @ x
    d_term = d_matrix @ x

    # 3. 三体相互作用 (如果存在)
    S = x.shape[0]
    e_term = np.zeros(S, dtype=np.float64)

    if e_tensor.shape[0] > 0 and np.any(e_tensor != 0):
        for i in range(S):
            val = 0.0
            for j in range(S):
                for k in range(S):
                    val += e_tensor[i, j, k] * x[j] * x[k]
            e_term[i] = val

    return nonlinear_term + c + d_term + e_term


@jit(nopython=True, cache=True)
def rk4_step(x, dt, c, d_matrix, e_tensor):
    """
    执行单步 RK4 积分
    """
    k1 = compute_rhs(x, c, d_matrix, e_tensor)
    k2 = compute_rhs(x + 0.5 * dt * k1, c, d_matrix, e_tensor)
    k3 = compute_rhs(x + 0.5 * dt * k2, c, d_matrix, e_tensor)
    k4 = compute_rhs(x + dt * k3, c, d_matrix, e_tensor)

    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@jit(nopython=True, cache=True)
def simulate_dynamics_jit(x_init, dt, t_steps, c, d_matrix, e_tensor):
    """
    运行整个时间演化过程。
    """
    x = x_init.copy()
    S = x.shape[0]

    # 预分配历史数组
    m_history = np.zeros(t_steps, dtype=np.float64)
    q_history = np.zeros(t_steps, dtype=np.float64)

    for t in range(t_steps):
        x = rk4_step(x, dt, c, d_matrix, e_tensor)

        # 计算序参量
        m_val = np.sum(x) / S
        q_val = np.sum(x ** 2) / S

        m_history[t] = m_val
        q_history[t] = q_val

    return x, m_history, q_history


# ==========================================
# 2. 模拟类与并行辅助函数
# ==========================================

def run_single_simulation(seed, params, initial_x_mean, initial_x_std):
    """
    单个模拟任务。
    """
    np.random.seed(seed)
    S = params['S']

    # --- 生成淬火无序参数 ---
    # c ~ N(mu_c, sigma_c^2)
    c = np.random.normal(params['mu_c'], params['sigma_c'], size=S)

    # D_ij ~ N(mu_d/S, sigma_d^2/S)
    d_matrix = np.random.normal(params['mu_d'] / S, params['sigma_d'] / np.sqrt(S), size=(S, S))
    np.fill_diagonal(d_matrix, 0)

    # E_ijk (仅当 sigma_e != 0 时生成)
    if params['sigma_e'] == 0:
        e_tensor = np.zeros((S, S, S))
    else:
        e_tensor = np.random.normal(params['mu_e'] / (S ** 2), params['sigma_e'] / S, size=(S, S, S))

    # --- 初始化状态 ---
    x_init = np.random.normal(initial_x_mean, initial_x_std, size=S)

    # --- 运行动力学 ---
    final_x, m_hist, q_hist = simulate_dynamics_jit(
        x_init, params['dt'], params['t_steps'], c, d_matrix, e_tensor
    )

    final_m = np.mean(final_x)
    final_q = np.mean(final_x ** 2)

    # 计算 phi (存活/吸引域比例)
    if initial_x_mean > 0:
        phi = np.sum(final_x > 0) / S
    else:
        phi = np.sum(final_x < 0) / S

    return final_x, final_m, final_q, phi, m_hist, q_hist


def run_multiple_batches_parallel(num_batches, params, initial_x_mean, initial_x_std):
    """
    并行运行多个批次模拟。
    """
    print(f"\n--- Running {num_batches} batches (Parallel, RK4, Initial mean={initial_x_mean}) ---")
    start_time = time.time()

    results = Parallel(n_jobs=-1)(
        delayed(run_single_simulation)(i, params, initial_x_mean, initial_x_std)
        for i in range(num_batches)
    )

    end_time = time.time()
    print(f"Simulation finished in {end_time - start_time:.2f} seconds.")

    # 整理结果
    all_final_x = []
    all_final_m = []
    all_final_q = []
    all_phi = []
    last_m_hist = None
    last_q_hist = None

    for res in results:
        final_x, final_m, final_q, phi, m_hist, q_hist = res
        all_final_x.extend(final_x)
        all_final_m.append(final_m)
        all_final_q.append(final_q)
        all_phi.append(phi)
        last_m_hist = m_hist
        last_q_hist = q_hist

    mean_m = np.mean(all_final_m)
    mean_q = np.mean(all_final_q)
    mean_phi = np.mean(all_phi)
    std_m = np.std(all_final_m)
    std_phi = np.std(all_phi)

    print(f"Simulation Results for Initial x ~ {initial_x_mean}:")
    print(f"  Average m   = {mean_m:.4f} (std: {std_m:.4f})")
    print(f"  Average q   = {mean_q:.4f}")
    print(f"  Average phi = {mean_phi:.4f} (std: {std_phi:.4f})")

    return np.array(all_final_x), mean_m, mean_q, mean_phi, last_m_hist, last_q_hist


# ==========================================
# 3. 理论计算函数 (平均场自洽方程)
# ==========================================

def solve_for_x_stable(h, initial_guess):
    """
    求解 x^3 - x - h = 0 的稳定根。
    使用 np.roots 直接求解多项式根，避免 fsolve 的收敛问题。
    """
    # 多项式系数: 1*x^3 + 0*x^2 - 1*x - h = 0
    coeffs = [1, 0, -1, -h]

    # 获取所有根
    roots = np.roots(coeffs)

    # 1. 筛选实数根 (虚部极小视为实数)
    real_roots = roots[np.abs(roots.imag) < 1e-6].real

    if len(real_roots) == 0:
        return np.nan

    # 2. 根据分支选择根 (迟滞效应)
    # 如果初始在正分支，倾向于停留在正的势阱 (最大的根)
    # 如果初始在负分支，倾向于停留在负的势阱 (最小的根)
    if initial_guess > 0:
        return np.max(real_roots)
    else:
        return np.min(real_roots)


def calculate_theoretical_properties(params, initial_x_mean_guess, num_h_samples=50000, max_iter=1000, tol=1e-5,
                                     damping=0.2):
    """
    计算理论上的 m, q, phi 和 x 的分布。
    """
    if params['sigma_e'] != 0:
        print("Warning: Theoretical calculation assumes sigma_e = 0.")

    mu_c = params['mu_c']
    sigma_c = params['sigma_c']
    mu_d = params['mu_d']
    sigma_d = params['sigma_d']

    # 初始猜测
    m_th = initial_x_mean_guess
    q_th = initial_x_mean_guess ** 2

    print(f"\n--- Calculating Theoretical Properties (Initial guess={initial_x_mean_guess}) ---")

    # 预先生成标准正态分布样本 z ~ N(0, 1)
    # 固定随机数流有助于迭代平滑收敛
    z_samples = np.random.normal(0, 1, num_h_samples)

    for i in range(max_iter):
        # 计算有效场 h 的分布参数
        # h ~ N(mu_h, sigma_h^2)
        mu_h = mu_c + mu_d * m_th
        sigma_h_sq = sigma_c ** 2 + sigma_d ** 2 * q_th
        sigma_h = np.sqrt(max(0, sigma_h_sq))

        # 生成 h 样本
        h_samples = mu_h + sigma_h * z_samples

        # 求解 x (这是最耗时的部分)
        x_samples = np.array([solve_for_x_stable(h_val, initial_x_mean_guess) for h_val in h_samples])

        # 去除无效值
        x_samples = x_samples[~np.isnan(x_samples)]

        if len(x_samples) == 0:
            print("Error: No valid roots found.")
            return np.nan, np.nan, np.nan, np.array([])

        m_new = np.mean(x_samples)
        q_new = np.mean(x_samples ** 2)

        # 检查收敛
        diff = abs(m_new - m_th) + abs(q_new - q_th)
        if diff < tol:
            print(f"Theoretical calculation converged in {i + 1} iterations. (Diff: {diff:.6e})")
            break

        # 阻尼更新: m = (1-alpha)*m_old + alpha*m_new
        m_th = (1 - damping) * m_th + damping * m_new
        q_th = (1 - damping) * q_th + damping * q_new
    else:
        print(f"Warning: Theoretical calculation did not converge within {max_iter} iterations.")

    # 计算 phi
    if initial_x_mean_guess > 0:
        phi_th = np.sum(x_samples > 0) / len(x_samples)
    else:
        phi_th = np.sum(x_samples < 0) / len(x_samples)

    print(f"Theoretical Results for Initial x ~ {initial_x_mean_guess}:")
    print(f"  Theoretical m   = {m_th:.4f}")
    print(f"  Theoretical q   = {q_th:.4f}")
    print(f"  Theoretical phi = {phi_th:.4f}")

    return m_th, q_th, phi_th, x_samples


# ==========================================
# 4. 绘图与主程序
# ==========================================

def plot_results(all_final_x, sim_m, sim_q, sim_phi, m_hist, q_hist, dt,
                 theoretical_x_samples, th_m, th_q, th_phi, title_prefix=""):
    plt.figure(figsize=(16, 6))

    # 1. PDF 分布图 (对比模拟与理论)
    plt.subplot(1, 2, 1)

    # 模拟分布 (直方图)
    plt.hist(all_final_x, bins=100, density=True, alpha=0.6, color='skyblue', edgecolor='white', label='Simulation')

    # 理论分布 (阶梯状线条，清晰展示轮廓)
    plt.hist(theoretical_x_samples, bins=100, density=True, alpha=1.0, color='red',
             histtype='step', linewidth=2, label='Theory (Mean Field)')

    plt.title(f"{title_prefix} Abundance PDF\n"
              f"Sim: m={sim_m:.2f}, q={sim_q:.2f}, $\phi$={sim_phi:.2f}\n"
              f"Th:  m={th_m:.2f}, q={th_q:.2f}, $\phi$={th_phi:.2f}")
    plt.xlabel("$x_i$")
    plt.ylabel("P(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. 时间演化图
    plt.subplot(1, 2, 2)
    time_axis = np.arange(len(m_hist)) * dt
    plt.plot(time_axis, m_hist, label='m(t)', color='blue')
    plt.plot(time_axis, q_hist, label='q(t)', color='red', linestyle='--')
    plt.axhline(y=th_m, color='blue', linestyle=':', alpha=0.6, label='m (Theory)')
    plt.axhline(y=th_q, color='red', linestyle=':', alpha=0.6, label='q (Theory)')

    plt.title(f"{title_prefix} Dynamics (Single Run - RK4)")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- 1. 设置参数 ---
    sim_params = {
        'S': 100,  # 系统大小 (S越大，模拟越接近理论)
        'dt': 0.01,  # 时间步长
        't_steps': 3000,  # 总步数
        'mu_c': 0.0,
        'sigma_c': 0.2,
        'mu_d': 0.3,  # 相互作用均值
        'sigma_d': 0.4,  # 相互作用方差
        'mu_e': 0.1,
        'sigma_e': 0.0  # 保持为0以匹配理论
    }

    num_batches = 30  # 模拟批次

    # --- 2. 模拟正分支 (Initial x ~ 1.0) ---
    initial_x_mean_pos = 1.0
    initial_x_std_pos = 0.1

    # 运行模拟
    all_x_pos, sim_m_pos, sim_q_pos, sim_phi_pos, m_hist_pos, q_hist_pos = run_multiple_batches_parallel(
        num_batches, sim_params, initial_x_mean_pos, initial_x_std_pos
    )

    # 运行理论计算
    th_m_pos, th_q_pos, th_phi_pos, theoretical_x_samples_pos = calculate_theoretical_properties(
        sim_params, initial_x_mean_pos
    )

    # 绘图
    plot_results(all_x_pos, sim_m_pos, sim_q_pos, sim_phi_pos, m_hist_pos, q_hist_pos, sim_params['dt'],
                 theoretical_x_samples_pos, th_m_pos, th_q_pos, th_phi_pos, "Positive Branch")

    # --- 3. 模拟负分支 (Initial x ~ -1.0) ---
    initial_x_mean_neg = -0.6
    initial_x_std_neg = 0.1

    # 运行模拟
    all_x_neg, sim_m_neg, sim_q_neg, sim_phi_neg, m_hist_neg, q_hist_neg = run_multiple_batches_parallel(
        num_batches, sim_params, initial_x_mean_neg, initial_x_std_neg
    )

    # 运行理论计算
    th_m_neg, th_q_neg, th_phi_neg, theoretical_x_samples_neg = calculate_theoretical_properties(
        sim_params, initial_x_mean_neg
    )

    # 绘图
    plot_results(all_x_neg, sim_m_neg, sim_q_neg, sim_phi_neg, m_hist_neg, q_hist_neg, sim_params['dt'],
                 theoretical_x_samples_neg, th_m_neg, th_q_neg, th_phi_neg, "Negative Branch")