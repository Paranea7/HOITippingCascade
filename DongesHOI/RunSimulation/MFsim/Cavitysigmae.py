import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from joblib import Parallel, delayed
import time
from scipy.stats import gaussian_kde


# ==========================================
# 1. Numba 加速的核心计算函数 (动力学模拟)
# ==========================================

@jit(nopython=True, cache=True)
def compute_rhs(x, c, d_matrix, e_tensor):
    """
    计算 dx/dt 的右侧项。
    方程: dx/dt = -x^3 + x + c + D@x + E@x@x
    """
    nonlinear_term = -x ** 3 + x
    d_term = d_matrix @ x

    S = x.shape[0]
    e_term = np.zeros(S, dtype=np.float64)

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
    """执行单步 RK4 积分"""
    k1 = compute_rhs(x, c, d_matrix, e_tensor)
    k2 = compute_rhs(x + 0.5 * dt * k1, c, d_matrix, e_tensor)
    k3 = compute_rhs(x + 0.5 * dt * k2, c, d_matrix, e_tensor)
    k4 = compute_rhs(x + dt * k3, c, d_matrix, e_tensor)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@jit(nopython=True, cache=True)
def simulate_dynamics_jit(x_init, dt, t_steps, c, d_matrix, e_tensor):
    """运行整个时间演化过程"""
    x = x_init.copy()
    S = x.shape[0]
    m_history = np.zeros(t_steps, dtype=np.float64)
    q_history = np.zeros(t_steps, dtype=np.float64)

    for t in range(t_steps):
        x = rk4_step(x, dt, c, d_matrix, e_tensor)
        m_val = np.sum(x) / S
        q_val = np.sum(x ** 2) / S
        m_history[t] = m_val
        q_history[t] = q_val

    return x, m_history, q_history


# ==========================================
# 2. 模拟类与并行辅助函数
# ==========================================

def run_single_simulation(seed, params, initial_x_mean, initial_x_std):
    np.random.seed(seed)
    S = params['S']

    c = np.random.normal(params['mu_c'], params['sigma_c'], size=S)
    d_matrix = np.random.normal(params['mu_d'] / S, params['sigma_d'] / np.sqrt(S), size=(S, S))
    np.fill_diagonal(d_matrix, 0)

    if params['sigma_e'] == 0 and params['mu_e'] == 0:
        e_tensor = np.zeros((0, 0, 0))
    else:
        e_tensor = np.random.normal(params['mu_e'] / (S ** 2), params['sigma_e'] / S, size=(S, S, S))

    x_init = np.random.normal(initial_x_mean, initial_x_std, size=S)

    final_x, m_hist, q_hist = simulate_dynamics_jit(
        x_init, params['dt'], params['t_steps'], c, d_matrix, e_tensor
    )

    final_m = np.mean(final_x)
    final_q = np.mean(final_x ** 2)

    if initial_x_mean > 0:
        phi = np.sum(final_x > 0) / S
    else:
        phi = np.sum(final_x < 0) / S

    return final_x, final_m, final_q, phi, m_hist, q_hist


def run_multiple_batches_parallel(num_batches, params, initial_x_mean, initial_x_std):
    print(f"\n--- Running {num_batches} batches (Parallel, RK4, Initial mean={initial_x_mean}) ---")
    start_time = time.time()

    results = Parallel(n_jobs=-1)(
        delayed(run_single_simulation)(i, params, initial_x_mean, initial_x_std)
        for i in range(num_batches)
    )

    end_time = time.time()
    print(f"Simulation finished in {end_time - start_time:.2f} seconds.")

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
# 3. 理论计算函数
# ==========================================

def solve_for_x_stable(h, initial_guess):
    coeffs = [1, 0, -1, -h]
    roots = np.roots(coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-6].real

    if len(real_roots) == 0:
        return np.nan

    if initial_guess > 0:
        return np.max(real_roots)
    else:
        return np.min(real_roots)


def calculate_theoretical_properties(params, initial_x_mean_guess, num_h_samples=50000, max_iter=2000, tol=1e-5,
                                     damping=0.2):
    mu_c = params['mu_c']
    sigma_c = params['sigma_c']
    mu_d = params['mu_d']
    sigma_d = params['sigma_d']
    mu_e = params.get('mu_e')
    sigma_e = params.get('sigma_e')

    m_th = initial_x_mean_guess
    q_th = initial_x_mean_guess ** 2

    print(f"\n--- Calculating Theoretical Properties (Initial guess={initial_x_mean_guess}) ---")
    z_samples = np.random.normal(0, 1, num_h_samples)

    for i in range(max_iter):
        mu_h = mu_c + mu_d * m_th + mu_e * (m_th ** 2)
        sigma_h_sq = (sigma_c ** 2) + (sigma_d ** 2 * q_th) + (sigma_e ** 2 * (q_th ** 2))
        sigma_h = np.sqrt(max(0, sigma_h_sq))

        h_samples = mu_h + sigma_h * z_samples
        x_samples = np.array([solve_for_x_stable(h_val, initial_x_mean_guess) for h_val in h_samples])
        x_samples = x_samples[~np.isnan(x_samples)]

        if len(x_samples) == 0:
            return np.nan, np.nan, np.nan, np.array([])

        m_new = np.mean(x_samples)
        q_new = np.mean(x_samples ** 2)

        diff = abs(m_new - m_th) + abs(q_new - q_th)
        if diff < tol:
            print(f"Theoretical calculation converged in {i + 1} iterations.")
            break

        m_th = (1 - damping) * m_th + damping * m_new
        q_th = (1 - damping) * q_th + damping * q_new
    else:
        print(f"Warning: Theoretical calculation did not converge.")

    if initial_x_mean_guess > 0:
        phi_th = np.sum(x_samples > 0) / len(x_samples)
    else:
        phi_th = np.sum(x_samples < 0) / len(x_samples)

    print(f"Theoretical Results: m={m_th:.4f}, q={q_th:.4f}, phi={phi_th:.4f}")
    return m_th, q_th, phi_th, x_samples


# ==========================================
# 4. 绘图与主程序 (已修改)
# ==========================================

def plot_results(all_final_x, sim_m, sim_q, sim_phi, m_hist, q_hist, dt,
                 theoretical_x_samples, th_m, th_q, th_phi, title_prefix=""):
    plt.figure(figsize=(16, 6))

    # --- 1. PDF 分布图 ---
    plt.subplot(1, 2, 1)
    plt.hist(all_final_x, bins=100, density=True, alpha=0.5, color='skyblue', edgecolor='white',
             label='Simulation (Hist)')

    if len(theoretical_x_samples) > 100:
        density = gaussian_kde(theoretical_x_samples)
        xs = np.linspace(np.min(theoretical_x_samples), np.max(theoretical_x_samples), 500)
        plt.plot(xs, density(xs), color='red', linewidth=2.5, label='Theory (Mean Field PDF)')
        plt.fill_between(xs, density(xs), color='red', alpha=0.1)

    plt.title(f"{title_prefix} Abundance PDF\nSim: m={sim_m:.2f}, q={sim_q:.2f} | Th: m={th_m:.2f}, q={th_q:.2f}")
    plt.xlabel("$x_i$")
    plt.ylabel("P(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- 2. 时间演化图 (已修改: 添加模拟均值辅助线) ---
    plt.subplot(1, 2, 2)
    time_axis = np.arange(len(m_hist)) * dt

    # A. 单次运行轨迹 (实线，降低透明度作为背景)
    plt.plot(time_axis, m_hist, label='m(t) - Single Run', color='blue', alpha=0.4, linewidth=1.5)
    plt.plot(time_axis, q_hist, label='q(t) - Single Run', color='red', alpha=0.4, linewidth=1.5)

    # B. 模拟的统计平均值 (新增: 虚线 Dashed)
    plt.axhline(y=sim_m, color='blue', linestyle='--', linewidth=2, label=f'm (Sim Avg): {sim_m:.2f}')
    plt.axhline(y=sim_q, color='red', linestyle='--', linewidth=2, label=f'q (Sim Avg): {sim_q:.2f}')

    # C. 理论预测值 (点线 Dotted)
    plt.axhline(y=th_m, color='blue', linestyle=':', linewidth=2.5, label=f'm (Theory): {th_m:.2f}')
    plt.axhline(y=th_q, color='red', linestyle=':', linewidth=2.5, label=f'q (Theory): {th_q:.2f}')

    plt.title(f"{title_prefix} Dynamics (Single Run vs Averages)")
    plt.xlabel("Time")

    # 将图例放在图外，以免遮挡曲线 (可选)
    plt.legend(loc='upper right', fontsize='small', framealpha=0.9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sim_params = {
        'S': 100,
        'dt': 0.01,
        't_steps': 3000,
        'mu_c': 0.0, 'sigma_c': 0.2,
        'mu_d': 0.3, 'sigma_d': 0.0,
        'mu_e': 0.1, 'sigma_e': 0.0
    }
    num_batches = 30

    # 正分支
    #print("\n--- POSITIVE BRANCH ---")
    '''initial_x_mean_pos = 1.0
    all_x_pos, sim_m_pos, sim_q_pos, sim_phi_pos, m_hist_pos, q_hist_pos = run_multiple_batches_parallel(
        num_batches, sim_params, initial_x_mean_pos, 0.1
    )
    th_m_pos, th_q_pos, th_phi_pos, th_x_pos = calculate_theoretical_properties(sim_params, initial_x_mean_pos)
    plot_results(all_x_pos, sim_m_pos, sim_q_pos, sim_phi_pos, m_hist_pos, q_hist_pos, sim_params['dt'],
                 th_x_pos, th_m_pos, th_q_pos, th_phi_pos, "Positive Branch")'''

    # 负分支
    print("\n--- NEGATIVE BRANCH ---")
    initial_x_mean_neg = -0.6
    all_x_neg, sim_m_neg, sim_q_neg, sim_phi_neg, m_hist_neg, q_hist_neg = run_multiple_batches_parallel(
        num_batches, sim_params, initial_x_mean_neg, 0.1
    )
    th_m_neg, th_q_neg, th_phi_neg, th_x_neg = calculate_theoretical_properties(sim_params, initial_x_mean_neg)
    plot_results(all_x_neg, sim_m_neg, sim_q_neg, sim_phi_neg, m_hist_neg, q_hist_neg, sim_params['dt'],
                 th_x_neg, th_m_neg, th_q_neg, th_phi_neg, "Negative Branch")