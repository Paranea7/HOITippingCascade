import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, phi_0=0.0, fixed_c_value=None):
    """
    生成模拟所需的参数。
    c_i: 根据 phi_0（比例或绝对个数）和 fixed_c_value 生成。
         - 如果 0 <= phi_0 <= 1: 将 count = round(phi_0 * s)
         - 如果 phi_0 > 1: 将 count = min(s, round(phi_0)) （视为绝对个数）
         随机选择 count 个不重复索引，将这些 c_i 设为 fixed_c_value，其余为 0。
         如果 fixed_c_value 为 None，则使用 mu_c 作为非零值。
    d_ij: 正态生成，并把对角线置零。
    d_ji: 与 d_ij 相关的矩阵（使用 rho_d 相关结构，保持矩阵形式）。
    e_ijk: 三体耦合，满足 e_iii=0 且对称 e_ijk = e_ikj。
    """
    # 处理 fixed_c_value
    if fixed_c_value is None:
        fixed_c_value = mu_c

    # 计算非零个数 count
    if phi_0 is None:
        count = 0
    else:
        phi_val = float(phi_0)
        if phi_val <= 0.0:
            count = 0
        elif 0.0 < phi_val <= 1.0:
            count = int(round(phi_val * s))
        else:
            count = int(round(phi_val))
            if count > s:
                count = s

    # 生成 c_i：先全 0，然后在 count 个随机位置设为 fixed_c_value
    c_i = np.zeros(s, dtype=float)
    if count > 0:
        idx = np.random.choice(s, size=count, replace=False)
        c_i[idx] = float(fixed_c_value)

    # 生成 d_ij，缩放按原逻辑 mu_d/s, sigma_d/s
    d_ij = np.random.normal(mu_d / s, sigma_d / s, (s, s))
    np.fill_diagonal(d_ij, 0.0)

    # 生成与 d_ij 相关的 d_ji：构造一个独立噪声矩阵并与 d_ij 做相关混合（保持矩阵形状）
    noise = np.random.normal(mu_d / s, sigma_d / s, (s, s))
    d_ji = rho_d * d_ij + np.sqrt(max(0.0, 1.0 - rho_d ** 2)) * noise
    # 通常也把对角线设为0
    np.fill_diagonal(d_ji, 0.0)

    # 生成 e_ijk，按 mu_e/s^2, sigma_e/s^2 缩放
    e_ijk = np.random.normal(mu_e / s ** 2, sigma_e / s ** 2, (s, s, s))

    # 设置 e_iii = 0
    for i in range(s):
        e_ijk[i, i, i] = 0.0

    # 确保 e_ijk = e_ikj（对 j,k 对称）
    # 我们只需让上/下三角对应元素相等
    for i in range(s):
        for j in range(s):
            for k in range(j + 1, s):
                # 设定 e[i,j,k] 与 e[i,k,j] 为同一值（取两者平均以减少偏差）
                val = 0.5 * (e_ijk[i, j, k] + e_ijk[i, k, j])
                e_ijk[i, j, k] = val
                e_ijk[i, k, j] = val

    return c_i, d_ij, d_ji, e_ijk


def compute_dynamics(x, c_i, d_ji, e_ijk):
    """
    计算 dx/dt = -x^3 + x + c_i + d_ji @ x + einsum(e_ijk, x, x)
    使用向量化的 einsum 计算三体项。
    """
    dx = -x ** 3 + x + c_i
    dx = dx + np.dot(d_ji, x)

    # 计算三体贡献：对 i 索引求和 e_ijk * x_j * x_k
    # 直接使用 einsum 更清晰且高效
    # e_contribution_i = sum_{j,k} e_ijk[i,j,k] * x[j] * x[k]
    e_contribution = np.einsum('ijk,j,k->i', e_ijk, x, x)
    dx = dx + e_contribution

    return dx


def runge_kutta_step(x, c_i, d_ji, e_ijk, dt):
    k1 = compute_dynamics(x, c_i, d_ji, e_ijk)
    k2 = compute_dynamics(x + 0.5 * dt * k1, c_i, d_ji, e_ijk)
    k3 = compute_dynamics(x + 0.5 * dt * k2, c_i, d_ji, e_ijk)
    k4 = compute_dynamics(x + dt * k3, c_i, d_ji, e_ijk)
    dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return x + dx * dt


def dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps):
    """
    返回形状为 (t_steps, s) 的 x_history（包含每个时间步的状态）。
    """
    x = x_init.copy()
    dt = 0.01
    x_history = np.zeros((t_steps, s), dtype=float)

    for t in range(t_steps):
        x_history[t] = x
        x = runge_kutta_step(x, c_i, d_ji, e_ijk, dt)

    return x_history


def parallel_dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps, n_jobs=1, repeats=1):
    """
    并行运行 repeats 次独立的 dynamics_simulation（每次独立重采样参数的情况下 repeats 可 >1）。
    返回合并后的结果形状为 (repeats*t_steps, s)（按时间拼接所有重复）
    注意：如果你只是想在单次参数下并行化个体内部计算，这里并不合适 —— 当前实现是并行重复独立模拟。
    """
    # 当 n_jobs == 1 且 repeats == 1 时，直接调用以避免 joblib 开销
    if n_jobs == 1 or repeats == 1:
        results = []
        for _ in range(repeats):
            results.append(dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps))
    else:
        # 使用 joblib 并行化 repeats 个独立运行（每次使用相同参数 c_i, d_ji, e_ijk）
        results = Parallel(n_jobs=n_jobs)(
            delayed(dynamics_simulation)(s, c_i, d_ji, e_ijk, x_init, t_steps) for _ in range(repeats)
        )

    combined_results = np.concatenate(results, axis=0)  # 形状 (repeats*t_steps, s)
    return combined_results


def plot_final_state_distribution(final_states):
    plt.figure(figsize=(10, 6))
    plt.hist(final_states, bins=5, density=True, alpha=0.6, color='b', edgecolor='black')
    plt.title('Final State Distribution')
    plt.xlabel('System State (x)')
    plt.ylabel('Density')
    plt.xlim(-1.8,1.8)
    plt.grid()
    plt.show()


def plot_evolution(x_history, max_lines=50):
    """
    绘制演化曲线。为了可视化清晰，限制绘制的变量条数（默认最多绘制 50 条轨迹）。
    """
    t_steps, s = x_history.shape
    plt.figure(figsize=(10, 6))

    # 如果变量数过多，只绘制部分变量以免图像过于拥挤
    indices = np.arange(s)
    if s > max_lines:
        rng = np.random.default_rng()
        indices = rng.choice(s, size=max_lines, replace=False)

    for idx in indices:
        plt.plot(np.arange(t_steps), x_history[:, idx], linewidth=0.7, alpha=0.8)

    plt.title('Evolution of x over Time (sampled variables)')
    plt.xlabel('Time Steps')
    plt.ylabel('State x')
    plt.grid()
    plt.show()


def main():
    # 模型参数
    s = 100
    mu_c = 0.0
    sigma_c = 2 * np.sqrt(3) / 9
    mu_d = 0.3
    sigma_d = 0.3
    rho_d = 0.0
    mu_e = 0.5
    sigma_e = 0.4

    # phi_0 设置为比例 0.16（即 16% 的节点拥有非零 c_i）
    phi_0 = 0.1
    fixed_c_value = 2.0 * np.sqrt(3.0) / 9.0  # 非零 c 的值

    # 生成参数（注意：generate_parameters 中会随机选择 count 个索引赋值为 fixed_c_value）
    c_i, d_ij, d_ji, e_ijk = generate_parameters(
        s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, phi_0=phi_0, fixed_c_value=fixed_c_value
    )

    # 初始条件和步数
    x_init = np.random.normal(0.,0.01,s)  # 你原脚本中为 0.16
    t_steps = 1500

    # 并行参数：n_jobs 为 joblib 并行进程数，repeats 表示重复独立运行的次数（每次 dynamics_simulation 返回 t_steps 行数据）
    n_jobs = 16
    repeats = 1

    # 运行仿真
    x_history = parallel_dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps, n_jobs=n_jobs, repeats=repeats)

    # x_history 形状为 (repeats*t_steps, s)。如果 repeats==1，则形状为 (t_steps, s)
    # 绘制末态分布（取最后一个时间步）
    plot_final_state_distribution(x_history[-1, :])

    # 绘制演化曲线（为了可视化，只绘制部分变量）
    # 如果 repeats>1，则只取最后一个重复的时间段绘图
    if repeats == 1:
        plot_evolution(x_history)
    else:
        # 取最后一段 t_steps 长度的历史作为演化绘图
        plot_evolution(x_history[-t_steps:, :])


if __name__ == "__main__":
    main()