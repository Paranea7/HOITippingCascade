#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, rng):
    """
    生成参数（在 worker 中使用 RNG 以保证可复现）：
      - c_i: shape (s,)
      - d_ij: shape (s,s), N(mu_d/s, sigma_d/s)，对角线设0
      - d_ji: 与 d_ij 相关的矩阵，形状 (s,s)
      - e_ijk: shape (s,s,s), N(mu_e/s^2, sigma_e/s^2)，对每个 i 保证 j,k 对称且 e[i,i,i]=0
    使用 rng (np.random.RandomState) 来生成随机数。
    """
    c_i = rng.normal(mu_c, sigma_c, size=s)

    d_ij = rng.normal(loc=mu_d / s, scale=sigma_d / s, size=(s, s))
    np.fill_diagonal(d_ij, 0.0)

    noise = rng.normal(loc=mu_d / s, scale=sigma_d / s, size=(s, s))
    rho_d_clipped = max(-1.0, min(1.0, rho_d))
    d_ji = rho_d_clipped * d_ij + np.sqrt(max(0.0, 1.0 - rho_d_clipped ** 2)) * noise
    np.fill_diagonal(d_ji, 0.0)

    e_ijk = rng.normal(loc=mu_e / (s ** 2), scale=sigma_e / (s ** 2), size=(s, s, s))
    # 对称化与对角归零
    for i in range(s):
        e_ijk[i, i, i] = 0.0
        for j in range(s):
            for k in range(j + 1, s):
                val = 0.5 * (e_ijk[i, j, k] + e_ijk[i, k, j])
                e_ijk[i, j, k] = val
                e_ijk[i, k, j] = val

    return c_i, d_ij, d_ji, e_ijk


def compute_dynamics(x, c_i, d_ji, e_ijk):
    dx = -x ** 3 + x + c_i
    dx = dx + d_ji.dot(x)
    coupling_nonlinear = np.einsum('ijk,j,k->i', e_ijk, x, x)
    dx = dx + coupling_nonlinear
    return dx


def rk4_step(x, c_i, d_ji, e_ijk, dt):
    k1 = compute_dynamics(x, c_i, d_ji, e_ijk)
    k2 = compute_dynamics(x + 0.5 * dt * k1, c_i, d_ji, e_ijk)
    k3 = compute_dynamics(x + 0.5 * dt * k2, c_i, d_ji, e_ijk)
    k4 = compute_dynamics(x + dt * k3, c_i, d_ji, e_ijk)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def dynamics_simulation_worker(job_id, s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e,
                               x_init_value, t_steps, dt, seed):
    """
    Worker 函数：在子进程中运行一套参数与动力学，返回 final state 与 survival counts。
    - seed: int，用于创建 np.random.RandomState(seed) 保证可复现
    """
    rng = np.random.RandomState(seed)

    c_i, d_ij, d_ji, e_ijk = generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, rng)

    # 初始向量（可改为随机）
    x = np.full(s, x_init_value, dtype=float)

    survival_counts = []
    for _ in range(t_steps):
        x = rk4_step(x, c_i, d_ji, e_ijk, dt)
        survival_counts.append(np.sum(x > 0))

    # 返回需要的结果（尽量减小返回对象大小以降低 IPC 成本）
    result = {
        'job_id': job_id,
        'final_state': x,
        'survival_counts': np.array(survival_counts, dtype=int),
        # 如果需要，可返回 c_i, d_ij, d_ji, e_ijk（但会显著增加数据传输量）
    }
    return result


def run_parallel_simulations(n_systems=20, s=100, mu_c=0.0, sigma_c=0.4,
                             mu_d=0.2, sigma_d=0.3, rho_d=1.0,
                             mu_e=0.2, sigma_e=0.4,
                             x_init_value=-0.6, t_steps=1500, dt=0.01,
                             max_workers=None, seed_base=None):
    """
    并行运行 n_systems 个独立实例，返回汇总结果：
      - final_states_array: shape (n_systems, s)
      - survival_matrix: shape (n_systems, t_steps)
    """
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 1) - 1)

    if seed_base is None:
        base = np.random.randint(0, 2 ** 31 - 1)
    else:
        base = int(seed_base)
    seeds = [int(base + i + 1) for i in range(n_systems)]

    futures = []
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i in range(n_systems):
            seed_i = seeds[i]
            futures.append(executor.submit(dynamics_simulation_worker,
                                           i, s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e,
                                           x_init_value, t_steps, dt, seed_i))
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)

    # 按 job_id 排序，组装数组
    results.sort(key=lambda r: r['job_id'])
    final_states_array = np.vstack([r['final_state'] for r in results])
    survival_matrix = np.vstack([r['survival_counts'] for r in results])

    return final_states_array, survival_matrix


def plot_final_state_distribution(final_states_array):
    # 将所有系统的 final_states 展平后绘图
    flat = final_states_array.ravel()
    plt.figure(figsize=(10, 6))
    plt.hist(flat, bins=200, density=True, alpha=0.6, color='b', edgecolor='black')
    plt.title('Final State Distribution (all systems)')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.xlim(-1.6, 1.6)
    plt.grid()
    plt.show()


def plot_average_survival(survival_matrix):
    # survival_matrix shape: (n_systems, t_steps)
    mean_survival = np.mean(survival_matrix, axis=0)
    std_survival = np.std(survival_matrix, axis=0)
    t = np.arange(mean_survival.size)
    plt.figure(figsize=(10, 6))
    plt.plot(t, mean_survival, color='blue', label='mean survival (x>0)')
    plt.fill_between(t, mean_survival - std_survival, mean_survival + std_survival, color='blue', alpha=0.3,
                     label='±1 std')
    plt.xlabel('time step')
    plt.ylabel('count of x>0')
    plt.title('Average survival over systems')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    # 参数（你可在此处调整）
    n_systems = 20        # 并行独立样本数（可根据内存/CPU 增减）
    s = 300
    mu_c = 0.0
    sigma_c = 0.4
    mu_d = 0.2
    sigma_d = 0.3
    rho_d = 1.0
    mu_e = 0.2
    sigma_e = 0.1
    x_init_value = -0.6
    t_steps = 1500
    dt = 0.01
    max_workers = None     # 默认为 CPU_count()-1
    seed_base = 123456

    final_states_array, survival_matrix = run_parallel_simulations(
        n_systems=n_systems, s=s,
        mu_c=mu_c, sigma_c=sigma_c,
        mu_d=mu_d, sigma_d=sigma_d, rho_d=rho_d,
        mu_e=mu_e, sigma_e=sigma_e,
        x_init_value=x_init_value, t_steps=t_steps, dt=dt,
        max_workers=max_workers, seed_base=seed_base
    )

    # 绘图
    plot_final_state_distribution(final_states_array)
    plot_average_survival(survival_matrix)

    # 简单汇总输出
    mean_mags = np.mean(final_states_array, axis=1)
    print("Per-system mean state (first 10):", mean_mags[:10])
    print("Overall mean:", mean_mags.mean(), "std:", mean_mags.std())


if __name__ == "__main__":
    main()