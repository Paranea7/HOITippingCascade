import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e):
    """
    生成模拟所需的参数。
    """
    c_i = np.random.normal(mu_c, sigma_c, s)  # 生成控制参数

    # 生成二体耦合强度，并将d_ii设置为0
    d_ij = np.random.normal(mu_d / s, sigma_d / s, (s, s))  # 二体耦合强度
    np.fill_diagonal(d_ij, 0)  # 将对角线元素设置为0

    d_ji = rho_d * d_ij + np.sqrt(1 - rho_d ** 2) * np.random.normal(mu_d / s, sigma_d / s, s)  # 相关耦合强度

    # 生成三体耦合强度，同时满足 e_iii = 0 和 e_ijk = e_ikj
    e_ijk = np.random.normal(mu_e / s ** 2, sigma_e / s ** 2, (s, s, s))  # 生成初始三体耦合强度

    # 设置对角线元素为0，满足 e_iii = 0
    for i in range(s):
        e_ijk[i, i, i] = 0

    # 确保 e_ijk = e_ikj
    for i in range(s):
        for j in range(s):
            for k in range(s):
                if j < k:  # 仅计算上三角部分
                    e_ijk[i, j, k] = e_ijk[i, k, j]  # 赋值以确保对称性
                e_ijk[i, k, j] = e_ijk[i, j, k]  # 赋值以确保对称性

    return c_i, d_ij, d_ji, e_ijk

def compute_dynamics(x, c_i, d_ji, e_ijk):
    dx = -x ** 3 + x + c_i
    dx += np.dot(d_ji, x)

    x_matrix = x[:, None, None] * x[None, :, None] * x[None, None, :]
    e_contribution = np.einsum('ijk->i', e_ijk * x_matrix)
    dx += e_contribution

    return dx


def runge_kutta_step(x, c_i, d_ji, e_ijk, dt):
    k1 = compute_dynamics(x, c_i, d_ji, e_ijk)
    k2 = compute_dynamics(x + 0.5 * dt * k1, c_i, d_ji, e_ijk)
    k3 = compute_dynamics(x + 0.5 * dt * k2, c_i, d_ji, e_ijk)
    k4 = compute_dynamics(x + dt * k3, c_i, d_ji, e_ijk)
    dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x + dx * dt


def dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps):
    x = x_init.copy()
    dt = 0.01
    x_history = []  # Store all time-step states

    for _ in range(t_steps):
        x_history.append(x.copy())  # Record current state
        x = runge_kutta_step(x, c_i, d_ji, e_ijk, dt)

    return np.array(x_history)  # Return state history


def parallel_dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps, n_jobs=1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(dynamics_simulation)(s, c_i, d_ji, e_ijk, x_init, t_steps) for _ in range(n_jobs)
    )
    combined_results = np.concatenate(results, axis=0)
    return combined_results


def plot_final_state_distribution(final_states):
    plt.figure(figsize=(10, 6))
    plt.hist(final_states, bins=100, density=True, alpha=0.6, color='b', edgecolor='black')
    plt.title('Final State Distribution')
    plt.xlabel('System State (x)')
    plt.ylabel('Density')
    plt.xlim(-1.6, 1.6)
    plt.grid()
    plt.show()


def plot_evolution(x_history):
    plt.figure(figsize=(10, 6))
    plt.plot(x_history)  # Plot the evolution of x
    plt.title('Evolution of x over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('State x')
    plt.grid()
    plt.show()


def main():
    s = 100
    mu_c = 0.0
    sigma_c = 1.0
    mu_d = 0.0
    sigma_d = 1.0
    rho_d = 1.0
    mu_e = 0.0
    sigma_e = 0.0

    c_i, d_ij, d_ji, e_ijk = generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e)
    x_init = np.full(s, -0.6)
    t_steps = 1500

    # 使用并行处理进行模拟
    x_history = parallel_dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps, n_jobs=1)

    plot_final_state_distribution(x_history[499, :])  # Plot final state distribution
    plot_evolution(x_history)  # Plot evolution of x


if __name__ == "__main__":
    main()