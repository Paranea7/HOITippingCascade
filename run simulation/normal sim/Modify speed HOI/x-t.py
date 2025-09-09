import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


# 生成参数函数（包含HOI修饰参数）
def generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_beta, sigma_beta, omega):
    """生成系统参数，包含：
    - c_i: 控制参数
    - d_ji: 二阶相互作用矩阵（s×s）
    - beta_ijk: 三阶修饰强度张量（s×s×s）
    - omega: 全局修饰速度标量
    """
    # 生成控制参数
    c_i = np.random.normal(mu_c, sigma_c, s)

    # 生成二阶相互作用矩阵
    d_base = np.random.normal(mu_d / s, sigma_d / s, (s, s))
    np.fill_diagonal(d_base, 0)  # 消除自相互作用
    noise = np.random.normal(mu_d / s, sigma_d / s, (s, s))
    d_ji = rho_d * d_base + np.sqrt(1 - rho_d ** 2) * noise

    # 生成三阶修饰强度张量
    beta_ijk = np.random.normal(mu_beta / s ** 2, sigma_beta / s ** 2, (s, s, s))
    # 消除自修饰项（i修饰i对j的作用）
    for i in range(s):
        beta_ijk[i, i, :] = 0
        beta_ijk[i, :, i] = 0

    return c_i, d_ji, beta_ijk, omega


# 动力学方程计算（同时更新x和m）
def compute_dynamics(x, m, c_i, d_ji, beta_ijk, omega):
    """计算状态变量x和修饰变量m的导数"""
    # 计算x的导数
    dx = -x ** 3 + x + c_i
    dx += np.einsum('ij,ij->i', d_ji, m) * x  # Σ_j (d_ji * m_ij) * x_j

    # 计算m的导数
    x_k = x.reshape(1, 1, -1)  # 转换为(1,1,s)张量
    beta_term = np.einsum('ijk,kl->ijl', beta_ijk, x_k)  # Σ_k beta_ijk x_k
    dm = omega * (1 - m + beta_term.squeeze())  # 移除冗余维度

    return dx, dm


# 四阶龙格-库塔积分步长
def runge_kutta_step(x, m, c_i, d_ji, beta_ijk, omega, dt):
    k1x, k1m = compute_dynamics(x, m, c_i, d_ji, beta_ijk, omega)
    k2x, k2m = compute_dynamics(x + 0.5 * dt * k1x, m + 0.5 * dt * k1m,
                                c_i, d_ji, beta_ijk, omega)
    k3x, k3m = compute_dynamics(x + 0.5 * dt * k2x, m + 0.5 * dt * k2m,
                                c_i, d_ji, beta_ijk, omega)
    k4x, k4m = compute_dynamics(x + dt * k3x, m + dt * k3m,
                                c_i, d_ji, beta_ijk, omega)

    x_new = x + (k1x + 2 * k2x + 2 * k3x + k4x) * dt / 6
    m_new = m + (k1m + 2 * k2m + 2 * k3m + k4m) * dt / 6

    return x_new, m_new


# 完整动力学模拟
def dynamics_simulation(s, c_i, d_ji, beta_ijk, omega, x_init, m_init, t_steps, dt=0.01):
    x = x_init.copy()
    m = m_init.copy()
    x_history = np.zeros((t_steps, s))
    m_history = np.zeros((t_steps, s, s))

    for t in range(t_steps):
        x_history[t] = x
        m_history[t] = m
        x, m = runge_kutta_step(x, m, c_i, d_ji, beta_ijk, omega, dt)

    return x_history, m_history


# 可视化函数
def plot_results(x_history, sample_species=3):
    """可视化前n个物种的演化"""
    plt.figure(figsize=(12, 6))

    # 绘制物种演化
    plt.subplot(1, 2, 1)
    for i in range(sample_species):
        plt.plot(x_history[:, i], label=f'Species {i + 1}')
    plt.title('Species Dynamics')
    plt.xlabel('Time Steps')
    plt.ylabel('Abundance')
    plt.legend()

    # 绘制修饰变量演化
    plt.subplot(1, 2, 2)
    m_avg = np.mean(x_history, axis=1)
    plt.plot(m_avg, color='purple')
    plt.title('Average Modification State')
    plt.xlabel('Time Steps')
    plt.ylabel('Mean m_ij')

    plt.tight_layout()
    plt.show()


# 主程序
def main():
    # 参数设置
    s = 5  # 物种数
    mu_c, sigma_c = 0.0, 0.5  # 控制参数分布
    mu_d, sigma_d = 0.0, 1.0  # 二阶相互作用
    rho_d = 0.7  # 相互作用对称性
    mu_beta, sigma_beta = -2.0, 1.0  # 修饰强度参数
    omega = 0.5  # 修饰速度

    # 生成参数
    c_i, d_ji, beta_ijk, omega = generate_parameters(
        s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_beta, sigma_beta, omega
    )

    # 初始化
    x_init = np.random.normal(-0.5, 0.1, s)  # 初始物种丰度
    m_init = np.ones((s, s))  # 初始无修饰

    # 运行模拟
    t_steps = 2000
    x_hist, m_hist = dynamics_simulation(s, c_i, d_ji, beta_ijk, omega,
                                         x_init, m_init, t_steps)

    # 结果可视化
    plot_results(x_hist)

    # 稳态分布分析
    final_states = x_hist[-500:]  # 取最后500步作为稳态
    plt.figure(figsize=(8, 6))
    plt.hist(final_states.flatten(), bins=50, density=True,
             alpha=0.7, color='teal', edgecolor='black')
    plt.title('Steady State Distribution')
    plt.xlabel('Species Abundance')
    plt.ylabel('Probability Density')
    plt.show()


if __name__ == "__main__":
    main()