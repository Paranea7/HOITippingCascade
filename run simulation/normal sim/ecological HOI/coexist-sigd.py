import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

def generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e):
    """
    生成模型参数：
    - c_i: 控制参数的随机值
    - d_ij: 二体耦合矩阵
    - d_ji: 修正后的耦合矩阵
    - e_ijk: 三体耦合张量
    """
    c_i = np.random.normal(mu_c, sigma_c, s)  # 从正态分布生成控制参数
    d_ij = np.random.normal(mu_d/s, sigma_d/s, (s, s))  # 二体耦合矩阵
    d_ji = rho_d * d_ij + np.sqrt(1 - rho_d**2) * np.random.normal(mu_d/s, sigma_d/s, (s, s))  # 耦合修正
    e_ijk = np.random.normal(mu_e/s**2, sigma_e/s**2, (s, s, s))  # 三体耦合张量
    return c_i, d_ij, d_ji, e_ijk

def dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps):
    """
    模拟动态行为：
    - x: 系统状态
    - dt: 时间步长
    - 循环 t_steps 次更新系统状态
    """
    x = x_init.copy()  # 复制初始状态
    dt = 0.01  # 时间步长

    for _ in range(t_steps):
        def compute_dx(x):
            """
            计算状态变化 dx 的内嵌函数
            """
            dx = -x**3 + x + c_i  # 计算基础动态
            dx += np.dot(d_ji, x)  # 二体耦合的影响
            dx += np.einsum('ijk,j,k->i', e_ijk, x, x)  # 三体耦合影响
            return dx

        # 四阶龙格-库塔法用于更新状态
        k1 = compute_dx(x)
        k2 = compute_dx(x + 0.5 * dt * k1)
        k3 = compute_dx(x + 0.5 * dt * k2)
        k4 = compute_dx(x + dt * k3)
        x += (k1 + 2*k2 + 2*k3 + k4) * dt / 6  # 更新状态

    return x  # 返回最终状态

def calculate_survival_rate(final_states):
    """
    计算并返回存活率：
    - 存活的个体数量与总个体数量的比例
    """
    survival = np.sum(final_states > 0)  # 存活的个体数量
    total = len(final_states)  # 总个体数
    survival_rate = survival / total  # 计算存活率
    return survival_rate

def single_simulation(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps):
    """
    执行单次模拟，返回存活率以用于并行处理。
    """
    c_i, _, d_ji, e_ijk = generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e)
    x_init = np.full(s, -0.8)  # 用固定值初始化状态
    final_states = dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps)  # 运行动态模拟
    return calculate_survival_rate(final_states)  # 返回存活率

def plot_survival_rate(sigma_d_values, survival_rates_list, errors_list, mu_d_values):
    """
    绘制存活率与 sigma_d 的关系图
    """
    plt.figure(figsize=(10, 6))
    for i, mu_d in enumerate(mu_d_values):
        plt.errorbar(sigma_d_values, survival_rates_list[i], yerr=errors_list[i], fmt='o-', capsize=5,
                     label=f'Mu_d = {mu_d}')  # 带误差条的绘制

    plt.title('Survival Rate vs Sigma_d with SE Error Bars')  # 图表标题
    plt.xlabel('Sigma_d (Standard Deviation of Two-body Coupling)')  # x轴标签
    plt.ylabel('Survival Rate')  # y轴标签
    plt.legend()  # 显示图例
    plt.grid()  # 显示网格
    plt.show()  # 显示图形

def main():
    """
    主函数，设置参数并运行多个动态模拟以获得存活率。
    """
    s = 50  # 系统个体数
    mu_c = 0  # 控制参数均值
    sigma_c = 2*np.sqrt(3)/9  # 控制参数标准差
    rho_d = 1.0  # 二体耦合相关系数
    mu_e = 0.0  # 三体耦合均值
    sigma_e = 1.0  # 三体耦合标准差
    t_steps = 10000  # 设置时间步数
    simulations_per_sigma = 50  # 每个 sigma_d 下的模拟次数

    sigma_d_values = np.linspace(0.0, 1.0, 21)  # sigma_d 的取值范围
    mu_d_values = [0.2, 0.3, 0.5]  # 不同的 mu_d 值
    survival_rates_list = []  # 存放存活率列表
    survival_all_list = []  # 存放所有存活结果以计算标准误

    with Pool() as pool:  # 使用池来进行并行计算
        for mu_d in mu_d_values:
            survival_rates = []  # 每个 mu_d 的存活率
            survival_all = []  # 存放每个 sigma_d 下的存活情况
            for sigma_d in sigma_d_values:
                survival_results = pool.starmap(single_simulation,
                                                [(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps) for _
                                                 in range(simulations_per_sigma)])  # 并行执行单次模拟

                survival_all.append(survival_results)  # 存放对应的存活率结果
                average_survival_rate = np.mean(survival_results)  # 计算平均存活率
                survival_rates.append(average_survival_rate)  # 存放平均值

            survival_rates_list.append(survival_rates)  # 收集所有存活率结果

            # 计算标准误（SE）：使用无偏样本标准差（ddof=1）除以 sqrt(N)
            N = simulations_per_sigma
            errors_se = [np.std(survivals, ddof=1) / np.sqrt(N) for survivals in survival_all]
            survival_all_list.append(errors_se)  # 存放标准误

    # 绘制存活率与 sigma_d 的关系图（误差条为标准误 SE）
    plot_survival_rate(sigma_d_values, survival_rates_list, survival_all_list, mu_d_values)

if __name__ == "__main__":
    main()  # 运行主函数