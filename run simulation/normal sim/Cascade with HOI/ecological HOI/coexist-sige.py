import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

def generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e):
    """生成模拟参数（含对称性约束和自耦合排除）"""
    # 生成控制参数，从正态分布中生成
    c_i = np.random.normal(mu_c, sigma_c, s)

    # 生成二体耦合参数
    d_ij = np.random.normal(mu_d / s, sigma_d / s, (s, s))
    np.fill_diagonal(d_ij, 0)  # 排除自耦合

    # 生成修正的二体耦合参数 d_ji
    d_ji = rho_d * d_ij + np.sqrt(1 - rho_d ** 2) * np.random.normal(mu_d / s, sigma_d / s, (s, s))
    np.fill_diagonal(d_ji, 0)  # 排除自耦合

    # 生成三体耦合参数（对称处理）
    e_ijk = np.random.normal(mu_e / s ** 2, sigma_e / s ** 2, (s, s, s))
    for i in range(s):
        # 对称化 j <-> k 维度
        e_ijk[i] = (e_ijk[i] + e_ijk[i].T) / 2
        # 排除自耦合情况
        np.fill_diagonal(e_ijk[i], 0)  # j=k 自耦合
        e_ijk[i, i, :] = 0  # i=j 自耦合
        e_ijk[i, :, i] = 0  # i=k 自耦合

    return c_i, d_ij, d_ji, e_ijk  # 返回生成的参数

def dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps):
    """动态模拟（向量化计算版本）"""
    x = x_init.copy()  # 复制初始状态
    dt = 0.01  # 时间步长

    for _ in range(t_steps):
        def compute_dx(x):
            """
            计算状态变化 dx 的内嵌函数
            """
            dx = -x ** 3 + x + c_i  # 基础动态方程
            dx += d_ji @ x  # 矩阵乘法计算二体耦合

            # 向量化计算三体耦合
            dx += np.einsum('ijk,j,k->i', e_ijk, x, x)  # 将三体耦合带入动态方程
            return dx

        # 集成步骤使用四阶龙格-库塔法
        k1 = compute_dx(x)
        k2 = compute_dx(x + 0.5 * dt * k1)
        k3 = compute_dx(x + 0.5 * dt * k2)
        k4 = compute_dx(x + dt * k3)
        x += (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6  # 更新状态

    return x  # 返回最终状态

def calculate_survival_rate(final_states):
    """存活率计算"""
    return np.mean(final_states > 0)  # 计算存活率，以向量化方式实现

def single_simulation(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps):
    """单次模拟包装函数"""
    c_i, _, d_ji, e_ijk = generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e)  # 生成参数
    x_init = np.full(s, -0.8)  # 固定初始状态
    final_states = dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps)  # 进行动态模拟
    return calculate_survival_rate(final_states)  # 返回存活率

def plot_survival_rate_vs_sigma_e(sigma_e_values, survival_rates_list, errors_list, mu_e_values):
    """结果可视化"""
    plt.figure(figsize=(10, 6))  # 设置图形大小
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 图例颜色

    for i, (rates, errs, mu_e) in enumerate(zip(survival_rates_list, errors_list, mu_e_values)):
        plt.errorbar(sigma_e_values, rates, yerr=errs, fmt='o--',
                     color=colors[i], capsize=5, label=f'μ_e = {mu_e}')  # 绘制带误差条的存活率

    plt.title('Survival Rate vs Three-body Coupling Strength', fontsize=14)  # 图表标题
    plt.xlabel('σ_e (Three-body Coupling Strength)', fontsize=12)  # x轴标签
    plt.ylabel('Survival Rate', fontsize=12)  # y轴标签
    plt.grid(alpha=0.3)  # 添加网格
    plt.legend()  # 显示图例
    plt.tight_layout()  # 自动调整布局
    plt.show()  # 显示图形

def main():
    """参数设置和主流程"""
    s = 5  # 系统节点数量
    t_steps = 1500  # 模拟的时间步长
    simulations_per_sigma = 100  # 每个参数点的模拟次数

    # 耦合参数设置
    mu_c = 0.0  # 控制参数均值
    sigma_c = 1.0  # 控制参数标准差
    mu_d = 0.0  # 二体耦合均值
    sigma_d = 0.0  # 二体耦合标准差
    rho_d = 1.0  # 二体耦合相关系数

    # 扫描不同的三体耦合强度参数
    sigma_e_values = np.linspace(0.0, 1.0, 11)  # 三体耦合强度扫描范围
    mu_e_values = [0.2, 0.5, 1.0]  # 不同三体耦合均值

    # 并行计算框架
    with Pool() as pool:  # 使用进程池来执行并行计算
        survival_rates_list = []  # 存放存活率
        errors_list = []  # 存放误差

        for mu_e in mu_e_values:  # 遍历不同的三体耦合均值
            survival_rates = []  # 存放每个 sigma_e 的存活率
            survival_all = []  # 存放存活情况

            for sigma_e in sigma_e_values:  # 遍历不同的三体耦合强度
                # 创建任务并行执行
                tasks = [(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps)
                         for _ in range(simulations_per_sigma)]

                results = pool.starmap(single_simulation, tasks)  # 并行执行模拟

                # 结果处理
                survival_all.append(results)  # 记录所有模拟结果
                survival_rates.append(np.mean(results))  # 计算平均存活率

            # 计算误差
            errors = [np.std(survivals) for survivals in survival_all]  # 计算每个组的标准差
            survival_rates_list.append(survival_rates)  # 存放平均存活率
            errors_list.append(errors)  # 存放误差

    # 结果可视化
    plot_survival_rate_vs_sigma_e(sigma_e_values, survival_rates_list, errors_list, mu_e_values)

if __name__ == "__main__":
    main()  # 调用主函数启动程序