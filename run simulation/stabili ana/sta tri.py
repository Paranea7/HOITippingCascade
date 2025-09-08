import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d import Axes3D  # 导入 3D 绘图工具
from multiprocessing import Pool


# 定义函数 F (三系统版本)
# x: 三维状态变量 [x1, x2, x3]
# c: 三维常数项 [c1, c2, c3]
# d_matrix: 3x3 耦合矩阵，d_matrix[i, j] 表示 xj 对 xi 的影响
def F(x, c, d_matrix):
    x1, x2, x3 = x
    c1, c2, c3 = c  # 解包常数项

    # 计算每个方程
    # 方程形式: -xi^3 + xi + ci + sum(d_matrix[i,j] * xj for j != i)
    eq1 = -x1 ** 3 + x1 + c1 + d_matrix[0, 1] * x2 + d_matrix[0, 2] * x3
    eq2 = -x2 ** 3 + x2 + c2 + d_matrix[1, 0] * x1 + d_matrix[1, 2] * x3
    eq3 = -x3 ** 3 + x3 + c3 + d_matrix[2, 0] * x1 + d_matrix[2, 1] * x2

    return np.array([eq1, eq2, eq3])


# 计算雅可比矩阵 (三系统版本)
# x: 三维状态变量 [x1, x2, x3]
# d_matrix: 3x3 耦合矩阵
def jacobian(x, d_matrix):
    x1, x2, x3 = x

    # 雅可比矩阵 J_ij = d(Fi)/d(xj)
    # 对角线元素: d(-xi^3 + xi + ci)/d(xi) = -3*xi^2 + 1
    # 非对角线元素: d(Fi)/d(xj) = d_matrix[i, j]
    return np.array([
        [-3 * x1 ** 2 + 1, d_matrix[0, 1], d_matrix[0, 2]],
        [d_matrix[1, 0], -3 * x2 ** 2 + 1, d_matrix[1, 2]],
        [d_matrix[2, 0], d_matrix[2, 1], -3 * x3 ** 2 + 1]
    ])


# 判断稳定性 (逻辑不变，但参数为三维)
# x: 三维固定点 [x1, x2, x3]
# d_matrix: 3x3 耦合矩阵
def is_stable(x, d_matrix):
    J = jacobian(x, d_matrix)
    eigenvalues = np.linalg.eigvals(J)
    # 如果所有特征值的实部都小于0，则稳定
    return np.all(np.real(eigenvalues) < 0)


# 查找固定点 (三系统版本)
# params: (c1, c2, c3, initial_guesses, d_matrix)
def find_fixed_points(params):
    c1, c2, c3, initial_guesses, d_matrix = params
    c_tuple = (c1, c2, c3)  # 将c参数打包成元组或数组
    found_stable_fixed_points = set()

    for x_initial in initial_guesses:
        # 使用 root 函数查找 F(x) = 0 的解
        # args 传递给 F 函数的额外参数
        sol = root(F, x_initial, args=(c_tuple, d_matrix))
        if sol.success:
            # 将找到的固定点四舍五入到一定精度，以便进行集合操作（去重）
            fixed_point = tuple(np.round(sol.x, 6))
            if fixed_point not in found_stable_fixed_points:
                # 判断固定点是否稳定
                if is_stable(fixed_point, d_matrix):
                    found_stable_fixed_points.add(fixed_point)
    return len(found_stable_fixed_points)


if __name__ == "__main__":  # 主程序的入口
    # 参数设置
    # 为了在 3D 图中更清晰地展示，并平衡计算时间，适当减少点数
    c1_range = np.linspace(0, 0.8, 15)  # 15个点
    c2_range = np.linspace(0, 0.8, 15)  # 15个点
    c3_range = np.linspace(0, 0.8, 15)  # 15个点

    # 定义不同的耦合矩阵配置
    d_matrix_list = [
        # 配置 1: 无耦合 (对角矩阵)
        np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]]),

        # 配置 2: 简单循环耦合 x1->x2->x3->x1
        np.array([[0, 0, 0.2],
                  [0.2, 0, 0],
                  [0, 0.2, 0]]),

        # 配置 3: 复杂一些的耦合
        np.array([[0, 0.1, 0.3],
                  [0.2, 0, 0.1],
                  [0.3, 0.2, 0]]),
    ]

    # 存储稳定固定点数量的矩阵
    # 维度: (d_matrix 配置数量, c1 范围长度, c2 范围长度, c3 范围长度)
    nr_stable_points_matrix = np.zeros((len(d_matrix_list), len(c1_range), len(c2_range), len(c3_range)))

    # 初始猜测的选择 (三维)
    initial_guesses = [
        [-0.8, -0.8, -0.8],
        [0.8, 0.8, 0.8],
        [-0.8, 0.8, 0.8],
        [0.8, -0.8, 0.8],
        [0.8, 0.8, -0.8],
        [-0.8, -0.8, 0.8],
        [-0.8, 0.8, -0.8],
        [0.8, -0.8, -0.8],
        [0, 0, 0]  # 增加原点附近的猜测
    ]

    # 使用 multiprocessing.Pool 执行并行计算
    print("开始计算稳定固定点数量...")
    for k, d_matrix in enumerate(d_matrix_list):
        print(f"正在处理耦合矩阵配置 {k + 1}/{len(d_matrix_list)}")

        # 创建参数元组列表，遍历 c1, c2, c3 的所有组合
        params = []
        for c1_val in c1_range:
            for c2_val in c2_range:
                for c3_val in c3_range:
                    params.append((c1_val, c2_val, c3_val, initial_guesses, d_matrix))

        # 使用 multiprocessing 管理进程池
        with Pool() as pool:
            results = pool.map(find_fixed_points, params)

        # 填充 stable points matrix
        # results 是一个一维列表，需要 reshape 成 (len(c1_range), len(c2_range), len(c3_range))
        nr_stable_points_matrix[k, :, :, :] = np.array(results).reshape(
            len(c1_range), len(c2_range), len(c3_range)
        )

    print("计算完成，开始绘图...")

    # 定义颜色映射和边界
    # 0: 灰色, 1: 浅蓝, 2: 浅绿, 3: 黄色, 4: 金色, 5+: 更深的颜色
    # 边界需要覆盖可能的最大稳定点数量，这里假设最多有 8 个 (2^3)
    cmap = ListedColormap(['lightgray', 'lightblue', 'lightgreen', 'yellow', 'gold', 'orange', 'red', 'purple'])
    boundaries = np.arange(-0.5, 8.5, 1)  # 从0到8
    norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

    # 绘制每个 d_matrix 配置的 3D 稳定性图
    for k, d_matrix in enumerate(d_matrix_list):
        fig = plt.figure(figsize=(10, 8))
        # 添加 3D 坐标轴
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f'd_matrix Configuration {k + 1}')

        # 准备用于 scatter plot 的数据
        # 使用 meshgrid 创建 c1, c2, c3 的坐标网格
        # indexing='ij' 确保 C1 沿第一个维度变化，C2 沿第二个，C3 沿第三个
        C1, C2, C3 = np.meshgrid(c1_range, c2_range, c3_range, indexing='ij')

        # 获取对应稳定点数量，并展平为一维数组
        stable_counts = nr_stable_points_matrix[k].flatten()

        # 绘制三维散点图，颜色表示稳定点的数量
        scatter = ax.scatter(C1.flatten(), C2.flatten(), C3.flatten(),
                             c=stable_counts, cmap=cmap, norm=norm,
                             s=50, alpha=0.8)  # s控制点的大小，alpha控制透明度

        # 设置坐标轴标签
        ax.set_xlabel('c1')
        ax.set_ylabel('c2')
        ax.set_zlabel('c3')

        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_ticks(np.arange(0, 8))  # 设置色条刻度
        cbar.set_ticklabels([str(i) for i in np.arange(0, 8)])
        cbar.set_label('Number of Stable Fixed Points')

    plt.tight_layout()  # 尝试调整布局，但对于 3D 图可能效果不佳
    plt.show()