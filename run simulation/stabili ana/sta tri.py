import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool


# 定义函数 F (含二阶相互作用项)
# x: 三维状态变量 [x1, x2, x3]
# c: 三维常数项 [c1, c2, c3]
# d_matrix: 3x3 线性耦合矩阵
# e_tensor: 3x3x3 高阶耦合张量，e_tensor[i,j,k] 表示项 e_ijk * x_j * x_k
def F(x, c, d_matrix, e_tensor):
    # 确保 x 是 numpy 数组
    x = np.asarray(x, dtype=float).reshape(3)
    c = np.asarray(c, dtype=float).reshape(3)

    # 自项与线性耦合（保留与原方程一致的结构）
    eq = np.zeros(3, dtype=float)
    eq[0] = -x[0] ** 3 + x[0] + c[0] + d_matrix[0, 1] * x[1] + d_matrix[0, 2] * x[2]
    eq[1] = -x[1] ** 3 + x[1] + c[1] + d_matrix[1, 0] * x[0] + d_matrix[1, 2] * x[2]
    eq[2] = -x[2] ** 3 + x[2] + c[2] + d_matrix[2, 0] * x[0] + d_matrix[2, 1] * x[1]

    # 添加二阶相互作用：对于每个 i，累加所有 j != k 且 j != i, k != i 的 e_tensor[i,j,k] * x_j * x_k
    # 这里直接遍历 j,k 并根据索引筛选（3 维系统，复杂度可接受）
    for i in range(3):
        sum_high = 0.0
        for j in range(3):
            for k in range(3):
                if i != j and i != k and j != k:
                    sum_high += e_tensor[i, j, k] * x[j] * x[k]
        eq[i] += sum_high

    return eq


# 计算雅可比矩阵（含高阶项对非对角元素的贡献）
# 说明：
# - 自项对角导数: d(-xi^3 + xi)/dxi = -3*xi^2 + 1
# - 线性耦合贡献已经在 d_matrix 中（常数）
# - 对于高阶项 e[i,j,k] * x_j * x_k（假设不包含 xi）：
#     dFi/dxj 包含 sum_k e[i,j,k] * x_k  (k != i, k != j)
#     dFi/dxk 包含 sum_j e[i,j,k] * x_j  (j != i, j != k)
# 因为我们在 F 中对所有 j,k（j!=k, j!=i, k!=i）都加了 e[i,j,k] * x_j * x_k，
# 所以在 J 中需要为每对 (i,j) 加上 sum_{k != i, k != j} e[i,j,k] * x[k]
def jacobian(x, d_matrix, e_tensor):
    x = np.asarray(x, dtype=float).reshape(3)
    J = np.zeros((3, 3), dtype=float)

    # 对角线（自项）
    J[0, 0] = -3 * x[0] ** 2 + 1
    J[1, 1] = -3 * x[1] ** 2 + 1
    J[2, 2] = -3 * x[2] ** 2 + 1

    # 线性耦合的非对角项
    J[0, 1] = d_matrix[0, 1]
    J[0, 2] = d_matrix[0, 2]
    J[1, 0] = d_matrix[1, 0]
    J[1, 2] = d_matrix[1, 2]
    J[2, 0] = d_matrix[2, 0]
    J[2, 1] = d_matrix[2, 1]

    # 高阶项对非对角元素的贡献：J[i,j] += sum_{k != i, k != j} e[i,j,k] * x[k]
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            contrib = 0.0
            for k in range(3):
                if k != i and k != j:
                    contrib += e_tensor[i, j, k] * x[k]
            J[i, j] += contrib

    return J


# 判断稳定性（所有雅可比特征值实部 < 0 即稳定）
def is_stable(x, d_matrix, e_tensor):
    J = jacobian(x, d_matrix, e_tensor)
    eigenvalues = np.linalg.eigvals(J)
    return np.all(np.real(eigenvalues) < 0)


# 查找固定点（带 e_tensor）
# params: (c1, c2, c3, initial_guesses, d_matrix, e_tensor)
def find_fixed_points(params):
    c1, c2, c3, initial_guesses, d_matrix, e_tensor = params
    c_tuple = (c1, c2, c3)
    found_stable_fixed_points = set()

    for x_initial in initial_guesses:
        # 使用 root 求解 F(x) = 0
        # 这里没有显式传入 jacobian 给 root；若想加速可传入 jac=jac_fun
        sol = root(F, x_initial, args=(c_tuple, d_matrix, e_tensor))
        if sol.success:
            # 四舍五入以便去重
            fixed_point = tuple(np.round(sol.x, 6))
            if fixed_point not in found_stable_fixed_points:
                if is_stable(sol.x, d_matrix, e_tensor):
                    found_stable_fixed_points.add(fixed_point)
    return len(found_stable_fixed_points)


if __name__ == "__main__":
    # 参数设置（为便于演示与调试，这里将分辨率调低；如需原始 161^3 网格请考虑显著增加运行时间）
    c1_range = np.linspace(0, 0.8, 81)  # 原来你写了 161，这里示例缩小到 17 点（可改回 161）
    c2_range = np.linspace(0, 0.8, 81)
    c3_range = np.linspace(0, 0.8, 81)

    # 定义不同的耦合矩阵配置（保留你原先的三种配置）
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

    # 构造 e_tensor：e[i,j,k] = 0.1 当且仅当 i != j 且 i != k 且 j != k，否则 0
    e_value = 0.1
    e_tensor = np.zeros((3, 3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i != j and i != k and j != k:
                    e_tensor[i, j, k] = e_value

    # 存储稳定固定点数量: 维度 (num_d_matrix, len(c1), len(c2), len(c3))
    nr_stable_points_matrix = np.zeros((len(d_matrix_list), len(c1_range), len(c2_range), len(c3_range)),
                                      dtype=int)

    # 初始猜测集合（保留你原始的 9 个初猜）
    initial_guesses = [
        [-0.8, -0.8, -0.8],
        [0.8, 0.8, 0.8],
        [-0.8, 0.8, 0.8],
        [0.8, -0.8, 0.8],
        [0.8, 0.8, -0.8],
        [-0.8, -0.8, 0.8],
        [-0.8, 0.8, -0.8],
        [0.8, -0.8, -0.8],
        [0, 0, 0]
    ]

    # 并行计算所有 c1,c2,c3 组合
    print("开始计算稳定固定点数量（含二阶相互作用 e_ijk = 0.1）...")
    for k, d_matrix in enumerate(d_matrix_list):
        print(f"正在处理耦合矩阵配置 {k + 1}/{len(d_matrix_list)}")

        params = []
        for i1, c1_val in enumerate(c1_range):
            for i2, c2_val in enumerate(c2_range):
                for i3, c3_val in enumerate(c3_range):
                    params.append((c1_val, c2_val, c3_val, initial_guesses, d_matrix, e_tensor))

        # 使用 multiprocessing Pool 并行处理
        # 注意：若问题规模大（很多参数组合），请根据机器内核数调整 pool 大小或分批次处理
        with Pool() as pool:
            results = pool.map(find_fixed_points, params)

        # 将一维结果 reshape 并保存
        nr_stable_points_matrix[k, :, :, :] = np.array(results).reshape(
            len(c1_range), len(c2_range), len(c3_range)
        )

    print("计算完成，开始绘图...")

    # 定义颜色映射和边界（你可根据实际稳定点数量调整颜色表）
    cmap = ListedColormap(['lightgray', 'lightblue', 'lightgreen', 'yellow', 'gold', 'orange', 'red', 'purple'])
    boundaries = np.arange(-0.5, 8.5, 1)  # 假设最多有 8 个稳定点（保守估计）
    norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

    # 绘制每个 d_matrix 配置的 3D 稳定性图
    for k, d_matrix in enumerate(d_matrix_list):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f'd_matrix Configuration {k + 1} (with 2nd-order interactions)')

        # 使用 meshgrid 创建 c1,c2,c3 网格 (indexing='ij' 保持索引一致)
        C1, C2, C3 = np.meshgrid(c1_range, c2_range, c3_range, indexing='ij')

        stable_counts = nr_stable_points_matrix[k].flatten()

        scatter = ax.scatter(C1.flatten(), C2.flatten(), C3.flatten(),
                             c=stable_counts, cmap=cmap, norm=norm,
                             s=50, alpha=0.8)

        ax.set_xlabel('c1')
        ax.set_ylabel('c2')
        ax.set_zlabel('c3')

        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        # colorbar 刻度与标签，可根据实际稳定点范围调整
        cbar.set_ticks(np.arange(0, min(8, stable_counts.max() + 1)))
        cbar.set_ticklabels([str(i) for i in np.arange(0, min(8, stable_counts.max() + 1))])
        cbar.set_label('Number of Stable Fixed Points')

    plt.tight_layout()
    plt.show()