import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from multiprocessing import Pool


# 定义函数 F
def F(x, c1, c2, d12, d21):
    x1, x2 = x
    return np.array([
        -x1 ** 3 + x1 + c1 + d21 * x2,
        -x2 ** 3 + x2 + c2 + d12 * x1
    ])


# 计算雅可比矩阵
def jacobian(x1, x2, d12, d21):
    return np.array([
        [-3 * x1 ** 2 + 1, d21],
        [d12, -3 * x2 ** 2 + 1]
    ])


# 判断稳定性
def is_stable(x1, x2, d12, d21):
    J = jacobian(x1, x2, d12, d21)
    eigenvalues = np.linalg.eigvals(J)
    return np.all(np.real(eigenvalues) < 0)


def find_fixed_points(params):
    c1, c2, initial_guesses, d12, d21 = params
    found_stable_fixed_points = set()
    for x_initial in initial_guesses:
        sol = root(F, x_initial, args=(c1, c2, d12, d21))
        if sol.success:
            fixed_point = tuple(np.round(sol.x, 6))
            if fixed_point not in found_stable_fixed_points:
                if is_stable(fixed_point[0], fixed_point[1], d12, d21):
                    found_stable_fixed_points.add(fixed_point)
    return len(found_stable_fixed_points)


if __name__ == "__main__":  # 主程序的入口
    # 参数设置
    c1_range = np.linspace(0, 0.8, 500)
    c2_range = np.linspace(0, 0.8, 500)
    d12_list = [0.0, 0.2, 0.3, 0.5, 0.7, 0.9]
    d21_list = [-0.0, -0.2, -0.3, -0.5, -0.7, -0.9]
    nr_stable_points_matrix = np.zeros((len(d12_list), len(d21_list), len(c1_range), len(c2_range)))

    # 初始猜测的选择
    initial_guesses = [
        [-0.8, -0.8],
        [0.8, 0.8],
        [-0.8, 0.8],
        [0.8, -0.8],
    ]

    # 使用 multiprocessing.Pool 执行并行计算
    for i, d12 in enumerate(d12_list):
        for j, d21 in enumerate(d21_list):
            # 创建参数元组
            params = [(c1, c2, initial_guesses, d12, d21) for c2 in c2_range for c1 in c1_range]

            # 使用 multiprocessing 管理进程池
            with Pool() as pool:
                results = pool.map(find_fixed_points, params)

            # 填充 stable points matrix
            nr_stable_points_matrix[i, j, :, :] = np.array(results).reshape(len(c1_range), len(c2_range))

    # 绘制热力图
    plt.figure(figsize=(14, 8))

    # 绘制每个 d12 和 d21 的热力图
    for i, d12 in enumerate(d12_list):
        for j, d21 in enumerate(d21_list):
            plt.subplot(len(d12_list), len(d21_list), i * len(d21_list) + j + 1)
            plt.title(f'd12={d12}, d21={d21}')
            cmap = ListedColormap(['lightgray', 'lightblue', 'lightgreen', 'yellow', 'gold'])
            boundaries = np.arange(-0.5, 5.5, 1)
            norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

            plt.imshow(nr_stable_points_matrix[i, j], extent=[c1_range[0], c1_range[-1], c2_range[0], c2_range[-1]],
                       origin='lower', aspect='auto', cmap=cmap, norm=norm)

            # 添加色条
            cbar = plt.colorbar()
            cbar.set_ticks([0, 1, 2, 3, 4])
            cbar.set_ticklabels(['0', '1', '2', '3', '4'])
            plt.xlabel('c1')
            plt.ylabel('c2')

    plt.tight_layout()
    plt.show()