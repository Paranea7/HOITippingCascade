import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from joblib import Parallel, delayed



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

def find_fixed_points(c1, c2, initial_guesses, d12, d21):
    found_stable_fixed_points = set()  # 使用集合确保不重复计数
    for x_initial in initial_guesses:
        sol = root(F, x_initial, args=(c1, c2, d12, d21))

        if sol.success:  # 确保成功找到根
            fixed_point = tuple(np.round(sol.x, 6))  # 用元组存储并四舍五入以避免浮点数比较问题
            if fixed_point not in found_stable_fixed_points:  # 检查是否已经计数
                if is_stable(fixed_point[0], fixed_point[1], d12, d21):
                    found_stable_fixed_points.add(fixed_point)  # 添加到找到的稳定不动点集合

    return len(found_stable_fixed_points)  # 返回找到的稳定不动点数量


# 参数设置
c1_range = np.linspace(0, 0.8, 200)
c2_range = np.linspace(0, 0.8, 200)
d12 = 0.5
d21 = -0.2
nr_stable_points = np.zeros((len(c1_range), len(c2_range)))

# 初始猜测的选择
initial_guesses = [
    [-0.8, -0.8],
    [0.8, 0.8],
    [-0.8, 0.8],
    [0.8, -0.8],
]



# 使用 Parallel 和 delayed 执行并行计算
for i, c1 in enumerate(c1_range):
    # 使用 joblib 的 Parallel 和 delayed 来并行执行, 使用8个核
    nr_stable_points[i, :] = Parallel(n_jobs=-1)(
        delayed(find_fixed_points)(c1, c2, initial_guesses, d12, d21) for c2 in c2_range
    )

# 绘制热力图
plt.figure(figsize=(14, 8))
cmap = ListedColormap(['lightgray', 'lightblue', 'lightgreen', 'yellow', 'gold'])
boundaries = np.arange(-0.5, 5.5, 1)  # 设置边界
norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

plt.imshow(nr_stable_points.T, extent=[c1_range[0], c1_range[-1], c2_range[0], c2_range[-1]],
           origin='lower', aspect='auto', cmap=cmap, norm=norm)

# 添加色条
cbar = plt.colorbar()
cbar.set_ticks([0, 1, 2, 3, 4])
cbar.set_ticklabels(['0', '1', '2', '3', '4'])
cbar.set_label('Number of Stable Fixed Points')
# 添加虚线
x_val = 2. * np.sqrt(3) / 9.
y_val = 2. * np.sqrt(3) / 9.
plt.axvline(x=x_val, color='k', linestyle='--', linewidth=0.8)
plt.axhline(y=y_val, color='k', linestyle='--', linewidth=0.8)
plt.title('Stability Map of Bidirectionally Coupled Tipping Elements')
plt.xlabel('c1')
plt.ylabel('c2')

plt.show()