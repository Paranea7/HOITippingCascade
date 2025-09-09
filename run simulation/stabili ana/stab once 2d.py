import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def find_fixed_points(c1, c2):
    # 形成多项式并解出固定点
    coeffs_x1 = [-1, 0, 1, c1]  # x1 的多项式系数
    fixed_points_x1 = np.roots(coeffs_x1)  # 计算固定点

    coeffs_x2 = [-1, 0, 1, c2]  # x2 的多项式系数
    fixed_points_x2 = np.roots(coeffs_x2)  # 计算固定点

    return fixed_points_x1, fixed_points_x2

def stability(x):
    # 定义稳定性函数
    return -3 * x ** 2 + 1

def is_stable(x):
    # 检查固定点是否稳定
    return np.isreal(x) and stability(np.real(x)) < 0

# 参数范围和耦合强度
c1_range = np.linspace(0, 0.8, 500)
c2_range = np.linspace(0, 0.8, 500)
nr_stable_points = np.zeros((len(c2_range), len(c1_range)))

# 计算稳定点数量以及相图
for i, c1 in enumerate(c1_range):
    for j, c2 in enumerate(c2_range):
        # 找到 x1 和 x2 的固定点
        fixed_points_x1, fixed_points_x2 = find_fixed_points(c1, c2)
        # 统计稳定点 x1
        stable_x1_count = 0  # 初始化稳定点计数器
        for x in fixed_points_x1:  # 遍历所有固定点 x1
            if is_stable(x):  # 判断当前固定点 x1 是否稳定
                stable_x1_count += 1  # 如果稳定，则计数器加一

        # 统计稳定点 x2
        stable_x2_count = 0  # 初始化稳定点计数器
        for x in fixed_points_x2:  # 遍历所有固定点 x2
            if is_stable(x):  # 判断当前固定点 x2 是否稳定
                stable_x2_count += 1  # 如果稳定，则计数器加一
        # 稳定点总数
        total_stable_points = stable_x1_count + stable_x2_count
        nr_stable_points[j, i] = total_stable_points  # 存储稳定点总数

# 修改 stable_points 的 color map 及边界设置
plt.figure(figsize=(14, 8))
cmap = ListedColormap(['lightgray', 'lightblue', 'lightgreen', 'yellow', 'gold'])
boundaries = [0, 1, 2, 3, 4, 5]  # 添加一个额外的边界，以确保规范覆盖可能的稳定点数量（最小值为 0，到最高 4）
norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

# 显示稳定性图
plt.imshow(nr_stable_points, extent=[c1_range[0], c1_range[-1], c2_range[0], c2_range[-1]],
           origin='lower', aspect='auto', cmap=cmap, norm=norm)

# 添加离散色条
cbar = plt.colorbar()
cbar.set_ticks([0, 1, 2, 3, 4])  # 指定刻度
cbar.set_ticklabels(['0', '1', '2', '3', '4'])  # 指定标签
cbar.set_label('Number of Stable Fixed Points')

# 在指定位置添加虚线
x_val = 2.*np.sqrt(3)/9.
y_val = 2.*np.sqrt(3)/9.
plt.axvline(x=x_val, color='k', linestyle='--', linewidth=0.8)  # 垂直虚线
plt.axhline(y=y_val, color='k', linestyle='--', linewidth=0.8)  # 水平虚线

plt.title('Stability Map of Bidirectionally Coupled Tipping Elements')
plt.xlabel('c1')
plt.ylabel('c2')

plt.show()