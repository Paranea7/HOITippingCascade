import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 定义微分方程
def f(x, c):
    return x - x ** 3 + c

# 判别不动点个数的函数
def discriminant(c):
    delta = 4 - 27 * c**2
    if delta > 0:
        return 3
    elif delta == 0:
        return 2
    else:
        return 1

# 定义 x'-x 图的函数
def phase_plot(ax, c):
    x = np.linspace(-2, 2, 1000)  # 选择 x 的范围
    x_dot = f(x, c)

    ax.clear()
    ax.plot(x, x_dot)
    ax.set_xlabel('x')
    ax.set_ylabel("x'")
    ax.set_title(f"Phase Plot for x'-x (c={c})")
    ax.grid(True)

    # 寻找不动点
    fixed_points = x[np.abs(x_dot) < 0.01]  # 找到 x' 接近 0 的点

    # 绘制不动点
    ax.plot(fixed_points, np.zeros_like(fixed_points), 'ro')

    # 显示不动点个数
    num_fixed_points = discriminant(c)
    ax.text(-1.9, np.max(x_dot)*0.8, f'Number of fixed points: {num_fixed_points}', fontsize=12, color='blue')

# 设置参数 c 的范围
c_values = np.linspace(-1, 1, 101)

# 创建图形和轴
fig, ax = plt.subplots()

# 动画函数
def animate(i):
    phase_plot(ax, c_values[i])

# 创建动画
ani = animation.FuncAnimation(fig, animate, frames=len(c_values), interval=100)

# 显示动画
plt.show()
