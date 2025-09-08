import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 定义参数
c1 = c2 = 0.2
d12 = 0.2
d21= 0.0

# 定义系统的方程
def system(X):
    x1, x2 = X
    dx1dt = -x1**3 + x1 + c1 + d21 * x2
    dx2dt = -x2**3 + x2 + c2 + d12 * x1
    return np.array([dx1dt, dx2dt])

# 使用fsolve找到不动点
initial_guesses = [[0, 0], [-1, -1], [1, 1], [2, 2], [-2, -2]]
fixed_points = [fsolve(system, guess) for guess in initial_guesses]

# 创建网格
x1 = np.linspace(-2, 2, 20)
x2 = np.linspace(-2, 2, 20)
X1, X2 = np.meshgrid(x1, x2)

# 计算向量场
DX1, DX2 = system((X1, X2))

# 绘制连续箭头（流线）
plt.figure(figsize=(10, 8))
plt.streamplot(X1, X2, DX1, DX2, color='r', linewidth=1, density=1.5)

# 标记不动点
for point in fixed_points:
    plt.plot(point[0], point[1], 'bo')  # 绘制不动点为蓝色圆点
    plt.text(point[0], point[1], f'({point[0]:.2f}, {point[1]:.2f})', fontsize=12, ha='right')

plt.title('Vector Field of the Coupled System with Fixed Points')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid()
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.show()