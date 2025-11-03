import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 定义参数
c = np.random.normal(0.0, 1.0, size=10)
d = np.full((10, 10), 0.1)  # 10x10的相互作用项

# 定义高阶相互作用项的系数（此处为示例）
e = np.zeros((10, 10, 10))
for i in range(10):
    for j in range(10):
        for k in range(10):
            if i != j and i != k and j != k:
                e[i, j, k] = 0.1  # 设置非对角项的高阶相互作用系数




# 定义系统的方程
def system(t, x):  # 输入格式 `t`, `x`
    dxdt = np.zeros(10)  # 创建一个长度为10的数组

    for i in range(10):
        # 基本动态方程，应用随机放缩项
        dxdt[i] =x[i] ** 3 + x[i] + c[i]

        # 添加一次相互作用项
        for j in range(10):
            if j != i:
                dxdt[i] += d[i][j] * x[j]

        # 添加高阶相互作用项
        for j in range(10):
            for k in range(10):
                if j != i and k != i and j != k:
                    dxdt[i] += e[i][j][k] * x[j] * x[k]

    return dxdt


# 示范如何使用该系统 (例如，运行一段时间的ODE)
# 初始条件
x0 = 2 * np.random.rand(10) - 1

# 时间范围
t_span = [0, 100]
t_eval = np.linspace(t_span[0], t_span[1], 100)

# 求解ODE
solution = solve_ivp(system, t_span, x0, t_eval=t_eval)

# 绘制结果
plt.figure(figsize=(12, 8))
for i in range(10):
    plt.plot(solution.t, solution.y[i], label=f'x{i + 1}')
plt.xlabel('Time')
plt.ylabel('Variables')
plt.title('Dynamics of the Extended System with Random Scaling')
plt.legend()
plt.grid()
plt.show()