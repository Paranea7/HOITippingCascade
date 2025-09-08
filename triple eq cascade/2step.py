import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 参数
a = 1.0
b = 1.0
c = 0.3
fixed_damping_coeff = 0.2  # 固定阻尼系数

def gamma_func(t, amplitude, frequency, offset):
    return amplitude * np.sin(frequency * t) + offset

# 时间变化阻尼系数的系统函数
def system_time_varying(t, y):
    x, v = y
    dxdt = v
    dvdt = -a * x**3 + b * x - c - gamma_func(t, 0.1, 0.1, 0.2) * v
    return [dxdt, dvdt]

# 固定阻尼系数的系统函数
def system_fixed_damping(t, y):
    x, v = y
    dxdt = v
    dvdt = -a * x**3 + b * x - c - fixed_damping_coeff * v
    return [dxdt, dvdt]

# 初始条件
x0 = 0.0
v0 = 0.0
t_span = [0, 100]
t_eval = np.linspace(0, 100, 1000)

# 求解时间变化阻尼系数的系统
sol_time_varying = solve_ivp(system_time_varying, t_span, [x0, v0], t_eval=t_eval)

# 求解固定阻尼系数的系统
sol_fixed_damping = solve_ivp(system_fixed_damping, t_span, [x0, v0], t_eval=t_eval)

# 绘制位置随时间变化
plt.figure(figsize=(10, 10))

# 合并时间变化和固定阻尼的子图
plt.subplot(2, 1, 1)
plt.plot(sol_time_varying.t, sol_time_varying.y[0], label='Time-Varying Damping')
plt.plot(sol_fixed_damping.t, sol_fixed_damping.y[0], label='Fixed Damping', color='orange')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Position vs Time')
plt.legend()
plt.grid(True)

# 合并相图
plt.subplot(2, 1, 2)
plt.plot(sol_time_varying.y[0], sol_time_varying.y[1], label='Time-Varying Damping')
plt.plot(sol_fixed_damping.y[0], sol_fixed_damping.y[1], label='Fixed Damping', color='orange')
plt.xlabel('x')
plt.ylabel('v')
plt.title('Phase Portrait')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
