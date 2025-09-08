import numpy as np
import matplotlib.pyplot as plt

# 定义势函数
def potential_energy(x, c):
    return 0.25 * x**4 - 0.5 * x**2 - c * x

# 生成 x 的范围
x_values = np.linspace(-3, 3, 400)
c = 1  # 可以调整c的值
V_values = potential_energy(x_values, c)

# 绘制势函数
plt.figure(figsize=(10, 6))
plt.plot(x_values, V_values, label=f'Potential Energy V(x) for c={c}', color='blue')
plt.title('Potential Energy Function')
plt.xlabel('x')
plt.ylabel('V(x)')
plt.grid()
plt.ylim(-2,8)
plt.axhline(0, color='black',linewidth=0.5, ls='--')
plt.axvline(0, color='black',linewidth=0.5, ls='--')
plt.legend()
plt.show()