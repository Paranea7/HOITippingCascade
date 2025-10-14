#!/usr/bin/env python3
"""
绘制势函数与力函数（不显示时间演化轨迹）
动力学: dx/dt = -x^3 + x + c
势函数: V(x) = x^4/4 - x^2/2 - c*x
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import matplotlib



matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
# ---------- 参数 ----------
c = 0.3             # 常数偏置 c，可修改
x_min, x_max = -3.0, 3.0
n_points = 1000

# ---------- 定义函数 ----------
def f(x, c=c):
    """力函数 f(x) = -x^3 + x + c"""
    return -x**3 + x + c

def V(x, c=c):
    """势函数 V(x) = x^4/4 - x^2/2 - c*x"""
    return 0.25 * x**4 - 0.5 * x**2 - c * x

# 网格与计算
x = np.linspace(x_min, x_max, n_points)
v = V(x, c)
f_vals = f(x, c)

# ---------- 找平衡点（f(x)=0） ----------
roots = []
for i in range(len(x)-1):
    a, b = x[i], x[i+1]
    fa, fb = f(a, c), f(b, c)
    if fa == 0:
        roots.append(a)
    elif fa * fb < 0:
        try:
            r = brentq(lambda xx: f(xx, c), a, b)
            roots.append(r)
        except ValueError:
            pass
# 去重与排序
roots = np.array(sorted(set([float(np.round(rr, 12)) for rr in roots])))
print("找到的平衡点（数值）:", roots)

# 线性化导数 f'(x) = -3 x^2 + 1
def df_dx(x):
    return -3.0 * x**2 + 1.0

# 打印稳定性信息
for xi in roots:
    lam = df_dx(xi)
    stability = "不稳定 (源)" if lam > 0 else ("稳定 (汇)" if lam < 0 else "临界（需高阶分析）")
    print(f"  x* = {xi:.6f}, f'(x*) = {lam:.6f}  => {stability}")

# ---------- 绘图 ----------
fig, (ax_v, ax_f) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)

# 势函数图
ax_v.plot(x, v, color='C0', lw=2, label=f'V(x) = x^4/4 - x^2/2 - c x   (c={c})')
if roots.size > 0:
    ax_v.plot(roots, V(roots, c), 'ro', label='平衡点')
    for xi in roots:
        ax_v.text(xi, V(xi, c) + 0.04*np.ptp(v), f'{xi:.3f}', ha='center')
ax_v.set_xlabel('x')
ax_v.set_ylabel('V(x)')
ax_v.set_title('势函数 V(x)')
ax_v.grid(True)
ax_v.legend()

# 力函数图与 -dV/dx 验证
ax_f.plot(x, f_vals, color='C1', lw=2, label=f'f(x) = -x^3 + x + c (c={c})')
dVdx_num = np.gradient(v, x)
ax_f.plot(x, -dVdx_num, '--', color='C2', lw=1.5, label='-dV/dx (数值差分)')
ax_f.axhline(0, color='k', lw=0.6)
if roots.size > 0:
    ax_f.plot(roots, f(roots, c), 'ro')
    for xi in roots:
        ax_f.text(xi, 0.05*np.ptp(f_vals), f'{xi:.3f}', ha='center')
ax_f.set_xlabel('x')
ax_f.set_ylabel('f(x)')
ax_f.set_title('力函数 f(x) 与 -dV/dx 验证')
ax_f.grid(True)
ax_f.legend()

plt.show()