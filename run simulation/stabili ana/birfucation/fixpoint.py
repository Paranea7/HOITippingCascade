#!/usr/bin/env python3
"""
dx/dt = -x^3 + x 的相图与数值模拟示例

本脚本做以下工作：
1) 给出不动点（解析）并进行线性稳定性分析；
2) 在相空间（x轴为 x）上绘制向量场（一维相图上用箭头表示流向）；
3) 对若干初值用 solve_ivp 数值积分并在相图上绘出解轨迹（x 随时间变化的轨迹，投影到相图上就是曲线）；
4) 绘制对应的 x(t) 曲线（时间演化）；
5) 可选：保存图像（取消注释保存行）。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 定义右侧函数 f(x) = -x^3 + x
def f(t, x):
    # 这里写成接受 (t, x) 以兼容 solve_ivp 的调用约定
    return -x**3 + x + 1

# 不动点与线性化
fixed_points = np.array([-1.0, 0.0, 1.0])
# 导数 f'(x) = -3 x^2 + 1
def df_dx(x):
    return -3.0 * x**2 + 1.0

print("不动点（解析）:", fixed_points)
for xi in fixed_points:
    lam = df_dx(xi)
    stability = "不稳定 (源)" if lam > 0 else ("稳定 (汇)" if lam < 0 else "临界（需更高阶分析）")
    print(f"  x* = {xi: .3f}, f'(x*) = {lam: .3f}  => {stability}")

# 相图（1D）设置
x_min, x_max = -2.5, 2.5
x_vals = np.linspace(x_min, x_max, 400)
f_vals = -x_vals**3 + x_vals

# 绘制相图（向量场箭头）和不动点
fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)

ax_phase = axes[0]
# 在一维相空间中，用箭头表示流向：我们用 y=0 为基线，将箭头放在该线上
y_base = 0.0
ax_phase.plot(x_vals, np.zeros_like(x_vals), 'k:', linewidth=0.6)  # 基线
# 画箭头：箭头方向由 f(x) 决定，箭头长度与 f 的大小成比例（经过归一化）
# 采用若干点绘制箭头以免太密
arrow_x = np.linspace(x_min, x_max, 30)
arrow_u = -arrow_x**3 + arrow_x
# 归一化箭头长度用于绘图（避免箭头过长）
max_u = np.max(np.abs(arrow_u))
arrow_scale = 0.3 / max_u if max_u != 0 else 1.0
for xi, ui in zip(arrow_x, arrow_u):
    dx_plot = xi
    dy_plot = y_base
    dx_to = xi + ui * arrow_scale
    dy_to = y_base
    ax_phase.annotate('', xy=(dx_to, dy_to), xytext=(dx_plot, dy_plot),
                      arrowprops=dict(arrowstyle='->', color='C0'))

# 标出 f(x) 曲线以辅助理解（虽然相图在一维用箭头即可）
ax_phase.plot(x_vals, f_vals, color='C7', alpha=0.6, label='f(x) = -x^3 + x (辅助)')
# 标记不动点
ax_phase.plot(fixed_points, np.zeros_like(fixed_points), 'ro', label='平衡点')
for xi in fixed_points:
    ax_phase.text(xi, 0.05, f'x={xi:.0f}', ha='center')

ax_phase.set_xlim(x_min, x_max)
ax_phase.set_ylim(-0.5, 0.5)
ax_phase.set_yticks([])
ax_phase.set_xlabel('x')
ax_phase.set_title('相图与流向（1D）')
ax_phase.legend(loc='upper right')

# 数值积分：若干初值
t_span = (0.0, 10.0)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

initial_conditions = [-2.0, -0.5, 0.2, 0.8, 1.8]  # 可修改或扩展
colors = ['C0', 'C1', 'C2', 'C3', 'C4']

ax_time = axes[1]
for x0, c in zip(initial_conditions, colors):
    sol = solve_ivp(f, t_span, [x0], t_eval=t_eval, rtol=1e-8, atol=1e-10)
    x_t = sol.y[0]
    t = sol.t
    # 在相图上画出初值到终点的轨迹（作为曲线）
    ax_phase.plot(x_t, np.zeros_like(x_t), color=c, alpha=0.8)
    # 时间演化图
    ax_time.plot(t, x_t, color=c, label=f'x0={x0}')
    # 标注起点和终点
    ax_phase.plot(x0, 0.0, marker='o', color=c)
    ax_phase.plot(x_t[-1], 0.0, marker='s', color=c)

ax_time.set_xlabel('t')
ax_time.set_ylabel('x(t)')
ax_time.set_title('解的时间演化 x(t)')
ax_time.legend()
ax_time.grid(True)

# 保存图像（如果需要）
# plt.savefig('dxdt_phase_time.png', dpi=300)

plt.show()

# 额外演示：对于任意初值，推断其极限点（t->∞）
def asymptotic_limit(x0, t_final=50.0):
    """对给定初值 x0 做数值积分到较大时间，返回近似的极限值 x(t_final)."""
    sol = solve_ivp(f, (0.0, t_final), [x0], t_eval=[t_final], rtol=1e-8, atol=1e-10)
    return float(sol.y[0, -1])

print("\n一些初值的数值极限（t->50）示例：")
test_inits = [-2.0, -0.2, 0.2, 0.6, 1.5]
for x0 in test_inits:
    x_inf = asymptotic_limit(x0, t_final=100.0)
    print(f"  x0 = {x0: .2f}  -> x(t=100) ≈ {x_inf: .6f}")