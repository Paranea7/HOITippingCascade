#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phi_beta_roots.py

计算并可视化参数 (phi, beta) 下三次方程
    m^3 - (1+beta) m - c_bar = 0,
其中 c_bar = phi * (2*sqrt(3)/9).

功能：
- 计算判别式 Delta(phi,beta) = (q/2)^2 + (p/3)^3，p=-(1+beta), q=-c_bar
- 在给定 (phi,beta) 点上数值解出实根并报告正根
- 在参数网格上绘制 Delta=0 边界并用颜色表示 Delta 符号 / 正根存在性

依赖：numpy, scipy, matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt
from scipy import optimize
from numpy import real
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
# 常量
C1 = 2.0 * np.sqrt(3.0) / 9.0  # c1; c_bar = phi * C1

def delta_phi_beta(phi, beta):
    """
    计算判别式 Delta = (q/2)^2 + (p/3)^3
    p = -(1+beta), q = -phi*C1
    """
    p = -(1.0 + beta)
    q = -phi * C1
    Delta = (q / 2.0)**2 + (p / 3.0)**3
    return Delta

def roots_for_phi_beta(phi, beta):
    """
    返回三次方程的三个数值根（可能含复数），以及只取实部的实根列表（用 tol 判定实数）
    方程： m^3 - (1+beta) m - phi*C1 = 0
    """
    # 多项式系数按 numpy.poly1d 形式从高次到常数： [1, 0, -(1+beta), -phi*C1]
    coeffs = [1.0, 0.0, -(1.0 + beta), -phi * C1]
    roots = np.roots(coeffs)  # 可能有复根
    # 近似判定为实根的容差
    tol = 1e-9
    real_roots = []
    for r in roots:
        if abs(r.imag) < tol:
            real_roots.append(r.real)
    # 也返回所有根（含复数）
    return roots, real_roots

def positive_roots_for_phi_beta(phi, beta):
    roots, real_roots = roots_for_phi_beta(phi, beta)
    pos = [r for r in real_roots if r > 0]
    return pos

# 示例：在给定点上输出信息
def report_point(phi, beta):
    Delta = delta_phi_beta(phi, beta)
    all_roots, real_roots = roots_for_phi_beta(phi, beta)
    pos_roots = [r for r in real_roots if r > 0]
    print(f"phi={phi:.4g}, beta={beta:.4g}, Delta={Delta:.6g}")
    print(f"  all roots (may be complex): {all_roots}")
    print(f"  real roots: {real_roots}")
    print(f"  positive real roots: {pos_roots}")
    # 额外提示
    if Delta > 0:
        print("  判别式 Δ>0 -> 仅 1 个实根（其余为复共轭对）。")
    elif abs(Delta) < 1e-12:
        print("  判别式 Δ≈0 -> 重根情形（3 个实根且含重根）。")
    else:
        print("  判别式 Δ<0 -> 3 个互异实根。")
    print("  注：若 beta<=-1，则三角解形式不适用，需要数值根求解（此处已使用数值根）。")
    print()

# 在参数网格上绘制 Delta 符号图与 Delta=0 边界
def plot_phase_diagram(phi_vals=None, beta_vals=None, show_positive_region=True):
    if phi_vals is None:
        phi_vals = np.linspace(0.0, 1.0, 201)  # phi>=0 的常用区间
    if beta_vals is None:
        beta_vals = np.linspace(-2.0, 2.0, 401)
    PHI, BETA = np.meshgrid(phi_vals, beta_vals)
    Delta_grid = delta_phi_beta(PHI, BETA)

    fig, ax = plt.subplots(figsize=(7,5))
    # 用 colormap 显示 Delta 的符号（正/负），并把Delta=0以黑线绘出
    cmap = plt.get_cmap('RdYlBu')
    im = ax.contourf(PHI, BETA, Delta_grid, levels=50, cmap=cmap, alpha=0.9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Delta value')

    # 绘制 Delta=0 边界（可能多段）
    cs = ax.contour(PHI, BETA, Delta_grid, levels=[0.0], colors='k', linewidths=1.2)
    ax.clabel(cs, fmt={0.0: r'$\Delta=0$'})

    if show_positive_region:
        # 在网格上标识是否存在正根（布尔矩阵）
        has_pos = np.zeros_like(Delta_grid, dtype=bool)
        # 为速度考虑可稀疏检查或向量化（此处直接循环）
        for i in range(PHI.shape[0]):
            for j in range(PHI.shape[1]):
                phi = PHI[i,j]; beta = BETA[i,j]
                # 数值求根并判断是否存在正实根
                _, real_roots = roots_for_phi_beta(phi, beta)
                if any(r > 1e-12 for r in real_roots):
                    has_pos[i,j] = True
        # 用透明度遮罩在图上显示存在正根的区域
        ax.contourf(PHI, BETA, has_pos, levels=[0.5, 1.5], colors=['none','none'],
                    hatches=['', '///'], alpha=0.0)
        # 添加图例说明
        ax.text(0.02, 0.95, '/// 区域表示存在正根', transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\beta$')
    ax.set_title(r'Parameter plane: $\Delta(\phi,\beta)$ and $\Delta=0$ boundary')
    plt.tight_layout()
    plt.show()

# 若作为脚本运行，则演示几个典型点和绘图
if __name__ == "__main__":
    # 若要测试若干点
    test_points = [
        (0.0, 0.5),
        (0.0, -0.5),
        (0.2, 0.0),
        (0.2, -1.5),
        (0.5, 0.0),
        (0.8, 1.0),
    ]
    for phi, beta in test_points:
        report_point(phi, beta)

    # 绘制相图（可能耗时）
    plot_phase_diagram(phi_vals=np.linspace(0,1,201), beta_vals=np.linspace(-2,2,401))