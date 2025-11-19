#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phi_beta_classify.py

在 (phi,beta) 平面上分类三次方程 m^3 - (1+beta)m - phi*C1 = 0 的根的符号情况：
 - "only_positive": 只有正实根（可能还有复根或重根，但不存在负实根或零根）
 - "pos_and_neg": 同时存在正实根和负实根（可能还有零根）
 - "no_positive": 没有正实根（可能有负实根或只有零或复根）

并且可选做一个简单的数值积分示例：从指定初值 m0（例如 -0.6）对微分方程 dm/dt = -m^3 + m + phi*C1 + beta*m 做时间积分，观察最终收敛到哪个稳定平衡并在图上标注。

用法：
 - 直接运行脚本会绘制相图并标注基于初值 m0=-0.6 的收敛类别点。
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from numpy import sqrt

# 常量
C1 = 2.0 * np.sqrt(3.0) / 9.0  # c1; c_bar = phi * C1

def delta_phi_beta(phi, beta):
    p = -(1.0 + beta)
    q = -phi * C1
    return (q/2.0)**2 + (p/3.0)**3

def roots_for_phi_beta(phi, beta, tol_real=1e-9):
    coeffs = [1.0, 0.0, -(1.0 + beta), -phi * C1]
    roots = np.roots(coeffs)
    real_roots = [r.real for r in roots if abs(r.imag) < tol_real]
    # sort real roots ascending for convenience
    real_roots.sort()
    return roots, real_roots

def classify_roots(phi, beta, tol_real=1e-9):
    """
    返回类别： 'only_positive', 'pos_and_neg', 'no_positive'
    说明：
     - only_positive: 至少有一个正实根，且没有负实根或零根（若有零且无负实根则算作 no_positive）
     - pos_and_neg: 同时至少有一个正实根和至少一个负实根
     - no_positive: 没有正实根（可能有负或零或全为复根）
    注：对零根的处理：若存在 root == 0（|root|<eps），视为非正根（计入 no_positive 或 pos_and_neg 的“负/零”一方）
    """
    _, real_roots = roots_for_phi_beta(phi, beta, tol_real=tol_real)
    eps_zero = 1e-12
    has_pos = any(r > eps_zero for r in real_roots)
    has_neg_or_zero = any(r < -eps_zero or abs(r) <= eps_zero for r in real_roots)
    if has_pos and not has_neg_or_zero:
        return 'only_positive'
    if has_pos and has_neg_or_zero:
        return 'pos_and_neg'
    return 'no_positive'

def integrate_trajectory(phi, beta, m0=-0.6, tmax=200.0, atol=1e-9, rtol=1e-8):
    """
    对微分方程 dm/dt = F(m) = -m^3 + m + phi*C1 + beta*m 进行数值积分
    使用 scipy.integrate.solve_ivp，从 t=0 到 t=tmax，检测是否在末端收敛到某个实根
    返回: (converged, m_final, root_matched or None)
    converged: bool（是否在 tmax 时看似收敛）
    m_final: 最后时刻的 m 值
    root_matched: 若收敛并匹配到某个实根，则返回该实根的值；否则 None
    """
    def F(t, m):
        return -m**3 + m + phi*C1 + beta*m

    sol = integrate.solve_ivp(F, [0.0, tmax], [m0], atol=atol, rtol=rtol, method='RK45', max_step=0.5)
    m_final = float(sol.y[0, -1])
    # 简单收敛判定：看末尾若干时间步是否变化很小
    # 若解在最后 10% 时间内波动幅度小于 tol_conv 则认为收敛
    n_steps = sol.y.shape[1]
    last_k = max(2, n_steps // 10)
    recent = sol.y[0, -last_k:]
    if np.max(np.abs(recent - recent[-1])) < 1e-6:
        converged = True
    else:
        converged = False

    # 若收敛，则尝试匹配到某个实根
    root_matched = None
    if converged:
        _, real_roots = roots_for_phi_beta(phi, beta)
        if real_roots:
            # 找到距离最近的实根
            diffs = [abs(m_final - r) for r in real_roots]
            idx = int(np.argmin(diffs))
            if diffs[idx] < 1e-4:  # 容差：若末值足够靠近某根则认为匹配
                root_matched = real_roots[idx]
    return converged, m_final, root_matched

def plot_phase_with_classification(phi_vals=None, beta_vals=None, m0=-0.6, do_integration=True):
    if phi_vals is None:
        phi_vals = np.linspace(0.0, 1.0, 201)
    if beta_vals is None:
        beta_vals = np.linspace(-2.0, 2.0, 401)
    PHI, BETA = np.meshgrid(phi_vals, beta_vals)

    # 准备分类网格
    class_grid = np.empty(PHI.shape, dtype=object)
    for i in range(PHI.shape[0]):
        for j in range(PHI.shape[1]):
            class_grid[i, j] = classify_roots(PHI[i, j], BETA[i, j])

    # 为绘图创建数值映射
    mapping = {'only_positive': 2, 'pos_and_neg': 1, 'no_positive': 0}
    numeric_grid = np.vectorize(mapping.get)(class_grid)

    fig, ax = plt.subplots(figsize=(8,6))
    cmap = plt.get_cmap('Set1')
    # uses 3 discrete colors
    im = ax.imshow(numeric_grid, origin='lower', extent=(phi_vals[0], phi_vals[-1], beta_vals[0], beta_vals[-1]),
                   aspect='auto', cmap=cmap, vmin=-0.5, vmax=2.5)
    # 增加颜色图例说明
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors = ['#d73027', '#fdae61', '#1a9850']  # mapping: no_positive, pos_and_neg, only_positive
    cmap2 = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap2.N)
    im = ax.imshow(numeric_grid, origin='lower', extent=(phi_vals[0], phi_vals[-1], beta_vals[0], beta_vals[-1]),
                   aspect='auto', cmap=cmap2, norm=norm)

    # 画 Delta=0 边界
    Delta_grid = delta_phi_beta(PHI, BETA)
    cs = ax.contour(PHI, BETA, Delta_grid, levels=[0.0], colors='k', linewidths=1.0)
    ax.clabel(cs, fmt={0.0: r'$\Delta=0$'})

    # 图例手工绘制
    import matplotlib.patches as mpatches
    p_no = mpatches.Patch(color=colors[0], label='no_positive (无正根)')
    p_both = mpatches.Patch(color=colors[1], label='pos_and_neg (正负根都有)')
    p_pos = mpatches.Patch(color=colors[2], label='only_positive (仅正根)')
    ax.legend(handles=[p_pos, p_both, p_no], loc='upper right')

    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\beta$')
    ax.set_title(f'Root sign classification; initial m0={m0}')

    # 若需要，用初值做积分并在图上标注积分分类点
    if do_integration:
        sample_phi = np.linspace(phi_vals[0], phi_vals[-1], 21)
        sample_beta = np.linspace(beta_vals[0], beta_vals[-1], 21)
        pts_phi = []
        pts_beta = []
        pts_marker = []
        pts_color = []
        for phi in sample_phi:
            for beta in sample_beta:
                converged, m_final, root_matched = integrate_trajectory(phi, beta, m0=m0, tmax=200.0)
                # 选择标记颜色/形状：若没有收敛则黑叉；否则按 root_matched 的正负或零标注
                if not converged:
                    pts_phi.append(phi); pts_beta.append(beta)
                    pts_marker.append('x'); pts_color.append('k')
                else:
                    if root_matched is None:
                        # 若收敛但未匹配到根（不常见），用灰点
                        pts_phi.append(phi); pts_beta.append(beta); pts_marker.append('.'); pts_color.append('0.5')
                    else:
                        if root_matched > 1e-8:
                            pts_phi.append(phi); pts_beta.append(beta); pts_marker.append('o'); pts_color.append('green')
                        elif root_matched < -1e-8:
                            pts_phi.append(phi); pts_beta.append(beta); pts_marker.append('s'); pts_color.append('red')
                        else:
                            pts_phi.append(phi); pts_beta.append(beta); pts_marker.append('d'); pts_color.append('blue')
        # 绘制这些点
        for ph, be, mk, col in zip(pts_phi, pts_beta, pts_marker, pts_color):
            ax.scatter(ph, be, marker=mk, color=col, s=30, edgecolors='k', linewidths=0.3, alpha=0.9)

        # 添加说明图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='converged -> positive root', markerfacecolor='green', markersize=6, markeredgecolor='k'),
            Line2D([0], [0], marker='s', color='w', label='converged -> negative root', markerfacecolor='red', markersize=6, markeredgecolor='k'),
            Line2D([0], [0], marker='d', color='w', label='converged -> zero root', markerfacecolor='blue', markersize=6, markeredgecolor='k'),
            Line2D([0], [0], marker='x', color='k', label='no convergence', markersize=6),
        ]
        ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 运行并绘图
    # 你可以调整 phi/beta 的范围与采样密度以加快或精细化图像
    plot_phase_with_classification(phi_vals=np.linspace(0,1,201), beta_vals=np.linspace(-2,2,401), m0=-0.6, do_integration=True)