#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-node bistable system  —— 伪弧长continuation 画分岔图
dx_i/dt = -x_i^3 + x_i + c_i + d*sum_{j≠i}x_j + e*x_j*x_k
----------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt

# -------------------- 参数 --------------------
c = np.array([0.2, 0.0, 0.0])     # 仅单元1被驱动
e = 0.2                           # 高阶耦合强度 ε₂（固定）
ds0 = 0.01                       # 初始步长（可正可负）
ds_min, ds_max = 1e-4, 0.05
max_arclen = 3.0                  # 总弧长上限
# --------------------------------------------

def rhs(x):
    """右侧动力学 f(x) （用于数值 Jacobian）"""
    x1, x2, x3 = x
    d_mat = d*(1-np.eye(3))
    ho = e*np.array([x2*x3, x1*x3, x1*x2])
    return -x**3 + x + c + d_mat@x + ho

def F(x):
    """稳态方程 F(x)=0 """
    return rhs(x)

def jac(x):
    """解析 Jacobian （3×3）"""
    x1, x2, x3 = x
    J = np.zeros((3, 3))
    # 对角
    for i in range(3):
        J[i, i] = -3*x[i]**2 + 1
    # 成对
    off = d*(1-np.eye(3))
    J += off
    # 高阶
    J[0, 1] += e*x3;  J[0, 2] += e*x2
    J[1, 0] += e*x3;  J[1, 2] += e*x1
    J[2, 0] += e*x2;  J[2, 1] += e*x1
    return J

def newton(x, tol=1e-10, maxit=20):
    """Newton-Raphson 求解 F(x)=0 """
    for _ in range(maxit):
        dx = np.linalg.solve(jac(x), -F(x))
        x = x + dx
        if np.linalg.norm(dx) < tol:
            break
    return x

def pseudo_arc_continuation(x0, d0, ds0):
    """伪弧长延拓，返回整条曲线 (d_vals, X_vals) """
    global d  # 让 F/jac 能看到当前 d
    x, d = x0.copy(), d0
    ds = ds0

    # 初始解
    x = newton(x)
    X_hist, d_hist = [x.copy()], [d]

    # 切向量初始化 （简单数值差分）
    eps = 1e-6
    d_plus = d + eps
    d = d_plus;  x_plus = newton(x + np.zeros(3))
    d = d - eps;  x_minus = newton(x + np.zeros(3))
    d = d_hist[-1]
    tangent = np.hstack([(x_plus - x_minus)/(2*eps), 1.0])
    tangent /= np.linalg.norm(tangent)

    arclen = 0.0
    while arclen < max_arclen:
        # 预测步
        x_pred = x + ds*tangent[:3]
        d_pred = d + ds*tangent[3]
        d = d_pred
        # 校正步 （Newton on augmented system）
        for _ in range(15):
            r = F(x_pred)
            J = jac(x_pred)
            aug = np.vstack([J, tangent])
            rhs = np.hstack([-r, 0.0])
            delta = np.linalg.lstsq(aug, rhs, rcond=None)[0]
            x_pred += delta[:3]
            d += delta[3]
            if np.linalg.norm(delta) < 1e-10:
                break
        x = x_pred
        # 更新切向量
        eps = 1e-6
        d_plus = d + eps
        d = d_plus;  x_plus = newton(x + 0*x)
        d = d - 2*eps; x_minus = newton(x + 0*x)
        d = d + eps
        new_tan = np.hstack([(x_plus - x_minus)/(2*eps), 1.0])
        new_tan /= np.linalg.norm(new_tan)
        tangent = new_tan

        X_hist.append(x.copy())
        d_hist.append(d)
        arclen += abs(ds)

        # 简单自适应步长
        if np.linalg.norm(delta) < 5e-11:
            ds = min(ds*1.5, ds_max)
        elif np.linalg.norm(delta) > 1e-9:
            ds = max(ds/2, ds_min)

    return np.array(d_hist), np.array(X_hist)

# -------------------- 主程序 --------------------
if __name__ == "__main__":
    d_start = 0.0
    x0 = np.array([0.1, 0.1, 0.1])

    # 正向延拓
    d_pos, X_pos = pseudo_arc_continuation(x0, d_start, ds0)
    # 反向延拓
    d_neg, X_neg = pseudo_arc_continuation(x0, d_start, -ds0)

    # 合并
    d_vals = np.hstack([d_neg[::-1], d_pos[1:]])
    X_vals = np.vstack([X_neg[::-1], X_pos[1:]])

    # -------------------- 绘图 --------------------
    plt.figure(figsize=(5, 4))
    plt.plot(d_vals, X_vals[:, 1], 'r', lw=2, label=r'$x_2$')
    plt.plot(d_vals, X_vals[:, 2], 'b', lw=2, label=r'$x_3$')
    plt.xlabel(r'pairwise coupling $d$')
    plt.ylabel(r'steady state $x_i$')
    plt.title(f'3-node bifurcation (e={e}, c_1=0.2)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()