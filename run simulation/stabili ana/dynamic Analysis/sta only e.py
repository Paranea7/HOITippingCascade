#!/usr/bin/env python3
"""
scan_eps123_eps231.py

在 eps123 - eps231 平面上扫描，固定 c1=c2=c3=0。
对每个参数点，寻找系统平衡并统计稳定平衡点数量。

依赖:
  numpy, scipy, tqdm, matplotlib (绘图可选)
运行:
  python scan_eps123_eps231.py
"""

import os
import math
import numpy as np
from copy import deepcopy
from itertools import product
from tqdm import tqdm

from scipy.optimize import root
# 可选绘图
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# -----------------------
# 配置区（可修改）
# -----------------------

# 默认所有六个 eps（三体耦合系数）
DEFAULT_COEFFS = {
    'eps123': 0.0,
    'eps231': 0.0,
    'eps312': 0.0,
    'eps321': 0.0,
    'eps213': 0.0,
    'eps132': 0.0
}

# 指定扫描的两个参数
SCAN_PARAM_X = 'eps123'
SCAN_PARAM_Y = 'eps231'

# 固定 c 向量（你要求 c1=c2=0 且 c3=0）
C_FIXED = (0.0, 0.0, 0.0)

# 网格设置：eps123 在 X_vals（行），eps231 在 Y_vals（列）
X_vals = np.linspace(-1.0, 1.0, 41)    # eps123 的取值，可修改
Y_vals = np.linspace(-1.0, 1.0, 41)    # eps231 的取值，可修改

# 初值集合（用于多起点寻根）
# 这里使用若干组合，包括原点、单位向量、小扰动等
INITIAL_XS = [
    np.array([0.0, 0.0, 0.0]),
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, -1.0, 0.0]),
    np.array([0.0, 0.0, -1.0]),
    np.array([0.5, 0.5, 0.5]),
    np.array([-0.5, 0.5, -0.5]),
    np.array([0.2, -0.2, 0.1])
]

# 寻根与去重的容忍阈值
F_TOL = 1e-8         # f 的容忍
ROOT_TOL = 1e-6      # 解去重时的距离阈值
MAX_STABLE_DISPLAY = 1000

# 输出文件
OUT_DIR = "scan_results"
OUT_FILENAME = os.path.join(OUT_DIR, "eps123_eps231_scan.npz")
PLOT_FILENAME = os.path.join(OUT_DIR, "eps123_eps231_heatmap.png")

# -----------------------
# 模型定义：f(x, c, eps) 与雅可比 J(x, c, eps)
# 这里按照你之前的描述建立三个方程（示例形式）
# 你可以根据真实模型修改 f1,f2,f3 的具体形式
# -----------------------

def f_and_J(x, c, eps):
    """
    计算 f(x; c, eps) 和雅可比 J(x; c, eps)。

    x: 长度3数组
    c: (c1, c2, c3)
    eps: 字典，包含六个 eps 键
    返回: (f_vec, J_mat)
    """
    x1, x2, x3 = float(x[0]), float(x[1]), float(x[2])
    c1, c2, c3 = float(c[0]), float(c[1]), float(c[2])

    e123 = float(eps['eps123'])
    e231 = float(eps['eps231'])
    e312 = float(eps['eps312'])
    e321 = float(eps['eps321'])
    e213 = float(eps['eps213'])
    e132 = float(eps['eps132'])

    # 示例动力学（可以根据你的系统替换）
    # f1 = -x1 + c1 + e123*x2*x3 + e132*x3*x2
    # f2 = -x2 + c2 + e231*x3*x1 + e213*x1*x3
    # f3 = -x3 + c3 + e312*x1*x2 + e321*x2*x1
    # 上面项中 e123*x2*x3 与 e132*x3*x2 是等价的；保留两项以匹配原脚本结构。

    f1 = -x1 + c1 + e123 * x2 * x3 + e132 * x3 * x2
    f2 = -x2 + c2 + e231 * x3 * x1 + e213 * x1 * x3
    f3 = -x3 + c3 + e312 * x1 * x2 + e321 * x2 * x1

    f = np.array([f1, f2, f3], dtype=float)

    # 雅可比矩阵 J_ij = df_i / dx_j
    J = np.zeros((3, 3), dtype=float)

    # df1/dx1 = -1
    J[0, 0] = -1.0
    # df1/dx2 = e123 * x3 + e132 * x3
    J[0, 1] = e123 * x3 + e132 * x3
    # df1/dx3 = e123 * x2 + e132 * x2
    J[0, 2] = e123 * x2 + e132 * x2

    # df2/dx1 = e231 * x3 + e213 * x3
    J[1, 0] = e231 * x3 + e213 * x3
    # df2/dx2 = -1
    J[1, 1] = -1.0
    # df2/dx3 = e231 * x1 + e213 * x1
    J[1, 2] = e231 * x1 + e213 * x1

    # df3/dx1 = e312 * x2 + e321 * x2
    J[2, 0] = e312 * x2 + e321 * x2
    # df3/dx2 = e312 * x1 + e321 * x1
    J[2, 1] = e312 * x1 + e321 * x1
    # df3/dx3 = -1
    J[2, 2] = -1.0

    return f, J

# -----------------------
# 参数构造函数
# -----------------------

def make_eps_dict(var_x_val, var_y_val):
    """
    构造 eps 字典：默认值来自 DEFAULT_COEFFS，
    将 SCAN_PARAM_X 设为 var_x_val，SCAN_PARAM_Y 设为 var_y_val。
    """
    eps = deepcopy(DEFAULT_COEFFS)
    eps[SCAN_PARAM_X] = float(var_x_val)
    eps[SCAN_PARAM_Y] = float(var_y_val)
    return eps

# -----------------------
# 寻根、去重、稳定性判定
# -----------------------

def find_roots_and_count_stable(eps, c_vec):
    """
    对给定 eps 和固定 c_vec，使用多初值寻根，去重并统计稳定平衡点数量。
    返回 n_stable（整数）和 roots_list（去重后的平衡点列表）。
    """
    roots = []

    for x0 in INITIAL_XS:
        # 使用 scipy.optimize.root（默认方法：hybr）
        try:
            sol = root(lambda xx: f_and_J(xx, c_vec, eps)[0], x0, tol=1e-10)
        except Exception:
            # 某些起点可能导致异常，跳过
            continue

        if not sol.success:
            # 若未收敛，跳过
            continue

        x_sol = sol.x.astype(float)
        # 检验残差
        fval = f_and_J(x_sol, c_vec, eps)[0]
        if np.linalg.norm(fval) > 1e-6:
            continue

        # 去重：若与已有根距离小于 ROOT_TOL 则认为相同
        duplicated = False
        for r in roots:
            if np.linalg.norm(r - x_sol) < ROOT_TOL:
                duplicated = True
                break
        if not duplicated:
            roots.append(x_sol)

    # 判定稳定性（实部全部 < 0）
    n_stable = 0
    stable_roots = []
    for r in roots:
        _, J = f_and_J(r, c_vec, eps)
        eigs = np.linalg.eigvals(J)
        if np.all(np.real(eigs) < 0.0):
            n_stable += 1
            stable_roots.append(r)

    # 限制最大返回值以防意外
    n_stable = int(min(n_stable, MAX_STABLE_DISPLAY))
    return n_stable, roots, stable_roots

# -----------------------
# 主扫描函数
# -----------------------

def run_scan():
    X = np.array(X_vals, dtype=float)
    Y = np.array(Y_vals, dtype=float)
    nx = X.size
    ny = Y.size

    # 结果矩阵：行对应 X（eps123），列对应 Y（eps231）
    result = np.zeros((nx, ny), dtype=int)

    # 可选：记录每个点的根（会占用较多内存，注释掉以节省）
    save_roots = False
    roots_map = {} if save_roots else None

    c_vec = tuple(float(v) for v in C_FIXED)  # 确保浮点数

    # 扫描
    total = nx * ny
    pbar = tqdm(total=total, desc="Scanning eps grid")
    for i, xv in enumerate(X):
        for j, yv in enumerate(Y):
            eps = make_eps_dict(xv, yv)
            n_stable, roots, stable_roots = find_roots_and_count_stable(eps, c_vec)
            result[i, j] = n_stable
            if save_roots:
                roots_map[(i, j)] = {
                    'eps': eps,
                    'roots': roots,
                    'stable_roots': stable_roots
                }
            pbar.update(1)
    pbar.close()

    # 保存结果
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(OUT_FILENAME, X=X, Y=Y, result=result)
    print("Saved scan result to:", OUT_FILENAME)

    # 可选绘图
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(6, 5))
        # imshow 需要 (ny,nx) 或 origin 调整：我们转置 result 以便 X 横轴, Y 纵轴
        im = ax.imshow(result.T, origin='lower',
                       extent=(X.min(), X.max(), Y.min(), Y.max()),
                       aspect='auto', cmap='viridis')
        ax.set_xlabel(SCAN_PARAM_X)
        ax.set_ylabel(SCAN_PARAM_Y)
        ax.set_title("Number of stable fixed points (c fixed at {})".format(c_vec))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('n_stable')
        plt.tight_layout()
        plt.savefig(PLOT_FILENAME, dpi=200)
        print("Saved heatmap to:", PLOT_FILENAME)
    else:
        print("matplotlib not available; skipping plot.")

    return X, Y, result

# -----------------------
# 主入口
# -----------------------

if __name__ == "__main__":
    X, Y, result = run_scan()
    # 简单打印一些统计信息
    unique, counts = np.unique(result, return_counts=True)
    print("Unique stable-count values and frequencies:")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c} grid points")
    print("Done.")