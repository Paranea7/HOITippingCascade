#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stability_threebody_3system_2D_cusp.py

S=3 的系统：
  dx_i/dt = -x_i^3 + x_i + c_i + sum_{j,k} e_{i j k} x_j x_k

只保留 6 个互不相等索引的三体耦合:
  eps123, eps231, eps312, eps321, eps213, eps132

在 (c1,c2) 平面上绘制每个点的稳定平衡点数量（c3 固定）。
"""
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
from matplotlib.colors import BoundaryNorm

# ---------------- 用户可调参数 ----------------
# 要扫描的两个 eps 分量名称（从下面六个中选两个）
# 'eps123','eps231','eps312','eps321','eps213','eps132'
SCAN_PARAM_X = 'eps123'
SCAN_PARAM_Y = 'eps231'

# 扫描 eps 值列表（示例）
X_vals = [0.0, 0.2, 0.5]
Y_vals = [0.0, -0.2, -0.5]

# 其余 eps 的默认值
DEFAULT_COEFFS = {
    'eps123': 0.0,
    'eps231': 0.0,
    'eps312': 0.0,
    'eps321': 0.0,
    'eps213': 0.0,
    'eps132': 0.0
}
# 你可以在这里修改默认值，例如:
# DEFAULT_COEFFS['eps312'] = 0.1

# 固定 c3
C3_FIXED = 0.0

# c1-c2 网格设置
c1_min, c1_max, N_c1 = -1.0, 1.0, 200   # 分辨率可调
c2_min, c2_max, N_c2 = -1.0, 1.0, 200
c1_arr = np.linspace(c1_min, c1_max, N_c1)
c2_arr = np.linspace(c2_min, c2_max, N_c2)

# Newton 初值网格（x 空间）
NX0 = 5
x1s0 = np.linspace(-2.0, 2.0, NX0)
x2s0 = np.linspace(-2.0, 2.0, NX0)
x3s0 = np.linspace(-2.0, 2.0, NX0)

# Newton 与去重参数
NEWTON_MAX_IT = 80
NEWTON_ATOL = 1e-10
F_TOL = 1e-8
ROOT_DUP_TOL = 1e-4   # 定点去重阈值（可调）

# 并行设置
WORKERS = None  # None -> mp.cpu_count()

# 可视化设定
MAX_STABLE_DISPLAY = 6   # 超过则显示为最大颜色
cmap = plt.get_cmap('viridis', MAX_STABLE_DISPLAY + 1)
bounds = np.arange(-0.5, MAX_STABLE_DISPLAY + 1.5, 1.0)
norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N, clip=True)

OUTDIR = 'stability_threebody_3system_results_2D_cusp'
os.makedirs(OUTDIR, exist_ok=True)

# ---------------- 帮助：构造 eps 字典 ----------------
def make_eps_dict(var_x_val, var_y_val):
    eps = DEFAULT_COEFFS.copy()
    eps[SCAN_PARAM_X] = float(var_x_val)
    eps[SCAN_PARAM_Y] = float(var_y_val)
    return eps

# ---------------- 模型与雅可比 ----------------
def f_and_J(x, c, eps):
    """
    输入:
      x: 长度3向量 (x1,x2,x3)
      c: 长度3向量 (c1,c2,c3)
      eps: dict 包含六个 eps 键
    输出:
      f: 长度3向量 f(x)
      J: 3x3 雅可比矩阵 J_{i,m} = ∂f_i/∂x_m
    f_i = -x_i^3 + x_i + c_i + sum_{j,k} eps_{i j k} x_j x_k
    """
    x1, x2, x3 = x
    c1, c2, c3 = c

    e123 = eps['eps123']
    e231 = eps['eps231']
    e312 = eps['eps312']
    e321 = eps['eps321']
    e213 = eps['eps213']
    e132 = eps['eps132']

    f1 = -x1**3 + x1 + c1 + e123 * x2 * x3 + e132 * x3 * x2
    f2 = -x2**3 + x2 + c2 + e231 * x3 * x1 + e213 * x1 * x3
    f3 = -x3**3 + x3 + c3 + e312 * x1 * x2 + e321 * x2 * x1

    f = np.array([f1, f2, f3], dtype=float)

    # 雅可比：
    # ∂(-x_i^3 + x_i)/∂x_i = -3 x_i^2 + 1
    J = np.zeros((3,3), dtype=float)
    J[0,0] = -3.0 * x1**2 + 1.0
    # ∂f1/∂x2: from e123 * x2*x3 -> e123 * x3 ; from e132 * x3*x2 -> e132 * x3
    J[0,1] = e123 * x3 + e132 * x3
    # ∂f1/∂x3:
    J[0,2] = e123 * x2 + e132 * x2

    J[1,0] = e231 * x3 + e213 * x3
    J[1,1] = -3.0 * x2**2 + 1.0
    J[1,2] = e231 * x1 + e213 * x1

    J[2,0] = e312 * x2 + e321 * x2
    J[2,1] = e312 * x1 + e321 * x1
    J[2,2] = -3.0 * x3**2 + 1.0

    return f, J

# ---------------- Newton 求根与稳定性计数 ----------------
def find_roots_and_count_stable(eps, c1, c2, c3_fixed=C3_FIXED):
    """
    针对给定的 eps 和 (c1,c2,c3_fixed)，用多初值 Newton 搜索定点、
    去重并统计稳定平衡点数（雅可比所有特征值实部 < 0）。
    返回稳定平衡点个数（int）。
    """
    c_vec = (float(c1), float(c2), float(c3_fixed))
    roots = []

    for x10 in x1s0:
        for x20 in x2s0:
            for x30 in x3s0:
                x = np.array([x10, x20, x30], dtype=float)
                f = None
                for _ in range(NEWTON_MAX_IT):
                    f, J = f_and_J(x, c_vec, eps)
                    try:
                        dx = np.linalg.solve(J, f)
                    except np.linalg.LinAlgError:
                        dx = None
                        break
                    x = x - dx
                    if np.linalg.norm(dx) < NEWTON_ATOL:
                        break
                if f is not None and np.linalg.norm(f) < F_TOL:
                    # 去重
                    is_dup = False
                    for xr in roots:
                        if np.linalg.norm(x - xr) < ROOT_DUP_TOL:
                            is_dup = True
                            break
                    if not is_dup:
                        roots.append(x.copy())

    # 判定稳定性
    n_stable = 0
    for xr in roots:
        _, J = f_and_J(xr, c_vec, eps)
        eigs = np.linalg.eigvals(J)
        if np.all(np.real(eigs) < 0.0):
            n_stable += 1

    # 限制返回上限，便于绘色
    return int(min(n_stable, MAX_STABLE_DISPLAY))

# ---------------- 并行：按 c2 行计算 ----------------
def compute_row(i_row, eps):
    """
    计算给定 eps 下第 i_row 行（固定 c2）的 N_c1 个 (c1,c2) 点的稳定数。
    返回 (i_row, row_array)
    """
    c2_val = c2_arr[i_row]
    row = np.zeros(N_c1, dtype=np.int32)
    for j, c1_val in enumerate(c1_arr):
        row[j] = find_roots_and_count_stable(eps, c1_val, c2_val, C3_FIXED)
    return (i_row, row)

def compute_for_param_pair_2D(xval, yval, workers=None, save=True, plot=True):
    eps = make_eps_dict(xval, yval)
    label = f'{SCAN_PARAM_X}_{xval}_AND_{SCAN_PARAM_Y}_{yval}'
    print(f'Computing 2D stab_map for {label} (c3 fixed={C3_FIXED})')

    num_workers = workers if workers is not None else max(1, mp.cpu_count())
    stab_map = np.zeros((N_c2, N_c1), dtype=np.int32)

    # 并行提交 N_c2 个任务（每个任务一行）
    with mp.Pool(processes=num_workers) as pool:
        tasks = [pool.apply_async(compute_row, args=(i, eps)) for i in range(N_c2)]
        for idx, t in enumerate(tasks):
            i_row, row = t.get()
            stab_map[i_row, :] = row
            if (idx+1) % max(1, N_c2//8) == 0 or (idx+1) == N_c2:
                print(f'  rows completed: {idx+1}/{N_c2}')

    if save:
        fn = os.path.join(OUTDIR, f'stabmap_{label}_c3_{C3_FIXED}.npy')
        np.save(fn, stab_map)
        print(f'  saved {fn}')

    if plot:
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(stab_map, extent=(c1_min, c1_max, c2_min, c2_max),
                       origin='lower', cmap=cmap, norm=norm, aspect='auto')
        ax.set_title(f'{label}, c3={C3_FIXED}')
        ax.set_xlabel('c1')
        ax.set_ylabel('c2')
        cbar = fig.colorbar(im, ax=ax, boundaries=bounds, ticks=np.arange(0, MAX_STABLE_DISPLAY+1))
        cbar.set_label('number of stable equilibria (capped)')
        out_png = os.path.join(OUTDIR, f'stabmap_{label}_c3_{C3_FIXED}.png')
        fig.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f'  saved {out_png}')

    return stab_map

def make_combined_grid(all_maps, X_vals, Y_vals, outfn=None):
    nrow = len(Y_vals)
    ncol = len(X_vals)
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(3*ncol, 2.5*nrow))
    axes = np.atleast_2d(axes)
    for i_y, yv in enumerate(Y_vals):
        for i_x, xv in enumerate(X_vals):
            ax = axes[i_y, i_x]
            stab_map = all_maps.get((xv, yv))
            if stab_map is None:
                ax.text(0.5,0.5,'missing',ha='center',va='center')
                ax.axis('off')
                continue
            ax.imshow(stab_map, extent=(c1_min, c1_max, c2_min, c2_max),
                      origin='lower', cmap=cmap, norm=norm, aspect='auto')
            ax.set_title(f'{SCAN_PARAM_X}={xv}, {SCAN_PARAM_Y}={yv}', fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(right=0.92, wspace=0.4, hspace=0.6)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cbar_ax, boundaries=bounds, ticks=np.arange(0, MAX_STABLE_DISPLAY+1))
    cb.set_label('number of stable equilibria (capped)')
    if outfn is None:
        outfn = os.path.join(OUTDIR, f'combined_{SCAN_PARAM_X}_vs_{SCAN_PARAM_Y}_c3_{C3_FIXED}.png')
    fig.suptitle(f'3-system three-body cusp stability maps (c3 fixed={C3_FIXED})', fontsize=12)
    fig.tight_layout(rect=[0,0,0.92,0.96])
    plt.savefig(outfn, dpi=200)
    plt.close(fig)
    print(f'Combined figure saved to {outfn}')

# ---------------- 主流程 ----------------
def run_all_2D():
    all_maps = {}
    for xv in X_vals:
        for yv in Y_vals:
            stab_map = compute_for_param_pair_2D(xv, yv, workers=WORKERS, save=True, plot=True)
            all_maps[(xv, yv)] = stab_map
    make_combined_grid(all_maps, X_vals, Y_vals)
    return all_maps

if __name__ == '__main__':
    WORKERS = None
    # 小规模快速测试建议：减小 N_c1,N_c2, 或 NX0
    # 例如 N_c1=N_c2=80; NX0=3
    run_all_2D()
    print('Done.')