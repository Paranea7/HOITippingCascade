
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import itertools
import os

# ---------- 参数（可根据需要调整以加速/提高精度） ----------
Ngrid = 200        # d12/d21 方向网格分辨率（越大计算越慢）
dlim = 0.9         # d12, d21 的范围为 [-dlim, dlim]
d21_arr = np.linspace(-dlim, dlim, Ngrid)
d12_arr = np.linspace(-dlim, dlim, Ngrid)

# 并行 worker 数（None => 使用 cpu_count）
WORKERS = None  # e.g. set to 8 to force 8 workers, or None to use os.cpu_count()

# Newton 初值网格（用于寻找所有根）
NX0 = 5
x1s0 = np.linspace(-2, 2, NX0)
x2s0 = np.linspace(-2, 2, NX0)

# Newton 迭代参数
NEWTON_MAX_IT = 50
NEWTON_ATOL = 1e-10
F_TOL = 1e-8
ROOT_DUP_TOL = 1e-2

# 用于绘图的 colormap 等
# 使用 matplotlib 的离散颜色和边界（0..4）
cmap = plt.get_cmap('viridis', 5)   # 5 个离散颜色

# 单元临界值（与原脚本一致）
ccrit = 2 / (3 * np.sqrt(3))
dc = ccrit / (1/np.sqrt(3) - (-1/np.sqrt(3)))  # 约 0.192

# 要扫描的 c1, c2 值（你给的是 5x5 网格）
c1s = [0.0, 0.2, 0.4, 0.6, 0.8]
c2s = [0.0, 0.2, 0.4, 0.6, 0.8]

# 最大可识别的稳定数（用于 colorbar 范围）
MAX_STABLE_DISPLAY = 4  # 假设稳定平衡数在 0..4 中；如超出，可增大此值并更新 cmap


# ---------- 函数定义 ----------
def fixed_points_and_stability(d21, d12, c1, c2):
    """
    对给定 (d21, d12, c1, c2)：
    - 在 [-2,2]^2 的若干初值上用 Newton 迭代找到所有定点
    - 判断稳定点数量并返回
    """
    roots = []

    for x10 in x1s0:
        for x20 in x2s0:
            x1, x2 = x10, x20
            f1 = f2 = None
            for _ in range(NEWTON_MAX_IT):
                f1 = -x1**3 + x1 + c1 + d21 * x2
                f2 = -x2**3 + x2 + c2 + d12 * x1
                J11 = -3 * x1**2 + 1
                J12 = d21
                J21 = d12
                J22 = -3 * x2**2 + 1
                detJ = J11 * J22 - J12 * J21
                if abs(detJ) < 1e-14:
                    break
                dx1 = ( J22 * f1 - J12 * f2) / detJ
                dx2 = (-J21 * f1 + J11 * f2) / detJ
                x1 -= dx1
                x2 -= dx2
                if np.hypot(dx1, dx2) < NEWTON_ATOL:
                    break

            if f1 is not None and np.hypot(f1, f2) < F_TOL:
                duplicate = False
                for xr in roots:
                    if np.hypot(x1 - xr[0], x2 - xr[1]) < ROOT_DUP_TOL:
                        duplicate = True
                        break
                if not duplicate:
                    roots.append([x1, x2])

    n_stable = 0
    for x1, x2 in roots:
        J11 = -3 * x1**2 + 1
        J12 = d21
        J21 = d12
        J22 = -3 * x2**2 + 1
        J = np.array([[J11, J12], [J21, J22]])
        eigs = np.linalg.eigvals(J)
        if np.all(np.real(eigs) < 0):
            n_stable += 1

    # 限制在 colorbar 范围内（若超出，仍返回真实值，但绘图时会被截断）
    return int(n_stable)


def compute_row_for_params(i_d21, d21, c1, c2):
    """
    计算给定 d21 行（即固定 d21），在所有 d12_arr 上的稳定点数量。
    返回：一维 numpy 数组长度为 Ngrid，对应该行的 stab_map 行。
    该函数设计为可在进程池中并行调用。
    """
    row = np.zeros(len(d12_arr), dtype=int)
    for j, d12 in enumerate(d12_arr):
        row[j] = fixed_points_and_stability(d21, d12, c1, c2)
    return (i_d21, row)


# ---------- 主循环与并行绘图 ----------
def main():
    num_workers = WORKERS if WORKERS is not None else max(1, mp.cpu_count() - 0)
    print(f'Using {num_workers} worker processes for parallel computation.')

    # 画布：nrow x ncol 子图
    nrow = len(c2s)
    ncol = len(c1s)
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(4*ncol, 3.5*nrow))
    # 规范 axes 为 2D 数组
    axes = np.atleast_2d(axes)

    # 为 colorbar 设置离散的边界：例如 0..MAX_STABLE_DISPLAY 每个整数一个颜色区间
    from matplotlib.colors import BoundaryNorm
    bounds = np.arange(-0.5, MAX_STABLE_DISPLAY + 1.5, 1.0)  # e.g. [-0.5,0.5,1.5,...,4.5]
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N, clip=True)

    # Pool 在每个 (c1,c2) 下重建，以避免子进程持有不必要的数据（也可以复用）
    for row_idx, c2 in enumerate(c2s):
        for col_idx, c1 in enumerate(c1s):
            print(f'\nComputing for c1={c1}, c2={c2}  (subplot row {row_idx}, col {col_idx})')

            stab_map = np.zeros((Ngrid, Ngrid), dtype=int)

            # 使用进程池并行计算每一行
            with mp.Pool(processes=num_workers) as pool:
                # 为每一行提交任务：传递行索引以便返回可放回正确位置
                tasks = [pool.apply_async(compute_row_for_params, args=(i, d21, c1, c2))
                         for i, d21 in enumerate(d21_arr)]

                # 等待结果并放回 stab_map
                for t_idx, task in enumerate(tasks):
                    i_row, row_data = task.get()
                    stab_map[i_row, :] = row_data
                    if (t_idx + 1) % max(1, len(tasks)//4) == 0:
                        print(f'  {t_idx+1}/{len(tasks)} rows completed for (c1={c1}, c2={c2})')

            # 绘图（每个子图）
            ax = axes[row_idx, col_idx]
            im = ax.imshow(stab_map, extent=(-dlim, dlim, -dlim, dlim),
                           origin='lower', cmap=cmap, norm=norm, aspect='auto')
            # 画单元临界线
            ax.axhline( dc, color='w', lw=1, ls='--')
            ax.axhline(-dc, color='w', lw=1, ls='--')
            ax.axvline( dc, color='w', lw=1, ls='--')
            ax.axvline(-dc, color='w', lw=1, ls='--')

            ax.set_xlabel(r'$d_{12}$')
            ax.set_ylabel(r'$d_{21}$')
            ax.set_title(f'c1={c1}, c2={c2}')

    # 统一 colorbar（在右侧）
    fig.subplots_adjust(right=0.88, wspace=0.35, hspace=0.45)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cbar_ax, boundaries=bounds, ticks=np.arange(0, MAX_STABLE_DISPLAY+1))
    cb.set_label('number of stable equilibria')
    # 将刻度标签放在颜色区间中心（BoundaryNorm 已经处理位置，ticks 为整数）
    cb.ax.set_yticklabels([str(i) for i in range(0, MAX_STABLE_DISPLAY+1)])

    plt.suptitle('Stability matrix for various (c1,c2) (parallelized)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    outfn = 'stability_matrix_c_grid_parallel.png'
    plt.savefig(outfn, dpi=300)
    print(f'Saved figure to {outfn}')
    plt.show()


if __name__ == '__main__':
    main()