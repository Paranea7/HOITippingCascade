#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stability_matrix_parallel_all_pairs.py

对给定的 d12_vals 与 d21_vals 的所有组合（6x6），在 c1-c2 平面上并行扫描稳定平衡点数量。
输出：
 - 每个 (d12,d21) 保存一个 .npy 数据文件（stab_map）和一张 PNG 图像；
 - （可选）把所有 6x6 子图合并到一张大图。

并行策略：按 c2 的行划分任务（每个任务计算一整行）。
"""
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
from functools import partial
from matplotlib.colors import BoundaryNorm

# ---------- 参数（可调整） ----------
d12_vals = [0.0, 0.2, 0.3, 0.5, 0.7, 0.9]
d21_vals = [0.0, -0.2, -0.3, -0.5, -0.7, -0.9]

c1_min, c1_max, N_c1 = -0.8, 0.8, 200   # 调试时可设为 80-100
c2_min, c2_max, N_c2 = -0.8, 0.8, 200
c1_arr = np.linspace(c1_min, c1_max, N_c1)
c2_arr = np.linspace(c2_min, c2_max, N_c2)

NX0 = 5
x1s0 = np.linspace(-2, 2, NX0)
x2s0 = np.linspace(-2, 2, NX0)

NEWTON_MAX_IT = 50
NEWTON_ATOL = 1e-10
F_TOL = 1e-8
ROOT_DUP_TOL = 1e-2

MAX_STABLE_DISPLAY = 4
cmap = plt.get_cmap('viridis', MAX_STABLE_DISPLAY + 1)
bounds = np.arange(-0.5, MAX_STABLE_DISPLAY + 1.5, 1.0)
norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N, clip=True)

WORKERS = None  # None -> use mp.cpu_count()

# 参考线参数（可注释掉）
ccrit = 2 / (3 * np.sqrt(3))
dc = ccrit / (1/np.sqrt(3) - (-1/np.sqrt(3)))

OUTDIR = 'stability_results'
os.makedirs(OUTDIR, exist_ok=True)

# ---------- 核心函数 ----------
def fixed_points_and_stability(d21, d12, c1, c2):
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
    return int(n_stable)

def compute_row(i_row, d21, d12):
    """
    计算给定 d21,d12 下第 i_row 行（对应 c2_arr[i_row]）的 stab 值数组（长度 N_c1）。
    返回 (i_row, row_array)
    """
    c2_val = c2_arr[i_row]
    row = np.empty(N_c1, dtype=np.int32)
    for j, c1_val in enumerate(c1_arr):
        row[j] = fixed_points_and_stability(d21, d12, c1_val, c2_val)
    return (i_row, row)

def compute_stab_map_for_pair(d21, d12, workers=None, save=True, plot=True):
    """
    并行计算某一对 (d21,d12) 的 stab_map（shape N_c2 x N_c1）。
    返回 stab_map（numpy array）。
    如果 save=True 会把 stab_map 保存为 .npy 并把单张图片保存为 PNG。
    """
    num_workers = workers if workers is not None else max(1, mp.cpu_count() - 0)
    print(f'Computing stab_map for d12={d12}, d21={d21} with {num_workers} workers...')
    stab_map = np.zeros((N_c2, N_c1), dtype=np.int32)

    # 并行按行计算
    with mp.Pool(processes=num_workers) as pool:
        # prepare tasks
        tasks = [pool.apply_async(compute_row, args=(i_row, d21, d12)) for i_row in range(N_c2)]
        # collect
        for idx, task in enumerate(tasks):
            i_row, row = task.get()
            stab_map[i_row, :] = row
            if (idx+1) % max(1, N_c2//8) == 0 or (idx+1) == N_c2:
                print(f'  [{d12},{d21}] completed {idx+1}/{N_c2} rows')

    if save:
        fn_npy = os.path.join(OUTDIR, f'stabmap_d12_{d12}_d21_{d21}.npy')
        np.save(fn_npy, stab_map)
        print(f'  saved {fn_npy}')

    if plot:
        fig, ax = plt.subplots(figsize=(5,4))
        im = ax.imshow(stab_map, extent=(c1_min, c1_max, c2_min, c2_max),
                       origin='lower', cmap=cmap, norm=norm, aspect='auto')
        ax.set_title(f'd12={d12}, d21={d21}')
        ax.set_xlabel('c1'); ax.set_ylabel('c2')
        ax.axhline(dc, color='w', lw=1, ls='--'); ax.axhline(-dc, color='w', lw=1, ls='--')
        ax.axvline(dc, color='w', lw=1, ls='--'); ax.axvline(-dc, color='w', lw=1, ls='--')
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                            boundaries=bounds, ticks=np.arange(0, MAX_STABLE_DISPLAY+1))
        cbar.set_label('number of stable equilibria')
        cbar.ax.set_yticklabels([str(i) for i in range(0, MAX_STABLE_DISPLAY+1)])
        out_png = os.path.join(OUTDIR, f'stabmap_d12_{d12}_d21_{d21}.png')
        fig.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f'  saved {out_png}')

    return stab_map

def make_combined_grid(all_pairs_maps, d12_vals, d21_vals, outfn=None):
    """
    把所有 (d12,d21) 的 stab_map（字典 keyed by (d12,d21)）布局为 6x6 大图并保存。
    all_pairs_maps: dict {(d12,d21): stab_map}
    """
    nrow = len(d21_vals)
    ncol = len(d12_vals)
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(3*ncol, 2.5*nrow))
    axes = np.atleast_2d(axes)

    for i_row, d21 in enumerate(d21_vals):
        for i_col, d12 in enumerate(d12_vals):
            ax = axes[i_row, i_col]
            stab_map = all_pairs_maps.get((d12, d21))
            if stab_map is None:
                ax.text(0.5, 0.5, 'missing', ha='center', va='center')
                ax.axis('off')
                continue
            ax.imshow(stab_map, extent=(c1_min, c1_max, c2_min, c2_max),
                      origin='lower', cmap=cmap, norm=norm, aspect='auto')
            ax.set_title(f'd12={d12}, d21={d21}', fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

    # add colorbar on the right
    fig.subplots_adjust(right=0.92, wspace=0.4, hspace=0.6)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cbar_ax, boundaries=bounds, ticks=np.arange(0, MAX_STABLE_DISPLAY+1))
    cb.set_label('number of stable equilibria')
    if outfn is None:
        outfn = os.path.join(OUTDIR, 'combined_stabmap_grid.png')
    fig.suptitle('Stability maps for all (d12,d21) pairs', fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    plt.savefig(outfn, dpi=200)
    plt.close(fig)
    print(f'Combined figure saved to {outfn}')

# ---------- 主流程 ----------
def run_all_pairs(d12_vals, d21_vals, workers=None, save_each=True, plot_each=True, make_combined=True):
    all_maps = {}
    # iterate over all pairs
    for d12 in d12_vals:
        for d21 in d21_vals:
            stab_map = compute_stab_map_for_pair(d21, d12, workers=workers, save=save_each, plot=plot_each)
            all_maps[(d12, d21)] = stab_map

    if make_combined:
        make_combined_grid(all_maps, d12_vals, d21_vals)

    return all_maps

if __name__ == '__main__':
    # 在这里设置并行 worker 数量（None 表示自动使用 cpu_count）
    WORKERS = None
    # 为快速测试，把网格尺寸降小。例如 N_c1=N_c2=80
    # N_c1 = 80; N_c2 = 80  # 如果你在运行前想修改网格，请在文件顶部修改
    run_all_pairs(d12_vals, d21_vals, workers=WORKERS, save_each=True, plot_each=True, make_combined=True)