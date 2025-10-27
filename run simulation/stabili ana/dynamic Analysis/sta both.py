#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stability_with_3_overlay_fixed.py

完整脚本：以二体为主的稳定性绘图程序，并在每个二体子图上叠加三体影响（轮廓与可选半透明掩膜）。
修正点：
 - 正确读取三体参数向量中的 c3（避免 c3 未定义导致的错误）
 - 保证 meshgrid 顺序与 stab arrays 对应
 - 增加诊断输出（stab3 vs stab2 差异计数）
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from matplotlib.colors import BoundaryNorm

# ---------------- Parameters (可调) ----------------
OUTDIR = 'stability_with_3_overlay_results'
os.makedirs(OUTDIR, exist_ok=True)

d12_vals = [0.0, 0.2, 0.3, 0.5, 0.7, 0.9]
d21_vals = [0.0, -0.2, -0.3, -0.5, -0.7, -0.9]

# 主网格（用于二体主图，也用于三体比较以便直接覆盖）
c1_min, c1_max, N_c1 = -0.8, 0.8, 200
c2_min, c2_max, N_c2 = -0.8, 0.8, 200
c1_arr = np.linspace(c1_min, c1_max, N_c1)
c2_arr = np.linspace(c2_min, c2_max, N_c2)

# Newton 初值（2-system）
NX0_2 = 5
x1s0_2 = np.linspace(-2, 2, NX0_2)
x2s0_2 = np.linspace(-2, 2, NX0_2)

NEWTON_MAX_IT_2 = 50
NEWTON_ATOL_2 = 1e-10
F_TOL_2 = 1e-8
ROOT_DUP_TOL_2 = 1e-2

MAX_STABLE_DISPLAY_2 = 4
cmap2 = plt.get_cmap('viridis', MAX_STABLE_DISPLAY_2 + 1)
bounds2 = np.arange(-0.5, MAX_STABLE_DISPLAY_2 + 1.5, 1.0)
norm2 = BoundaryNorm(boundaries=bounds2, ncolors=cmap2.N, clip=True)

# 三体设置（固定 eps）
EPS_3 = {'eps123': 0.2, 'eps231': -0.2, 'eps312': 0.0, 'eps321': 0.0, 'eps213': 0.0, 'eps132': 0.0}
C3_FIXED = 0.0

# Newton 初值（3-system）
NX0_3 = 5
x1s0_3 = np.linspace(-2, 2, NX0_3)
x2s0_3 = np.linspace(-2, 2, NX0_3)
x3s0_3 = np.linspace(-2, 2, NX0_3)

NEWTON_MAX_IT_3 = 80
NEWTON_ATOL_3 = 1e-10
F_TOL_3 = 1e-8
ROOT_DUP_TOL_3 = 1e-4

MAX_STABLE_DISPLAY_3 = 6
cmap3 = plt.get_cmap('viridis', MAX_STABLE_DISPLAY_3 + 1)
bounds3 = np.arange(-0.5, MAX_STABLE_DISPLAY_3 + 1.5, 1.0)
norm3 = BoundaryNorm(boundaries=bounds3, ncolors=cmap3.N, clip=True)

# ---------------- Core functions ----------------
def fixed_points_and_stability_2sys(d21, d12, c1, c2,
                                   x1s0, x2s0,
                                   NEWTON_MAX_IT, NEWTON_ATOL, F_TOL, ROOT_DUP_TOL):
    """二体：用多初值 Newton 找定点并计数稳定个数（不截断）"""
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

def f_and_J_3(x, c, eps):
    """三体 f 与雅可比（修正：正确读取 c3）"""
    x1, x2, x3 = x
    c1, c2, c3 = c
    e123 = eps.get('eps123', 0.0); e231 = eps.get('eps231', 0.0); e312 = eps.get('eps312', 0.0)
    e321 = eps.get('eps321', 0.0); e213 = eps.get('eps213', 0.0); e132 = eps.get('eps132', 0.0)

    f1 = -x1**3 + x1 + c1 + e123 * x2 * x3 + e132 * x3 * x2
    f2 = -x2**3 + x2 + c2 + e231 * x3 * x1 + e213 * x1 * x3
    f3 = -x3**3 + x3 + c3 + e312 * x1 * x2 + e321 * x2 * x1
    f = np.array([f1, f2, f3], dtype=float)

    J = np.zeros((3,3), dtype=float)
    J[0,0] = -3.0 * x1**2 + 1.0
    J[0,1] = e123 * x3 + e132 * x3
    J[0,2] = e123 * x2 + e132 * x2

    J[1,0] = e231 * x3 + e213 * x3
    J[1,1] = -3.0 * x2**2 + 1.0
    J[1,2] = e231 * x1 + e213 * x1

    J[2,0] = e312 * x2 + e321 * x2
    J[2,1] = e312 * x1 + e321 * x1
    J[2,2] = -3.0 * x3**2 + 1.0

    return f, J

def find_roots_and_count_stable_3sys(eps, c1, c2, c3_fixed,
                                     x1s0, x2s0, x3s0,
                                     NEWTON_MAX_IT, NEWTON_ATOL, F_TOL, ROOT_DUP_TOL):
    """三体：多初值 Newton，返回稳定点数量"""
    c_vec = (float(c1), float(c2), float(c3_fixed))
    roots = []
    for x10 in x1s0:
        for x20 in x2s0:
            for x30 in x3s0:
                x = np.array([x10, x20, x30], dtype=float)
                f = None
                for _ in range(NEWTON_MAX_IT):
                    f, J = f_and_J_3(x, c_vec, eps)
                    try:
                        dx = np.linalg.solve(J, f)
                    except np.linalg.LinAlgError:
                        dx = None
                        break
                    x = x - dx
                    if np.linalg.norm(dx) < NEWTON_ATOL:
                        break
                if f is not None and np.linalg.norm(f) < F_TOL:
                    is_dup = False
                    for xr in roots:
                        if np.linalg.norm(x - xr) < ROOT_DUP_TOL:
                            is_dup = True
                            break
                    if not is_dup:
                        roots.append(x.copy())

    n_stable = 0
    for xr in roots:
        _, J = f_and_J_3(xr, c_vec, eps)
        eigs = np.linalg.eigvals(J)
        if np.all(np.real(eigs) < 0.0):
            n_stable += 1
    return int(n_stable)

# ---------------- Row compute for parallel processing ----------------
def compute_row_2sys(i_row, d21, d12, c1_arr_local, c2_arr_local, params2):
    c2_val = c2_arr_local[i_row]
    row = np.empty(c1_arr_local.shape, dtype=np.int32)
    for j, c1_val in enumerate(c1_arr_local):
        n = fixed_points_and_stability_2sys(d21, d12, c1_val, c2_val,
                                            params2['x1s0'], params2['x2s0'],
                                            params2['NEWTON_MAX_IT'], params2['NEWTON_ATOL'],
                                            params2['F_TOL'], params2['ROOT_DUP_TOL'])
        row[j] = int(min(n, params2['MAX_STABLE_DISPLAY']))
    return (i_row, row)

def compute_row_3sys(i_row, eps, c1_arr_local, c2_arr_local, params3):
    c2_val = c2_arr_local[i_row]
    row = np.empty(c1_arr_local.shape, dtype=np.int32)
    for j, c1_val in enumerate(c1_arr_local):
        n = find_roots_and_count_stable_3sys(eps, c1_val, c2_val, params3['C3_FIXED'],
                                            params3['x1s0'], params3['x2s0'], params3['x3s0'],
                                            params3['NEWTON_MAX_IT'], params3['NEWTON_ATOL'],
                                            params3['F_TOL'], params3['ROOT_DUP_TOL'])
        row[j] = int(min(n, params3['MAX_STABLE_DISPLAY']))
    return (i_row, row)

# ---------------- Compute full stab map for a pair (d12,d21) ----------------
def compute_pair_with_3_overlay(d21, d12, params2, params3, workers=None, save=True, plot=True, overlay3=True, outdir=OUTDIR, overlay3_mask=True):
    """
    计算：
      - 二体 stab_map（并保存 .npy 与 PNG）
      - 三体 stab_map（并保存 .npy）
      - 如果 overlay3=True：在二体 PNG 上叠加三体轮廓与可选半透明掩膜并保存
    """
    num_workers = workers if workers is not None else max(1, mp.cpu_count() - 1)
    print(f'Processing pair d12={d12}, d21={d21} with {num_workers} workers; overlay3={overlay3}')

    c1_arr_local = np.linspace(c1_min, c1_max, N_c1)
    c2_arr_local = np.linspace(c2_min, c2_max, N_c2)

    # 1) compute 2sys
    stab2 = np.zeros((N_c2, N_c1), dtype=np.int32)
    with mp.Pool(processes=num_workers) as pool:
        tasks = [pool.apply_async(compute_row_2sys, args=(i, d21, d12, c1_arr_local, c2_arr_local, params2)) for i in range(N_c2)]
        for idx, t in enumerate(tasks):
            i_row, row = t.get()
            stab2[i_row, :] = row
            if (idx+1) % max(1, N_c2//8) == 0 or (idx+1) == N_c2:
                print(f'  [2sys {d12},{d21}] rows {idx+1}/{N_c2}')

    if save:
        fn2 = os.path.join(outdir, f'stabmap_2sys_d12_{d12}_d21_{d21}.npy')
        np.save(fn2, stab2)
        print('  saved', fn2)

    # 2) compute 3sys (on same grid) if overlay requested
    stab3 = None
    if overlay3:
        eps = EPS_3.copy()
        stab3 = np.zeros((N_c2, N_c1), dtype=np.int32)
        with mp.Pool(processes=num_workers) as pool:
            tasks = [pool.apply_async(compute_row_3sys, args=(i, eps, c1_arr_local, c2_arr_local, params3)) for i in range(N_c2)]
            for idx, t in enumerate(tasks):
                i_row, row = t.get()
                stab3[i_row, :] = row
                if (idx+1) % max(1, N_c2//8) == 0 or (idx+1) == N_c2:
                    print(f'  [3sys] rows {idx+1}/{N_c2}')

        if save:
            fn3 = os.path.join(outdir, f'stabmap_3sys_eps123_{eps["eps123"]}_eps231_{eps["eps231"]}_d12_{d12}_d21_{d21}.npy')
            np.save(fn3, stab3)
            print('  saved', fn3)

    # 3) plot 2sys (base) and overlay 3sys contours/mask
    if plot:
        fig, ax = plt.subplots(figsize=(5,4))
        im = ax.imshow(stab2, extent=(c1_min, c1_max, c2_min, c2_max),
                       origin='lower', cmap=cmap2, norm=norm2, aspect='auto')
        ax.set_title(f'2-sys d12={d12}, d21={d21}')
        ax.set_xlabel('c1'); ax.set_ylabel('c2')

        # overlay: contours showing levels of stab3 (if computed)
        if (stab3 is not None):
            # diagnostic: 确认 stab3 与 stab2 的差异
            diff = stab3.astype(int) - stab2.astype(int)
            nz = int(np.count_nonzero(diff))
            print(f'  [diag] stab3 vs stab2 diff nonzero count = {nz}, min={diff.min()}, max={diff.max()}, mean={diff.mean():.3f}')

            # contour levels — 我们绘制 stab3 的整数水平线
            levels = np.arange(0, min(params3['MAX_STABLE_DISPLAY'], MAX_STABLE_DISPLAY_3) + 1)
            # 为 contour 需要网格，注意 meshgrid(c1_arr_local, c2_arr_local) 以匹配 stab arrays shape (N_c2, N_c1)
            C1, C2 = np.meshgrid(c1_arr_local, c2_arr_local)
            cs = ax.contour(C1, C2, stab3, levels=levels, colors='k', linewidths=0.6, alpha=0.9)
            try:
                ax.clabel(cs, fmt='%d', fontsize=6)
            except Exception:
                pass

            # 半透明掩膜：标记 stab3 > stab2 区域（增加，红）和 stab3 < stab2 区域（减少，蓝）
            if overlay3_mask:
                mask_inc = diff > 0
                mask_dec = diff < 0
                if np.any(mask_inc):
                    ax.imshow(np.where(mask_inc, 1.0, np.nan), extent=(c1_min, c1_max, c2_min, c2_max),
                              origin='lower', cmap=plt.cm.Reds, alpha=0.25, vmin=0, vmax=1, aspect='auto')
                if np.any(mask_dec):
                    ax.imshow(np.where(mask_dec, 1.0, np.nan), extent=(c1_min, c1_max, c2_min, c2_max),
                              origin='lower', cmap=plt.cm.Blues, alpha=0.20, vmin=0, vmax=1, aspect='auto')

        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm2, cmap=cmap2), ax=ax,
                            boundaries=bounds2, ticks=np.arange(0, params2['MAX_STABLE_DISPLAY']+1))
        cbar.set_label('number of stable equilibria (2-sys, capped)')
        out_png = os.path.join(outdir, f'stabmap_2sys_with3overlay_d12_{d12}_d21_{d21}.png')
        fig.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close(fig)
        print('  saved', out_png)

    return stab2, stab3

# ---------------- Combined grid figure for all pairs (2sys with overlay if available) ----------------
def make_combined_grid_with_overlay(all_pairs_maps_2, all_pairs_maps_3, d12_vals, d21_vals, outfn=None, outdir=OUTDIR, overlay3_mask=True):
    nrow = len(d21_vals); ncol = len(d12_vals)
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(3*ncol, 2.5*nrow))
    axes = np.atleast_2d(axes)
    cmap = plt.get_cmap('viridis', MAX_STABLE_DISPLAY_2 + 1)
    bounds = np.arange(-0.5, MAX_STABLE_DISPLAY_2 + 1.5, 1.0)
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N, clip=True)

    for i_row, d21 in enumerate(d21_vals):
        for i_col, d12 in enumerate(d12_vals):
            ax = axes[i_row, i_col]
            stab2 = all_pairs_maps_2.get((d12, d21))
            stab3 = all_pairs_maps_3.get((d12, d21))
            if stab2 is None:
                ax.text(0.5, 0.5, 'missing', ha='center', va='center')
                ax.axis('off')
                continue
            ax.imshow(stab2, extent=(c1_min, c1_max, c2_min, c2_max),
                      origin='lower', cmap=cmap, norm=norm, aspect='auto')
            ax.set_title(f'd12={d12}, d21={d21}', fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

            if stab3 is not None:
                C1, C2 = np.meshgrid(c1_arr, c2_arr)
                levels = np.arange(0, min(params3_global['MAX_STABLE_DISPLAY'], MAX_STABLE_DISPLAY_3) + 1)
                cs = ax.contour(C1, C2, stab3, levels=levels, colors='k', linewidths=0.4, alpha=0.9)

                diff = stab3.astype(int) - stab2.astype(int)
                mask_inc = diff > 0
                mask_dec = diff < 0
                if overlay3_mask:
                    if np.any(mask_inc):
                        ax.imshow(np.where(mask_inc, 1.0, np.nan), extent=(c1_min, c1_max, c2_min, c2_max),
                                  origin='lower', cmap=plt.cm.Reds, alpha=0.18, vmin=0, vmax=1, aspect='auto')
                    if np.any(mask_dec):
                        ax.imshow(np.where(mask_dec, 1.0, np.nan), extent=(c1_min, c1_max, c2_min, c2_max),
                                  origin='lower', cmap=plt.cm.Blues, alpha=0.12, vmin=0, vmax=1, aspect='auto')

    fig.subplots_adjust(right=0.92, wspace=0.4, hspace=0.6)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cbar_ax, boundaries=bounds, ticks=np.arange(0, MAX_STABLE_DISPLAY_2+1))
    cb.set_label('number of stable equilibria (2-sys)')
    if outfn is None:
        outfn = os.path.join(outdir, 'combined_stabmap_grid_2sys_with3overlay.png')
    fig.suptitle('2-system stability maps with 3-system overlay (eps123=0.2, eps231=-0.2)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    plt.savefig(outfn, dpi=200)
    plt.close(fig)
    print('Combined figure saved to', outfn)

# ---------------- Global params holder for plotting combined (set later) ----------------
params3_global = None

# ---------------- Main runner ----------------
def main(do_overlay=True, workers=None, quick=False, outdir=OUTDIR, overlay3_mask=True):
    global params3_global
    if quick:
        Nc1 = 80; Nc2 = 80
        c1_local = np.linspace(c1_min, c1_max, Nc1)
        c2_local = np.linspace(c2_min, c2_max, Nc2)
        p2 = {
            'x1s0': np.linspace(-1.5,1.5,3),
            'x2s0': np.linspace(-1.5,1.5,3),
            'NEWTON_MAX_IT': 40,
            'NEWTON_ATOL': NEWTON_ATOL_2,
            'F_TOL': F_TOL_2,
            'ROOT_DUP_TOL': ROOT_DUP_TOL_2,
            'MAX_STABLE_DISPLAY': MAX_STABLE_DISPLAY_2
        }
        p3 = {
            'x1s0': np.linspace(-1.5,1.5,3),
            'x2s0': np.linspace(-1.5,1.5,3),
            'x3s0': np.linspace(-1.5,1.5,3),
            'NEWTON_MAX_IT': 60,
            'NEWTON_ATOL': NEWTON_ATOL_3,
            'F_TOL': F_TOL_3,
            'ROOT_DUP_TOL': ROOT_DUP_TOL_3,
            'MAX_STABLE_DISPLAY': MAX_STABLE_DISPLAY_3,
            'C3_FIXED': C3_FIXED
        }
    else:
        Nc1 = N_c1; Nc2 = N_c2
        c1_local = c1_arr; c2_local = c2_arr
        p2 = {
            'x1s0': x1s0_2,
            'x2s0': x2s0_2,
            'NEWTON_MAX_IT': NEWTON_MAX_IT_2,
            'NEWTON_ATOL': NEWTON_ATOL_2,
            'F_TOL': F_TOL_2,
            'ROOT_DUP_TOL': ROOT_DUP_TOL_2,
            'MAX_STABLE_DISPLAY': MAX_STABLE_DISPLAY_2
        }
        p3 = {
            'x1s0': x1s0_3,
            'x2s0': x2s0_3,
            'x3s0': x3s0_3,
            'NEWTON_MAX_IT': NEWTON_MAX_IT_3,
            'NEWTON_ATOL': NEWTON_ATOL_3,
            'F_TOL': F_TOL_3,
            'ROOT_DUP_TOL': ROOT_DUP_TOL_3,
            'MAX_STABLE_DISPLAY': MAX_STABLE_DISPLAY_3,
            'C3_FIXED': C3_FIXED
        }

    # set global for combined plotting
    params3_global = p3

    all_maps_2 = {}
    all_maps_3 = {}

    # iterate pairs
    for d12 in d12_vals:
        for d21 in d21_vals:
            stab2, stab3 = compute_pair_with_3_overlay(d21, d12, p2, p3, workers=workers, save=True, plot=True, overlay3=do_overlay, outdir=outdir, overlay3_mask=overlay3_mask)
            all_maps_2[(d12, d21)] = stab2
            if stab3 is not None:
                all_maps_3[(d12, d21)] = stab3

    # combined grid (2sys base with overlays from computed 3sys)
    make_combined_grid_with_overlay(all_maps_2, all_maps_3, d12_vals, d21_vals, outfn=None, outdir=outdir, overlay3_mask=overlay3_mask)

# ---------------- CLI ----------------
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Compute 2-system stability maps and overlay 3-system results (contours + optional masks).')
    p.add_argument('--no-3overlay', action='store_true', help='Do not compute/overlay 3-system results.')
    p.add_argument('--workers', type=int, default=-1, help='Number of worker processes (-1 => cpu_count()-1).')
    p.add_argument('--quick', action='store_true', help='Quick mode: smaller grids and fewer initial guesses.')
    p.add_argument('--outdir', type=str, default=OUTDIR, help='Output directory.')
    p.add_argument('--no-masks', action='store_true', help='Do not draw red/blue masks for increases/decreases; keep contours only.')
    args = p.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    if args.workers is None or args.workers < 0:
        workers = max(1, mp.cpu_count() - 1)
    elif args.workers == 0:
        workers = 1
    else:
        workers = args.workers

    # allow multiprocessing start method best-effort
    try:
        if os.name == 'posix':
            mp.set_start_method('fork', force=False)
        else:
            mp.set_start_method('spawn', force=False)
    except Exception:
        pass

    print('Starting. overlay3=', not args.no_3overlay, 'workers=', workers, 'quick=', args.quick, 'masks=', not args.no_masks)
    main(do_overlay=(not args.no_3overlay), workers=workers, quick=args.quick, outdir=outdir, overlay3_mask=(not args.no_masks))
    print('Finished. Results saved in', outdir)