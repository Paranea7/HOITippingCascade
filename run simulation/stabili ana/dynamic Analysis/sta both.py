#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stability_3d_fullgrid.py

扩展脚本：把所有线性耦合 d_ij (i != j) 都纳入参数网格遍历，
并且允许对任意三体项 e_{i,j,k} 指定取值列表进行遍历。
对每一组参数(所有线性项 + 三体项组合)，在 c1-c2 平面做稳定性扫描（c3 固定为 0）。
并行策略：
  - 对单个参数组合（即固定所有 d_ij 与 e_ijk）内，按 c2 的行并行计算（multiprocessing.Pool）。
  - 参数组合之间按序执行以避免资源耗尽；可通过 MAX_PARALLEL_COMBINATIONS >1 来允许并行执行多组（谨慎使用）。
注意：参数组合数会迅速爆炸，请合理选择每个参数的取值列表长度。
"""

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import itertools
import os
from functools import partial
from matplotlib.colors import BoundaryNorm
import time

# -------------------------- 用户可调参数 --------------------------
# 线性耦合 d_ij 列表（i != j）。若某项不想遍历则设为 [0.0]。
# 示例：保持与原来类似的值范围
d12_list = [0.0, 0.2, 0.3, 0.5, 0.7, 0.9]
d21_list = [0.0, -0.2, -0.3, -0.5, -0.7, -0.9]
# 新增与 x3 的线性耦合（如果只想固定为 0，请设为 [0.0]）
d13_list = [0.0]   # f1 <- x3
d31_list = [0.0]   # f3 <- x1
d23_list = [0.0]   # f2 <- x3
d32_list = [0.0]   # f3 <- x2

# 若需遍历所有组合，把上面列表改为需要的值列表（但注意组合数）

# 三体参数字典：键为 (i,j,k)（1-based）, 值为列表（可只包含 0.0）
# 示例：原默认只遍历 (1,2,3) 与 (2,3,1)
EPS_dict_lists = {
    (1,2,3): [0.2, 0.5],
    (2,3,1): [-0.2, -0.5],
    # 若你想在未来加入其它三体项，例如 (1,1,2): [0.0, 0.1], 可在此添加
    # (1,1,2): [0.0],
}

# c1-c2 网格参数（可根据需要缩小用于测试）
c1_min, c1_max, N_c1 = -0.8, 0.8, 200
c2_min, c2_max, N_c2 = -0.8, 0.8, 200
c1_arr = np.linspace(c1_min, c1_max, N_c1)
c2_arr = np.linspace(c2_min, c2_max, N_c2)

# Newton 初值网格（x1,x2,x3）
NX0 = 5
x1s0 = np.linspace(-2, 2, NX0)
x2s0 = np.linspace(-2, 2, NX0)
x3s0 = np.linspace(-2, 2, NX0)

# Newton / 数值参数
NEWTON_MAX_IT = 80
NEWTON_ATOL = 1e-10
F_TOL = 1e-8
ROOT_DUP_TOL = 1e-2

# 绘图与输出
MAX_STABLE_DISPLAY = 4
cmap = plt.get_cmap('viridis', MAX_STABLE_DISPLAY + 1)
bounds = np.arange(-0.5, MAX_STABLE_DISPLAY + 1.5, 1.0)
norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N, clip=True)

OUTDIR = 'stability_results_3d_fullgrid'
os.makedirs(OUTDIR, exist_ok=True)

# 并行与资源控制
WORKERS = None  # None -> mp.cpu_count()
# 同时并行运行的参数组合上限（每个组合内部又会使用 WORKERS 个进程）
MAX_PARALLEL_COMBINATIONS = 1  # 设为 1 表示参数组合按序执行；>1 则并行运行多个组合（谨慎）
# ---------------------------------------------------------------

# 参考线（可注释）
ccrit = 2 / (3 * np.sqrt(3))
dc = ccrit / (1/np.sqrt(3) - (-1/np.sqrt(3)))

# -------------------------- 工具函数 --------------------------
def make_param_grid(d_lists, eps_lists):
    """
    将线性耦合字典与三体 eps 列表组合成参数组合生成器。
    d_lists: dict with keys ('d12','d21','d13','d31','d23','d32') -> list of floats
    eps_lists: dict with keys (i,j,k) -> list of floats
    返回生成器，yield (d_params_dict, eps_dict) 每次为一个具体组合
    """
    # 线性参数迭代顺序固定
    d_keys = ['d12','d21','d13','d31','d23','d32']
    d_values_list = [d_lists[k] for k in d_keys]
    # 三体参数列表
    eps_keys = list(eps_lists.keys())
    eps_values_list = [eps_lists[k] for k in eps_keys]

    for d_vals in itertools.product(*d_values_list):
        d_params = dict(zip(d_keys, d_vals))
        for eps_vals in itertools.product(*eps_values_list):
            eps_dict = { eps_keys[i]: eps_vals[i] for i in range(len(eps_keys)) }
            yield d_params, eps_dict

def three_body_contributions(x, eps_dict):
    """
    计算 tb 和 d_tb，参见之前脚本注释
    """
    x1, x2, x3 = x
    tb = np.zeros(3, dtype=float)
    d_tb = np.zeros((3,3), dtype=float)
    for (i,j,k), val in eps_dict.items():
        if val == 0.0:
            continue
        xj = (x1, x2, x3)[j-1]
        xk = (x1, x2, x3)[k-1]
        tb[i-1] += val * xj * xk
        for m_idx, m in enumerate((1,2,3)):
            contrib = 0.0
            if m == j:
                contrib += val * xk
            if m == k:
                contrib += val * xj
            d_tb[i-1, m_idx] += contrib
    return tb, d_tb

def fixed_points_and_stability_3d(d_params, c1, c2, eps_dict, c3_fixed=0.0):
    """
    d_params: dict with keys 'd12','d21','d13','d31','d23','d32'
    eps_dict: dict {(i,j,k): value}
    返回：稳定平衡数量（int）
    """
    d12 = d_params['d12']
    d21 = d_params['d21']
    d13 = d_params['d13']
    d31 = d_params['d31']
    d23 = d_params['d23']
    d32 = d_params['d32']

    roots = []
    for x10 in x1s0:
        for x20 in x2s0:
            for x30 in x3s0:
                x = np.array([x10, x20, x30], dtype=float)
                converged = False
                for _ in range(NEWTON_MAX_IT):
                    tb, d_tb = three_body_contributions(x, eps_dict)
                    f = np.zeros(3, dtype=float)
                    # 完整线性耦合写出（包含与 x3 的项）
                    f[0] = -x[0]**3 + x[0] + c1 + d21 * x[1] + d31 * x[2] + tb[0]
                    f[1] = -x[1]**3 + x[1] + c2 + d12 * x[0] + d32 * x[2] + tb[1]
                    f[2] = -x[2]**3 + x[2] + c3_fixed + d13 * x[0] + d23 * x[1] + tb[2]

                    # 雅可比（包含线性耦合与三体导数）
                    J = np.zeros((3,3), dtype=float)
                    J[0,0] = -3 * x[0]**2 + 1 + d_tb[0,0]
                    J[1,1] = -3 * x[1]**2 + 1 + d_tb[1,1]
                    J[2,2] = -3 * x[2]**2 + 1 + d_tb[2,2]

                    J[0,1] = d21 + d_tb[0,1]
                    J[0,2] = d31 + d_tb[0,2]
                    J[1,0] = d12 + d_tb[1,0]
                    J[1,2] = d32 + d_tb[1,2]
                    J[2,0] = d13 + d_tb[2,0]
                    J[2,1] = d23 + d_tb[2,1]

                    try:
                        dx = np.linalg.solve(J, f)
                    except np.linalg.LinAlgError:
                        break
                    x_new = x - dx
                    if np.linalg.norm(dx) < NEWTON_ATOL:
                        x = x_new
                        converged = True
                        break
                    x = x_new

                if converged:
                    tb_chk, _ = three_body_contributions(x, eps_dict)
                    f_chk = np.array([
                        -x[0]**3 + x[0] + c1 + d21 * x[1] + d31 * x[2] + tb_chk[0],
                        -x[1]**3 + x[1] + c2 + d12 * x[0] + d32 * x[2] + tb_chk[1],
                        -x[2]**3 + x[2] + c3_fixed + d13 * x[0] + d23 * x[1] + tb_chk[2]
                    ])
                    if np.linalg.norm(f_chk) < F_TOL:
                        duplicate = False
                        for xr in roots:
                            if np.linalg.norm(x - xr) < ROOT_DUP_TOL:
                                duplicate = True
                                break
                        if not duplicate:
                            roots.append(x.copy())

    # 统计稳定根
    n_stable = 0
    for x in roots:
        tb, d_tb = three_body_contributions(x, eps_dict)
        J = np.zeros((3,3), dtype=float)
        J[0,0] = -3 * x[0]**2 + 1 + d_tb[0,0]
        J[1,1] = -3 * x[1]**2 + 1 + d_tb[1,1]
        J[2,2] = -3 * x[2]**2 + 1 + d_tb[2,2]

        J[0,1] = d21 + d_tb[0,1]
        J[0,2] = d31 + d_tb[0,2]
        J[1,0] = d12 + d_tb[1,0]
        J[1,2] = d32 + d_tb[1,2]
        J[2,0] = d13 + d_tb[2,0]
        J[2,1] = d23 + d_tb[2,1]

        eigs = np.linalg.eigvals(J)
        if np.all(np.real(eigs) < 0):
            n_stable += 1
    return int(n_stable)

# -------------------------- 并行按行计算 --------------------------
def compute_row(i_row, d_params, eps_dict):
    c2_val = c2_arr[i_row]
    row = np.empty(N_c1, dtype=np.int32)
    for j, c1_val in enumerate(c1_arr):
        row[j] = fixed_points_and_stability_3d(d_params, c1_val, c2_val, eps_dict)
    return (i_row, row)

def compute_stab_map_for_params(d_params, eps_dict, workers=None, save=True, plot=True):
    """
    对单个参数组合（d_params, eps_dict）计算整个 c1-c2 stab_map。
    返回 stab_map (N_c2, N_c1)
    """
    num_workers = workers if workers is not None else max(1, mp.cpu_count() - 0)
    print(f'Computing stab_map for params: d={d_params}, eps={eps_dict} using {num_workers} workers')
    stab_map = np.zeros((N_c2, N_c1), dtype=np.int32)

    with mp.Pool(processes=num_workers) as pool:
        tasks = [pool.apply_async(compute_row, args=(i_row, d_params, eps_dict)) for i_row in range(N_c2)]
        for idx, task in enumerate(tasks):
            i_row, row = task.get()
            stab_map[i_row, :] = row
            if (idx+1) % max(1, N_c2//8) == 0 or (idx+1) == N_c2:
                print(f'  row progress: {idx+1}/{N_c2}')

    if save:
        # construct tag
        d_tag = '_'.join([f'{k}{d_params[k]:+.3f}' for k in sorted(d_params.keys())])
        eps_tag = '__'.join([f'e{"".join(map(str,k))}_{v:+.3f}' for k,v in sorted(eps_dict.items())])
        tag = f'{d_tag}__{eps_tag}'
        fn_npy = os.path.join(OUTDIR, f'stabmap__{tag}.npy')
        np.save(fn_npy, stab_map)
        print(f'  saved {fn_npy}')

    if plot:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.imshow(stab_map, extent=(c1_min, c1_max, c2_min, c2_max),
                  origin='lower', cmap=cmap, norm=norm, aspect='auto')
        ax.set_title(tag, fontsize=8)
        ax.set_xlabel('c1'); ax.set_ylabel('c2')
        ax.axhline(dc, color='w', lw=1, ls='--'); ax.axhline(-dc, color='w', lw=1, ls='--')
        ax.axvline(dc, color='w', lw=1, ls='--'); ax.axvline(-dc, color='w', lw=1, ls='--')
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                            boundaries=bounds, ticks=np.arange(0, MAX_STABLE_DISPLAY+1))
        cbar.set_label('number of stable equilibria')
        out_png = os.path.join(OUTDIR, f'stabmap__{tag}.png')
        fig.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f'  saved {out_png}')

    return stab_map

# -------------------------- 主执行函数 --------------------------
def run_all_parameter_combinations(d_lists_dict, eps_lists_dict, workers=None, max_parallel_combinations=1):
    """
    d_lists_dict: dict of lists for d12..d32
    eps_lists_dict: dict of lists for e_{i,j,k}
    max_parallel_combinations: 同时对多个参数组合并行执行的上限（每个组合内部又使用 workers）
    """
    # 生成参数组合列表（注意可能非常大）
    combos = list(make_param_grid(d_lists_dict, eps_lists_dict))
    print(f'Number of parameter combinations to process: {len(combos)}')
    if len(combos) == 0:
        return {}

    all_maps = {}
    # 如果允许并行执行多个参数组合，使用 Pool 管理，否则顺序执行
    if max_parallel_combinations <= 1:
        # 顺序逐个执行（每个内部使用 pool 并行 c2 行）
        for idx, (d_params, eps_dict) in enumerate(combos):
            print(f'=== Processing combo {idx+1}/{len(combos)} ===')
            stab_map = compute_stab_map_for_params(d_params, eps_dict, workers=workers, save=True, plot=True)
            all_maps[(tuple(sorted(d_params.items())), tuple(sorted(eps_dict.items())))] = stab_map
    else:
        # 并行执行多个参数组合，但要注意资源：每个组合内部又会占用 workers 进程
        # 我们采用进程池提交 compute_stab_map_for_params 的调用（每个调用会再启动进程池 -> 可引发嵌套池问题）
        # 为避免嵌套 Pool 的复杂性，建议将 compute_stab_map_for_params 改写为非嵌套（如在子进程内直接做顺序计算）
        raise NotImplementedError("Parallel execution across parameter combinations is not implemented due to nested-pool complexity. "
                                  "Set MAX_PARALLEL_COMBINATIONS=1 (default) or request implementation using joblib/dask.")

    return all_maps

# -------------------------- 运行脚本 --------------------------
if __name__ == '__main__':
    start_time = time.time()

    # 组织线性参数字典供遍历
    d_lists_dict = {
        'd12': d12_list,
        'd21': d21_list,
        'd13': d13_list,
        'd31': d31_list,
        'd23': d23_list,
        'd32': d32_list
    }

    # 三体参数列表字典（EPS_dict_lists 已在顶部定义）
    eps_lists_dict = EPS_dict_lists

    # 检查预计组合数量，给出提示
    n_d_comb = np.prod([len(v) for v in d_lists_dict.values()])
    n_eps_comb = np.prod([len(v) for v in eps_lists_dict.values()]) if len(eps_lists_dict)>0 else 1
    print(f'Will iterate over {n_d_comb} linear-coupling combos and {n_eps_comb} eps combos -> total {n_d_comb*n_eps_comb} parameter combos')

    # 强烈建议测试时把 N_c1,N_c2,NX0 调小以便快速反馈

    all_maps = run_all_parameter_combinations(d_lists_dict, eps_lists_dict, workers=WORKERS, max_parallel_combinations=MAX_PARALLEL_COMBINATIONS)

    elapsed = time.time() - start_time
    print(f'All done. Elapsed time: {elapsed:.1f} s')