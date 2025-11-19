#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing as mp
import os
from numba import njit, prange

# -----------------------------------------
# 扫描参数（三体耦合）
# -----------------------------------------
e123_vals = [0.0, 0.2, 0.3, 0.5, 0.7, 0.9]
e231_vals = [0.0, -0.2, -0.3, -0.5, -0.7, -0.9]

# -----------------------------------------
# 固定 c3
# -----------------------------------------
c3_fixed = 0.4

# 扫描 c1, c2 范围
c1_min, c1_max, N_c1 = -0.8, 0.8, 200
c2_min, c2_max, N_c2 = -0.8, 0.8, 200

c1_arr = np.linspace(c1_min, c1_max, N_c1)
c2_arr = np.linspace(c2_min, c2_max, N_c2)

# 初始点网格
NX0 = 8
x10 = np.linspace(-2, 2, NX0)
x20 = np.linspace(-2, 2, NX0)
x30 = np.linspace(-2, 2, NX0)

NEWTON_MAX_IT = 50
NEWTON_ATOL = 1e-10
F_TOL = 1e-8
ROOT_DUP_BASE = 1e-2
ROOT_DUP_MIN = 1e-6
STAB_EPS = 1e-6  # 稳定性判定的 real(λ) < -eps

OUTDIR = "stability_results_3sys_threebody"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------------------
# 初始点矩阵 X0
# -----------------------------------------
X0 = np.zeros((NX0 * NX0 * NX0, 3))
idx = 0
for a in x10:
    for b in x20:
        for c in x30:
            X0[idx, :] = (a, b, c)
            idx += 1


# -----------------------------------------
# Newton 求解器（并行 numba）
# -----------------------------------------
@njit(parallel=True, fastmath=True)
def newton_all(e231, e123, c1, c2, c3, X0):
    M = X0.shape[0]
    roots = np.zeros((M, 3))
    good = np.zeros(M, dtype=np.uint8)

    for i in prange(M):
        x1 = X0[i, 0]
        x2 = X0[i, 1]
        x3 = X0[i, 2]

        for _ in range(NEWTON_MAX_IT):

            # ----------------------------
            # 三体耦合动力学（已修正 f2、f3）
            # ----------------------------
            f1 = -x1*x1*x1 + x1 + c1 + e123 * x2 * x3
            f2 = -x2*x2*x2 + x2 + c2 + e231 * x3 * x1
            f3 = -x3*x3*x3 + x3 + c3

            # Jacobian
            J11 = 1 - 3*x1*x1
            J22 = 1 - 3*x2*x2
            J33 = 1 - 3*x3*x3

            J = np.array([
                [J11,          e123 * x3,       e123 * x2],
                [e231 * x3,    J22,             e231 * x1],
                [0.0,          0.0,             J33]
            ])

            det = np.linalg.det(J)
            if abs(det) < 1e-14:
                break

            dx = np.linalg.solve(J, np.array([f1, f2, f3]))

            x1 -= dx[0]
            x2 -= dx[1]
            x3 -= dx[2]

            if dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2] < NEWTON_ATOL:
                break

        # 计算残差
        f1 = -x1*x1*x1 + x1 + c1 + e123 * x2 * x3
        f2 = -x2*x2*x2 + x2 + c2 + e231 * x3 * x1
        f3 = -x3*x3*x3 + x3 + c3

        if max(abs(f1), abs(f2), abs(f3)) < F_TOL:
            good[i] = 1
            roots[i, 0] = x1
            roots[i, 1] = x2
            roots[i, 2] = x3

    return roots, good


# -----------------------------------------
# 自适应去重
# -----------------------------------------
@njit
def adaptive_tol(x1, x2, x3):
    scale = (abs(x1) + abs(x2) + abs(x3)) / 3.0
    return max(ROOT_DUP_MIN, ROOT_DUP_BASE * scale)


@njit
def unique_roots(roots, good):
    M = roots.shape[0]
    uniq = np.zeros((M, 3))
    k = 0

    for i in range(M):
        if good[i] == 0:
            continue

        xi1 = roots[i, 0]
        xi2 = roots[i, 1]
        xi3 = roots[i, 2]

        tol = adaptive_tol(xi1, xi2, xi3)
        tol2 = tol * tol

        duplicate = False
        for j in range(k):
            dx = xi1 - uniq[j, 0]
            dy = xi2 - uniq[j, 1]
            dz = xi3 - uniq[j, 2]
            if dx*dx + dy*dy + dz*dz < tol2:
                duplicate = True
                break

        if not duplicate:
            uniq[k, 0] = xi1
            uniq[k, 1] = xi2
            uniq[k, 2] = xi3
            k += 1

    return uniq[:k]


# -----------------------------------------
# 稳定性判定
# -----------------------------------------
def count_stable(e231, e123, c1, c2, c3):
    roots, good = newton_all(e231, e123, c1, c2, c3, X0)
    uniq = unique_roots(roots, good)

    n_stable = 0

    for x1, x2, x3 in uniq:

        J = np.array([
            [1 - 3*x1*x1,      e123 * x3,        e123 * x2],
            [e231 * x3,        1 - 3*x2*x2,      e231 * x1],
            [0.0,              0.0,              1 - 3*x3*x3]
        ])

        eigs = np.linalg.eigvals(J)
        if np.all(np.real(eigs) < -STAB_EPS):
            n_stable += 1

    return n_stable


# -----------------------------------------
# 单行计算
# -----------------------------------------
def compute_row(i_row, e231, e123, c3):
    c2_val = c2_arr[i_row]
    row = np.zeros(N_c1, dtype=np.int32)

    for j, c1_val in enumerate(c1_arr):
        row[j] = count_stable(e231, e123, c1_val, c2_val, c3)

    return (i_row, row)


# -----------------------------------------
# 全局扫描
# -----------------------------------------
def compute_stab_map(e231, e123, c3, workers=None):
    num_workers = workers or max(1, mp.cpu_count())
    stab_map = np.zeros((N_c2, N_c1), dtype=np.int32)

    with mp.Pool(num_workers) as pool:
        tasks = [
            pool.apply_async(compute_row, args=(i, e231, e123, c3))
            for i in range(N_c2)
        ]

        for idx, t in enumerate(tasks):
            i_row, row = t.get()
            stab_map[i_row] = row
            if (idx + 1) % max(1, N_c2 // 8) == 0:
                print(f"  row {idx + 1}/{N_c2}")

    return stab_map


# -----------------------------------------
# 主程序
# -----------------------------------------
def run_all():

    c3 = c3_fixed

    for e123 in e123_vals:
        for e231 in e231_vals:

            print(f"Computing e123={e123}, e231={e231}, c3={c3} ...")

            stab_map = compute_stab_map(e231, e123, c3)

            fn = os.path.join(OUTDIR, f"stabmap_e123_{e123}_e231_{e231}.npy")
            np.save(fn, stab_map)

            print(f"Saved {fn}")


if __name__ == "__main__":
    run_all()