#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing as mp
import os
from numba import njit, prange

# -----------------------------------------
# 参数（d12, d21 扫描）
# -----------------------------------------
d12_vals = [0.0, 0.2, 0.3, 0.5, 0.7, 0.9]
d21_vals = [0.0, -0.2, -0.3, -0.5, -0.7, -0.9]

# -----------------------------------------
# 固定三体耦合
# -----------------------------------------
e123 = 0.2
e231 = -0.2

# -----------------------------------------
# 线性耦合常数（保持原设定）
# -----------------------------------------
d31 = -0.2
d32 = 0.2
d13 = 0.0
d23 = 0.0

# 固定 c3
c3_fixed = 0.4

# 扫描
c1_min, c1_max, N_c1 = -0.8, 0.8, 200
c2_min, c2_max, N_c2 = -0.8, 0.8, 200
c1_arr = np.linspace(c1_min, c1_max, N_c1)
c2_arr = np.linspace(c2_min, c2_max, N_c2)

# 初始点
NX0 = 8
x10 = np.linspace(-2, 2, NX0)
x20 = np.linspace(-2, 2, NX0)
x30 = np.linspace(-2, 2, NX0)

NEWTON_MAX_IT = 50
NEWTON_ATOL = 1e-10
F_TOL = 1e-8
ROOT_DUP_BASE = 1e-2
ROOT_DUP_MIN = 1e-6
STAB_EPS = 1e-6

OUTDIR = "stability_results_3sys_threebody_fixE"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------------------
# 初始点堆叠成 (NX0^3, 3)
# -----------------------------------------
X0 = np.zeros((NX0 * NX0 * NX0, 3))
idx = 0
for a in x10:
    for b in x20:
        for c in x30:
            X0[idx] = (a, b, c)
            idx += 1


# -----------------------------------------
# Newton (加入三体耦合 e123,e231)
# -----------------------------------------
@njit(parallel=True, fastmath=True)
def newton_all(d21, d12, c1, c2, c3, X0):
    M = X0.shape[0]
    roots = np.zeros((M, 3))
    good = np.zeros(M, dtype=np.uint8)

    for i in prange(M):
        x1 = X0[i, 0]
        x2 = X0[i, 1]
        x3 = X0[i, 2]

        for _ in range(NEWTON_MAX_IT):

            # 三体耦合加入：e123 x2 x3, e231 x3 x1
            f1 = -x1*x1*x1 + x1 + c1 + d21*x2 + d31*x3 + e123*x2*x3
            f2 = -x2*x2*x2 + x2 + c2 + d12*x1 + d32*x3 + e231*x3*x1
            f3 = -x3*x3*x3 + x3 + c3

            J11 = 1 - 3*x1*x1
            J22 = 1 - 3*x2*x2
            J33 = 1 - 3*x3*x3

            J = np.array([
                [J11,               d21 + e123*x3,      d31 + e123*x2],
                [d12 + e231*x3,     J22,                d32 + e231*x1],
                [d13,               d23,                J33]
            ])

            det = np.linalg.det(J)
            if abs(det) < 1e-14:
                break

            dx = np.linalg.solve(J, np.array([f1, f2, f3]))
            x1 -= dx[0]
            x2 -= dx[1]
            x3 -= dx[2]

            if dx[0]**2 + dx[1]**2 + dx[2]**2 < NEWTON_ATOL:
                break

        f1 = -x1*x1*x1 + x1 + c1 + d21*x2 + d31*x3 + e123*x2*x3
        f2 = -x2*x2*x2 + x2 + c2 + d12*x1 + d32*x3 + e231*x3*x1
        f3 = -x3*x3*x3 + x3 + c3
        if max(abs(f1), abs(f2), abs(f3)) < F_TOL:
            good[i] = 1
            roots[i] = (x1, x2, x3)

    return roots, good


# -----------------------------------------
# 去重
# -----------------------------------------
@njit
def adaptive_tol(x1, x2, x3):
    scale = (abs(x1)+abs(x2)+abs(x3)) / 3.0
    return max(ROOT_DUP_MIN, ROOT_DUP_BASE * scale)


@njit
def unique_roots(roots, good):
    M = roots.shape[0]
    uniq = np.zeros((M, 3))
    k = 0
    for i in range(M):
        if good[i] == 0:
            continue
        x1,x2,x3 = roots[i]
        tol = adaptive_tol(x1,x2,x3)
        tol2 = tol*tol

        dup=False
        for j in range(k):
            dx = x1 - uniq[j,0]
            dy = x2 - uniq[j,1]
            dz = x3 - uniq[j,2]
            if dx*dx+dy*dy+dz*dz < tol2:
                dup=True
                break
        if not dup:
            uniq[k] = (x1,x2,x3)
            k += 1
    return uniq[:k]


# -----------------------------------------
# 稳定性判定
# -----------------------------------------
def count_stable(d21, d12, c1, c2, c3):
    roots, good = newton_all(d21, d12, c1, c2, c3, X0)
    uniq = unique_roots(roots, good)

    n=0
    for x1,x2,x3 in uniq:

        J = np.array([
            [1-3*x1*x1,          d21 + e123*x3,      d31 + e123*x2],
            [d12 + e231*x3,      1-3*x2*x2,          d32 + e231*x1],
            [d13,                d23,                1-3*x3*x3]
        ])

        eigs = np.linalg.eigvals(J)
        if np.all(np.real(eigs) < -STAB_EPS):
            n += 1
    return n


# -----------------------------------------
# 单行计算
# -----------------------------------------
def compute_row(i_row, d21, d12, c3):
    c2v = c2_arr[i_row]
    row = np.zeros(N_c1, dtype=np.int32)
    for j,c1v in enumerate(c1_arr):
        row[j] = count_stable(d21, d12, c1v, c2v, c3)
    return (i_row, row)


# -----------------------------------------
# stability map
# -----------------------------------------
def compute_stab_map(d21, d12, c3, workers=None):
    num_workers = workers or max(1, mp.cpu_count())
    stab_map = np.zeros((N_c2, N_c1), dtype=np.int32)

    with mp.Pool(num_workers) as pool:
        tasks = [
            pool.apply_async(compute_row, args=(i, d21, d12, c3))
            for i in range(N_c2)
        ]

        for idx,t in enumerate(tasks):
            irow,row=t.get()
            stab_map[irow]=row
            if (idx+1) % max(1, N_c2//8)==0:
                print(f"  row {idx+1}/{N_c2}")

    return stab_map


# -----------------------------------------
# 主程序：扫描 d12, d21
# -----------------------------------------
def run_all():

    c3 = c3_fixed

    for d12 in d12_vals:
        for d21 in d21_vals:

            print(f"Computing d12={d12}, d21={d21} (fixed e123={e123}, e231={e231})")

            stab_map = compute_stab_map(d21, d12, c3)

            fn = os.path.join(OUTDIR, f"stabmap_d12_{d12}_d21_{d21}.npy")
            np.save(fn, stab_map)

            print("Saved", fn)


if __name__ == "__main__":
    run_all()