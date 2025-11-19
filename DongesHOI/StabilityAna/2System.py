#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing as mp
import os

# 参数
d12_vals = [0.0, 0.2, 0.3, 0.5, 0.7, 0.9]
d21_vals = [0.0, -0.2, -0.3, -0.5, -0.7, -0.9]

c1_min, c1_max, N_c1 = -0.8, 0.8, 200
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

OUTDIR = "stability_results"
os.makedirs(OUTDIR, exist_ok=True)

def fixed_points_and_stability(d21, d12, c1, c2):
    roots = []
    for x10 in x1s0:
        for x20 in x2s0:
            x1, x2 = x10, x20
            f1 = f2 = None
            for _ in range(NEWTON_MAX_IT):
                f1 = -x1**3 + x1 + c1 + d21*x2
                f2 = -x2**3 + x2 + c2 + d12*x1
                J11 = -3*x1**2 + 1
                J12 = d21
                J21 = d12
                J22 = -3*x2**2 + 1
                detJ = J11*J22 - J12*J21
                if abs(detJ) < 1e-14:
                    break
                dx1 = (J22*f1 - J12*f2) / detJ
                dx2 = (-J21*f1 + J11*f2) / detJ
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
        J11 = -3*x1**2 + 1
        J12 = d21
        J21 = d12
        J22 = -3*x2**2 + 1
        J = np.array([[J11, J12], [J21, J22]])
        eigs = np.linalg.eigvals(J)
        if np.all(np.real(eigs) < 0):
            n_stable += 1

    return int(n_stable)


def compute_row(i_row, d21, d12):
    c2_val = c2_arr[i_row]
    row = np.empty(N_c1, dtype=np.int32)
    for j, c1_val in enumerate(c1_arr):
        row[j] = fixed_points_and_stability(d21, d12, c1_val, c2_val)
    return (i_row, row)


def compute_stab_map(d21, d12, workers=None):
    num_workers = workers or max(1, mp.cpu_count())
    stab_map = np.zeros((N_c2, N_c1), dtype=np.int32)

    with mp.Pool(num_workers) as pool:
        tasks = [pool.apply_async(compute_row, args=(i, d21, d12))
                 for i in range(N_c2)]

        for idx, t in enumerate(tasks):
            i_row, row = t.get()
            stab_map[i_row, :] = row
            if (idx+1) % max(1, N_c2//8) == 0:
                print(f"  row {idx+1}/{N_c2}")

    return stab_map


def run_all():
    for d12 in d12_vals:
        for d21 in d21_vals:
            print(f"Computing d12={d12}, d21={d21} ...")
            sm = compute_stab_map(d21, d12)
            fn = os.path.join(OUTDIR, f"stabmap_d12_{d12}_d21_{d21}.npy")
            np.save(fn, sm)
            print(f"Saved {fn}")


if __name__ == "__main__":
    run_all()