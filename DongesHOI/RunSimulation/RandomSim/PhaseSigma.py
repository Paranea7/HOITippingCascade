#!/usr/bin/env python3
import os
import csv
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# ===================== numba 加速核心 =====================

@njit
def compute_dx_numba(x, c_i, d_ji, e_ijk):
    s = x.shape[0]
    dx = -x * x * x + x + c_i
    dx += d_ji @ x

    out = np.zeros(s)
    for i in range(s):
        acc = 0.0
        for j in range(s):
            xj = x[j]
            for k in range(s):
                acc += e_ijk[i, j, k] * xj * x[k]
        out[i] = acc

    dx += out
    return dx


@njit
def dynamics_simulation_numba(s, c_i, d_ji, e_ijk, x_init, t_steps):
    x = x_init.copy()
    dt = 0.01
    for _ in range(t_steps):
        k1 = compute_dx_numba(x, c_i, d_ji, e_ijk)
        k2 = compute_dx_numba(x + 0.5 * dt * k1, c_i, d_ji, e_ijk)
        k3 = compute_dx_numba(x + 0.5 * dt * k2, c_i, d_ji, e_ijk)
        k4 = compute_dx_numba(x + dt * k3, c_i, d_ji, e_ijk)
        x += (k1 + 2*k2 + 2*k3 + k4) * dt / 6.0
    return x

# ===================== Python 逻辑 =====================

def generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e):
    c_i = np.random.normal(mu_c, sigma_c, s)

    # two-body scaling
    mean_d = mu_d / s
    std_d = sigma_d / np.sqrt(s)
    d_ij = np.random.normal(mean_d, std_d, (s, s))
    d_ji = rho_d * d_ij + np.sqrt(1 - rho_d ** 2) * \
           np.random.normal(mean_d, std_d, (s, s))
    # -------- 清零 d[ii] --------
    np.fill_diagonal(d_ij, 0.0)
    np.fill_diagonal(d_ji, 0.0)
    # three-body scaling
    mean_e = mu_e / (s ** 2)
    std_e = sigma_e / s
    e_ijk = np.random.normal(mean_e, std_e, (s, s, s))

    # -------- 清零 e[i,i,i] --------
    for i in range(s):
        e_ijk[i, i, i] = 0.0

    # -------- 清零 e[i,i,k] --------
    for i in range(s):
        for k in range(s):
            e_ijk[i, i, k] = 0.0

    # -------- 清零 e[i,j,i] --------
    for i in range(s):
        for j in range(s):
            e_ijk[i, j, i] = 0.0

    return c_i, d_ij, d_ji, e_ijk


def dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps):
    return dynamics_simulation_numba(s, c_i, d_ji, e_ijk, x_init, t_steps)


def calculate_survival_rate(final_states):
    survival = np.sum(final_states > 0)
    return float(survival) / float(len(final_states))


def single_simulation_once(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps, x0=-0.6):
    c_i, _, d_ji, e_ijk = generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e)
    x_init = np.full(s, x0, float)
    final_states = dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps)
    return calculate_survival_rate(final_states)

# ===================== worker （可 pickle） =====================

def worker_task(params):
    s_local, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps, reps = params
    vals = []
    for _ in range(reps):
        vals.append(
            single_simulation_once(
                s_local, mu_c, sigma_c,
                mu_d, sigma_d, rho_d,
                mu_e, sigma_e, t_steps
            )
        )
    return float(np.mean(vals))

# ===================== 网格计算 =====================

def compute_grid(
    s,
    mu_e,
    mu_d,
    sigma_d_vals,
    sigma_e_vals,
    t_steps=3000,
    repeats=10,
    n_workers=None,
    mu_c=0.0,
    sigma_c=0.0
):

    if n_workers is None:
        n_workers = max(1, cpu_count())

    rho_d = 0.0

    tasks = []
    for sigma_e in sigma_e_vals:
        for sigma_d in sigma_d_vals:
            tasks.append(
                (
                    s, mu_c, sigma_c, mu_d,
                    float(sigma_d), rho_d, mu_e,
                    float(sigma_e), t_steps, repeats
                )
            )

    with Pool(processes=n_workers) as pool:
        results = pool.map(worker_task, tasks)

    grid = np.array(results).reshape(len(sigma_e_vals), len(sigma_d_vals))
    return sigma_d_vals, sigma_e_vals, grid

# ===================== 绘图与保存 =====================

def plot_heatmap(sigma_d_vals, sigma_e_vals, grid,
                 out_png="phase_sigma_d_sigma_e.png",
                 cmap='viridis'):
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(
        grid,
        origin='lower',
        aspect='auto',
        extent=[sigma_d_vals[0], sigma_d_vals[-1],
                sigma_e_vals[0], sigma_e_vals[-1]],
        cmap=cmap,
        vmin=0.0, vmax=1.0
    )
    ax.set_xlabel("sigma_d")
    ax.set_ylabel("sigma_e")
    ax.set_title("Survival rate")
    fig.colorbar(im, ax=ax).set_label("Survival rate")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def save_grid_csv(sigma_d_vals, sigma_e_vals, grid,
                  out_csv="phase_sigma_d_sigma_e.csv"):
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["sigma_e\\sigma_d"] + [f"{v:.6g}" for v in sigma_d_vals])
        for j, sigma_e in enumerate(sigma_e_vals):
            row = [f"{sigma_e:.6g}"] + [f"{v:.6g}" for v in grid[j, :]]
            w.writerow(row)
    return out_csv

# ===================== 主入口 =====================

def main():

    s = 50
    mu_e = 0.1
    mu_d = 0.3
    nx = 100
    ny = 100
    t_steps = 3000
    repeats = 3
    n_workers = None
    out_dir = "output_phase_Sigma"

    mu_c = 0.0
    sigma_c =2. * np.sqrt(3.)/27.

    os.makedirs(out_dir, exist_ok=True)

    sigma_d_vals = np.linspace(0, 1, nx)
    sigma_e_vals = np.linspace(0, 1, ny)

    print("Starting compute grid (parallel)...")

    sigma_d_vals, sigma_e_vals, grid = compute_grid(
        s=s,
        mu_e=mu_e,
        mu_d=mu_d,
        sigma_d_vals=sigma_d_vals,
        sigma_e_vals=sigma_e_vals,
        t_steps=t_steps,
        repeats=repeats,
        n_workers=n_workers,
        mu_c=mu_c,
        sigma_c=sigma_c
    )

    out_png = os.path.join(
        out_dir,
        f"s_{s}_muc{mu_c}_sigmac{sigma_c}_mud{mu_d}_mue{mu_e}_phase.png"
    )
    out_csv = os.path.join(
        out_dir,
        f"s_{s}_muc{mu_c}_sigmac{sigma_c}_mud{mu_d}_mue{mu_e}_phase.csv"
    )

    print("Saving results...")
    plot_heatmap(sigma_d_vals, sigma_e_vals, grid, out_png)
    save_grid_csv(sigma_d_vals, sigma_e_vals, grid, out_csv)

    print("Done. Files saved to", out_dir)


if __name__ == "__main__":
    main()