#!/usr/bin/env python3
import os
import csv
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt

# 全局参数
mu_c = 0.0
sigma_c = 2.0 * np.sqrt(3.0) / 27.0
# -------------------- 参数生成 --------------------
def generate_parameters(s,
                        mu_c, sigma_c,
                        mu_d, sigma_d, rho_d,
                        mu_e, sigma_e):
    c_i = np.random.normal(mu_c, sigma_c, s)

    mean_d = mu_d / s
    d_ij = np.random.normal(mean_d/s, sigma_d/np.sqrt(s), (s, s))
    eps = np.random.normal(mean_d/s, sigma_d/np.sqrt(s), (s, s))
    d_ji = rho_d * d_ij + np.sqrt(max(0.0, 1 - rho_d**2)) * eps

    mean_e = mu_e / (s * s)
    e_ijk = np.random.normal(mean_e, sigma_e/s, (s, s, s))

    return c_i, d_ij, d_ji, e_ijk


# -------------------- 动力学 --------------------
def dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps):
    x = x_init.copy()
    dt = 0.01

    for _ in range(t_steps):

        def f(xloc):
            dx = -xloc**3 + xloc + c_i
            dx += np.dot(d_ji, xloc)
            dx += np.einsum('ijk,j,k->i', e_ijk, xloc, xloc)
            return dx

        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x += (k1 + 2*k2 + 2*k3 + k4) * dt / 6.0

    return x


def calculate_survival_rate(final_states):
    return float(np.sum(final_states > 0)) / len(final_states)


def single_simulation_once(s, mu_c, sigma_c,
                           mu_d, sigma_d, rho_d,
                           mu_e, sigma_e, t_steps):
    c_i, _, d_ji, e_ijk = generate_parameters(
        s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e
    )
    x_init = np.full(s, -0.6)
    final_states = dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps)
    return calculate_survival_rate(final_states)


# -------------------- Worker（重要：放到最外层） --------------------
def worker(p):
    s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps, repeats = p
    vals = [
        single_simulation_once(
            s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps
        )
        for _ in range(repeats)
    ]
    return float(np.mean(vals))


# -------------------- 网格计算 --------------------
def compute_grid(mu_d_vals, mu_e_vals,
                 s, sigma_d, sigma_e,
                 t_steps, repeats,
                 workers=None):

    rho_d = 0.0

    tasks = []
    for mu_e in mu_e_vals:
        for mu_d in mu_d_vals:
            tasks.append((s, mu_c, sigma_c, mu_d, sigma_d, rho_d,
                          mu_e, sigma_e, t_steps, repeats))

    if workers is None:
        workers = max(1, cpu_count())

    with Pool(workers) as pool:
        results = pool.map(worker, tasks)

    grid = np.array(results).reshape(len(mu_e_vals), len(mu_d_vals))
    return grid


# -------------------- 输出 --------------------
def plot_heatmap(mu_d, mu_e, grid, out_png):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, origin='lower', aspect='auto',
                   extent=[mu_d[0], mu_d[-1], mu_e[0], mu_e[-1]],
                   cmap='viridis', vmin=0, vmax=1)
    ax.set_xlabel("mu_d")
    ax.set_ylabel("mu_e")
    ax.set_title("Survival rate")
    plt.colorbar(im, ax=ax, label="Survival rate")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_csv(mu_d, mu_e, grid, out_csv):
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["mu_e \\ mu_d"] + list(mu_d))
        for j, val in enumerate(mu_e):
            w.writerow([val] + list(grid[j]))


# -------------------- 主程序（无需命令行） --------------------
def main():

    nx = 100
    ny = 100
    mu_d_min = -0.5
    mu_d_max = 0.5
    mu_e_min = -0.5
    mu_e_max = 0.5
    s = 50
    sigma_d = 0.2
    sigma_e = 0.2
    t_steps = 2400
    repeats = 1
    workers = None
    out = "out_mu_d_mu_e"

    os.makedirs(out, exist_ok=True)

    mu_d_vals = np.linspace(mu_d_min, mu_d_max, nx)
    mu_e_vals = np.linspace(mu_e_min, mu_e_max, ny)

    print("Computing grid (parallel, no pickle errors)...")

    grid = compute_grid(mu_d_vals, mu_e_vals,
                        s=s,
                        sigma_d=sigma_d,
                        sigma_e=sigma_e,
                        t_steps=t_steps,
                        repeats=repeats,
                        workers=workers)

    png = os.path.join(out, f"s_{s}_mu_c_{mu_c}_sigma_c_{sigma_c}_sigma_d_{sigma_d}_sigma_e_{sigma_e}_phase.png")
    csvf = os.path.join(out, f"s_{s}_mu_c_{mu_c}_sigma_c_{sigma_c}_sigma_d_{sigma_d}_sigma_e_{sigma_e}_phase.csv")

    print("Saving output...")
    plot_heatmap(mu_d_vals, mu_e_vals, grid, png)
    save_csv(mu_d_vals, mu_e_vals, grid, csvf)

    print("Done.")
    print("PNG:", png)
    print("CSV:", csvf)


if __name__ == "__main__":
    main()