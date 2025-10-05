#!/usr/bin/env python3
"""
phase_mu_d_mu_e.py

在固定 s=30, sigma_d=0.3, sigma_e=0.1 的条件下，
在 mu_d x mu_e 网格上计算系统的存活率相图（heatmap）。

说明:
- 网格尺寸: 默认横纵各 80 点（可用 --nx/--ny 调整）
- 每个格点上运行 repeats 次随机仿真并取平均（默认 repeats=1）
- t_steps 控制积分步数（默认 2000）
- 可选并行计算 (--parallel)
- 输出 PNG + CSV 到指定目录
"""
import os
import csv
import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt

# -------------------- 核心模型函数（复用） --------------------
def generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e):
    c_i = np.random.normal(mu_c, sigma_c, s)
    d_ij = np.random.normal(mu_d / s, sigma_d / s, (s, s))
    d_ji = rho_d * d_ij + np.sqrt(max(0.0, 1 - rho_d**2)) * np.random.normal(mu_d / s, sigma_d / s, (s, s))
    e_ijk = np.random.normal(mu_e / s**2, sigma_e / s**2, (s, s, s))
    return c_i, d_ij, d_ji, e_ijk

def dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps):
    x = x_init.copy()
    dt = 0.01
    for _ in range(t_steps):
        def compute_dx(x_local):
            dx = -x_local**3 + x_local + c_i
            dx = dx + np.dot(d_ji, x_local)
            dx = dx + np.einsum('ijk,j,k->i', e_ijk, x_local, x_local)
            return dx
        k1 = compute_dx(x)
        k2 = compute_dx(x + 0.5 * dt * k1)
        k3 = compute_dx(x + 0.5 * dt * k2)
        k4 = compute_dx(x + dt * k3)
        x += (k1 + 2*k2 + 2*k3 + k4) * dt / 6.0
    return x

def calculate_survival_rate(final_states):
    return float(np.sum(final_states > 0)) / float(len(final_states))

def single_simulation_once(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps, x0=0.6):
    c_i, _, d_ji, e_ijk = generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e)
    x_init = np.random.normal(0,1, s)
    final_states = dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps)
    return calculate_survival_rate(final_states)

# -------------------- 网格计算 --------------------
def compute_grid_mu(mu_d_vals, mu_e_vals,
                    s=30,
                    sigma_d=0.2,
                    sigma_e=0.01,
                    t_steps=2000,
                    repeats=1,
                    use_parallel=True,
                    n_workers=None):
    """
    在 mu_d_vals x mu_e_vals 网格上计算生存率（每格点 repeats 次平均）。
    返回：mu_d_vals, mu_e_vals, grid (shape len(mu_e_vals) x len(mu_d_vals))
    """
    if n_workers is None:
        n_workers = max(1, cpu_count())

    mu_c = 0.0
    sigma_c = 2.0 * np.sqrt(3.0) / 9.0
    rho_d = 1.0

    tasks = []
    for mu_e in mu_e_vals:
        for mu_d in mu_d_vals:
            tasks.append((s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps, repeats))

    def worker_task(params):
        s_local, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps, repeats = params
        vals = [single_simulation_once(s_local, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps)
                for _ in range(repeats)]
        return float(np.mean(vals))

    if use_parallel:
        with Pool(processes=n_workers) as pool:
            results = pool.map(worker_task, tasks)
    else:
        results = [worker_task(p) for p in tasks]

    grid = np.array(results).reshape((len(mu_e_vals), len(mu_d_vals)))
    return mu_d_vals, mu_e_vals, grid

# -------------------- 输出 --------------------
def plot_heatmap(mu_d_vals, mu_e_vals, grid, out_png="phase_mu_d_mu_e.png", cmap='viridis'):
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(grid, origin='lower', aspect='auto',
                   extent=[mu_d_vals[0], mu_d_vals[-1], mu_e_vals[0], mu_e_vals[-1]],
                   cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xlabel("mu_d")
    ax.set_ylabel("mu_e")
    ax.set_title("Survival rate (s=30, sigma_d=0.3, sigma_e=0.1)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Survival rate")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png

def save_grid_csv(mu_d_vals, mu_e_vals, grid, out_csv="phase_mu_d_mu_e.csv"):
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["mu_e\\mu_d"] + [f"{v:.6g}" for v in mu_d_vals]
        writer.writerow(header)
        for j, mu_e in enumerate(mu_e_vals):
            row = [f"{mu_e:.6g}"] + [f"{v:.6g}" for v in grid[j, :]]
            writer.writerow(row)
    return out_csv

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Scan mu_d x mu_e at fixed sigma_d=0.3, sigma_e=0.1")
    p.add_argument("--nx", type=int, default=80, help="number of mu_d points")
    p.add_argument("--ny", type=int, default=80, help="number of mu_e points")
    p.add_argument("--mu_d_min", type=float, default=0.0, help="min mu_d")
    p.add_argument("--mu_d_max", type=float, default=1.0, help="max mu_d")
    p.add_argument("--mu_e_min", type=float, default=0.0, help="min mu_e")
    p.add_argument("--mu_e_max", type=float, default=1.0, help="max mu_e")
    p.add_argument("--s", type=int, default=30)
    p.add_argument("--sigma_d", type=float, default=0.2)
    p.add_argument("--sigma_e", type=float, default=0.01)
    p.add_argument("--t_steps", type=int, default=2000)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out_dir", type=str, default="output_mu_scan")
    p.add_argument("--quick", action="store_true", help="smaller grid and steps for quick test")
    return p.parse_args()

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def main():
    args = parse_args()
    if args.quick:
        args.nx = min(40, args.nx)
        args.ny = min(40, args.ny)
        args.t_steps = max(200, args.t_steps // 10)
        print("Quick mode enabled.")

    mu_d_vals = np.linspace(args.mu_d_min, args.mu_d_max, args.nx)
    mu_e_vals = np.linspace(args.mu_e_min, args.mu_e_max, args.ny)

    out_dir = ensure_dir(args.out_dir)
    out_png = os.path.join(out_dir, f"phase_mu_d_mu_e_s{args.s}_sigmad{args.sigma_d}_sigmae{args.sigma_e}_nx{args.nx}_ny{args.ny}.png")
    out_csv = os.path.join(out_dir, f"phase_mu_d_mu_e_s{args.s}_sigmad{args.sigma_d}_sigmae{args.sigma_e}_nx{args.nx}_ny{args.ny}.csv")

    print("Parameters:")
    print(" s", args.s, " sigma_d", args.sigma_d, " sigma_e", args.sigma_e)
    print(" mu_d range:", args.mu_d_min, "->", args.mu_d_max, " (nx=", args.nx, ")")
    print(" mu_e range:", args.mu_e_min, "->", args.mu_e_max, " (ny=", args.ny, ")")
    print(" t_steps", args.t_steps, " repeats", args.repeats, " parallel", args.parallel)

    mu_d_vals, mu_e_vals, grid = compute_grid_mu(
        mu_d_vals, mu_e_vals,
        s=args.s,
        sigma_d=args.sigma_d,
        sigma_e=args.sigma_e,
        t_steps=args.t_steps,
        repeats=args.repeats,
        use_parallel=args.parallel,
        n_workers=args.workers
    )

    print("Saving results...")
    png_path = plot_heatmap(mu_d_vals, mu_e_vals, grid, out_png=out_png)
    csv_path = save_grid_csv(mu_d_vals, mu_e_vals, grid, out_csv=out_csv)
    print("Saved PNG:", png_path)
    print("Saved CSV:", csv_path)

if __name__ == "__main__":
    main()