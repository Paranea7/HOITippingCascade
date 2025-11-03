#!/usr/bin/env python3
"""
phase_sigma_d_sigma_e.py

固定 s=30, mu_e=0.1, mu_d=0.2，
在 sigma_d x sigma_e 网格上计算系统的存活率相图（heatmap）。

说明:
- 网格尺寸: 默认横纵各 80 点 (可通过变量 sigma_d_vals 与 sigma_e_vals 修改)。
- 每个格点上运行 repeats 次随机仿真并取平均。默认 repeats=1（单次随机样本）。
- t_steps 控制动力学积分步数（默认 2000）。增大 t_steps 可提高精度但更慢。
- 可以开启并行计算 use_parallel=True（在 Windows 平台请确保在 if __name__ == '__main__' 下运行脚本）。
- 结果会保存为 PNG（heatmap）与 CSV（原始网格数据）。
"""

import os
import csv
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

# -------------------- 核心模型函数（与用户提供的实现保持一致） --------------------

def generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e):
    """
    生成模型参数：
    c_i: 控制参数向量（长度 s）
    d_ij, d_ji: 二体耦合矩阵（按 s 缩放）
    e_ijk: 三体耦合张量（按 s^2 缩放）
    """
    c_i = np.random.normal(mu_c, sigma_c, s)
    d_ij = np.random.normal(mu_d / s, sigma_d / s, (s, s))
    d_ji = rho_d * d_ij + np.sqrt(max(0.0, 1 - rho_d**2)) * np.random.normal(mu_d / s, sigma_d / s, (s, s))
    e_ijk = np.random.normal(mu_e / s**2, sigma_e / s**2, (s, s, s))
    return c_i, d_ij, d_ji, e_ijk

def dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps):
    """
    使用 RK4 积分器迭代动力学方程：
    dx/dt = -x^3 + x + c_i + d_ji @ x + einsum(e_ijk, x, x)
    返回最终状态向量 x（长度 s）
    """
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
    """
    根据最终状态计算存活率（>0 的个体比例）
    """
    survival = np.sum(final_states > 0)
    total = len(final_states)
    return float(survival) / float(total)

def single_simulation_once(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps, x0=0.6):
    """
    执行一次随机参数生成 + 动力学积分，返回单次存活率（float）。
    """
    c_i, _, d_ji, e_ijk = generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e)
    x_init = np.full(s, x0, dtype=float)
    final_states = dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps)
    return calculate_survival_rate(final_states)

# -------------------- 网格计算与并行支持 --------------------

def compute_grid(s,
                 mu_e,
                 mu_d,
                 sigma_d_vals,
                 sigma_e_vals,
                 t_steps=5000,
                 repeats=50,
                 use_parallel=False,
                 n_workers=None):
    """
    在 sigma_d_vals x sigma_e_vals 网格上计算 survival rate（每格点重复 repeats 次求平均）。
    返回：sigma_d_vals, sigma_e_vals, grid (shape: len(sigma_e_vals) x len(sigma_d_vals))
    """
    if n_workers is None:
        n_workers = max(1, cpu_count())

    # 固定的其他参数
    mu_c = 0.0
    sigma_c = 2.0 * np.sqrt(3.0) / 9.0
    rho_d = 1.0

    # 构造任务参数列表：按行 (sigma_e) 优先，再列 (sigma_d)，便于 reshape 回网格
    tasks = []
    for sigma_e in sigma_e_vals:
        for sigma_d in sigma_d_vals:
            tasks.append((s, mu_c, sigma_c, mu_d, float(sigma_d), rho_d, mu_e, float(sigma_e), t_steps, repeats))

    def worker_task(params):
        s_local, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps, repeats = params
        vals = []
        for _ in range(repeats):
            vals.append(single_simulation_once(s_local, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps))
        return float(np.mean(vals))

    if use_parallel:
        # 使用 multiprocessing Pool 并行执行
        with Pool(processes=n_workers) as pool:
            results = pool.map(worker_task, tasks)
    else:
        # 串行执行（跨平台稳定）
        results = [worker_task(p) for p in tasks]

    # reshape: rows = len(sigma_e_vals), cols = len(sigma_d_vals)
    grid = np.array(results).reshape((len(sigma_e_vals), len(sigma_d_vals)))
    return sigma_d_vals, sigma_e_vals, grid

# -------------------- 输出（PNG + CSV） --------------------

def plot_heatmap(sigma_d_vals, sigma_e_vals, grid, out_png="phase_sigma_d_sigma_e.png", cmap='viridis'):
    """
    使用 imshow 绘制 heatmap 并保存为 PNG。返回保存路径。
    """
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(grid, origin='lower', aspect='auto',
                   extent=[sigma_d_vals[0], sigma_d_vals[-1], sigma_e_vals[0], sigma_e_vals[-1]],
                   cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xlabel("sigma_d")
    ax.set_ylabel("sigma_e")
    ax.set_title("Survival rate (s=30, mu_e=0.1, mu_d=0.2)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Survival rate")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png

def save_grid_csv(sigma_d_vals, sigma_e_vals, grid, out_csv="phase_sigma_d_sigma_e.csv"):
    """
    将网格数据保存为 CSV。
    CSV 格式：第一行是 header：sigma_d values（列），第一列是 sigma_e
    每行代表一个 sigma_e（从小到大），列为对应 sigma_d 的生存率。
    """
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # header: 空格 + sigma_d 值
        header = ["sigma_e\\sigma_d"] + [f"{v:.6g}" for v in sigma_d_vals]
        writer.writerow(header)
        for j, sigma_e in enumerate(sigma_e_vals):
            row = [f"{sigma_e:.6g}"] + [f"{v:.6g}" for v in grid[j, :]]
            writer.writerow(row)
    return out_csv

# -------------------- 命令行接口 --------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compute phase diagram (survival rate) over sigma_d x sigma_e for fixed s=30, mu_e=0.1, mu_d=0.2.")
    p.add_argument("--s", type=int, default=30, help="system size (default 30)")
    p.add_argument("--mu_e", type=float, default=0.01, help="mu_e (default 0.1)")
    p.add_argument("--mu_d", type=float, default=0., help="mu_d (default 0.2)")
    p.add_argument("--nx", type=int, default=80, help="number of sigma_d points (default 80)")
    p.add_argument("--ny", type=int, default=80, help="number of sigma_e points (default 80)")
    p.add_argument("--t_steps", type=int, default=2000, help="number of integration steps (default 2000)")
    p.add_argument("--repeats", type=int, default=1, help="repeats per grid point to average (default 1)")
    p.add_argument("--parallel", action="store_true", help="enable multiprocessing")
    p.add_argument("--workers", type=int, default=None, help="number of workers for multiprocessing (default cpu_count()-1)")
    p.add_argument("--out_dir", type=str, default="output_phase", help="output directory (PNG + CSV) (default 'output_phase')")
    p.add_argument("--quick", action="store_true", help="quick mode: reduce nx, ny, t_steps for fast test")
    return p.parse_args()

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def main():
    args = parse_args()

    # quick 模式下自动缩小格点与步数，便于快速测试
    if args.quick:
        args.nx = min(40, args.nx)
        args.ny = min(40, args.ny)
        args.t_steps = max(200, args.t_steps // 10)
        print("Quick mode enabled: reduced grid and t_steps for faster execution.")

    sigma_d_vals = np.linspace(0.0, 1.0, args.nx)
    sigma_e_vals = np.linspace(0.0, 1.0, args.ny)

    out_dir = ensure_dir(args.out_dir)
    out_png = os.path.join(out_dir, f"phase_s{args.s}_mue{args.mu_e}_mud{args.mu_d}_nx{args.nx}_ny{args.ny}.png")
    out_csv = os.path.join(out_dir, f"phase_s{args.s}_mue{args.mu_e}_mud{args.mu_d}_nx{args.nx}_ny{args.ny}.csv")

    print("Parameters:")
    print(" s =", args.s)
    print(" mu_e =", args.mu_e, " mu_d =", args.mu_d)
    print(" grid:", args.nx, "x", args.ny)
    print(" t_steps =", args.t_steps, " repeats =", args.repeats)
    print(" parallel =", args.parallel, " workers =", args.workers)
    print(" outputs ->", out_dir)
    print("Starting computation... (this will take time)")

    sigma_d_vals, sigma_e_vals, grid = compute_grid(
        s=args.s,
        mu_e=args.mu_e,
        mu_d=args.mu_d,
        sigma_d_vals=sigma_d_vals,
        sigma_e_vals=sigma_e_vals,
        t_steps=args.t_steps,
        repeats=args.repeats,
        use_parallel=args.parallel,
        n_workers=args.workers
    )

    print("Computation finished. Saving results...")
    png_path = plot_heatmap(sigma_d_vals, sigma_e_vals, grid, out_png=out_png)
    csv_path = save_grid_csv(sigma_d_vals, sigma_e_vals, grid, out_csv=out_csv)
    print("Saved PNG to:", png_path)
    print("Saved CSV to:", csv_path)
    print("Done.")

if __name__ == "__main__":
    main()