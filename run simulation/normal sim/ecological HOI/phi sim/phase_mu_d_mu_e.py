#!/usr/bin/env python3
"""
phase_mu_d_mu_e.py

在 mu_d x mu_e 网格上计算系统的存活率相图（heatmap），固定：
- s=30（可通过命令行修改）
- sigma_d = sigma_e = 0.5（固定）
- c_i 的生成：phi_0=0.3 的比例随机选择被设置为 c_high=2.0*sqrt(3)/9，其余为 0
- 网格为 mu_d x mu_e，默认 100x100，范围均为 [0,1]

用法示例:
    python phase_mu_d_mu_e.py --nx 100 --ny 100 --repeats 10 --t_steps 2000 --parallel

注意：
- 并行模式在 Windows 上要确保在 if __name__ == '__main__' 下运行（脚本已如此）。
- 运行时间随 repeats、t_steps、网格大小线性增长。
"""

import os
import csv
import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt

# -------------------- 核心模型函数（按照新要求修改 c 的生成与固定 sigma） --------------------

def generate_parameters(s, phi_0, c_high, mu_d, sigma_d, rho_d, mu_e, sigma_e):
    """
    生成模型参数：
    c_i: 按 phi_0 比例随机选择 floor(phi_0*s) 个元素设置为 c_high，其余为 0
    d_ij, d_ji: 二体耦合矩阵（按 s 缩放）
    e_ijk: 三体耦合张量（按 s^2 缩放）
    """
    c_i = np.zeros(s, dtype=float)
    k = int(np.floor(phi_0 * s))
    if k > 0:
        idx = np.random.choice(s, size=k, replace=False)
        c_i[idx] = c_high

    # 二体耦合（按 s 缩放）
    d_ij = np.random.normal(mu_d / s, sigma_d / s, (s, s))
    # 生成与 d_ij 相关的 d_ji（相关系数 rho_d）
    d_ji = rho_d * d_ij + np.sqrt(max(0.0, 1 - rho_d**2)) * np.random.normal(mu_d / s, sigma_d / s, (s, s))

    # 三体耦合（按 s^2 缩放）
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

def single_simulation_once(s, phi_0, c_high, mu_d, sigma_d_used, rho_d, mu_e, sigma_e_used, t_steps, x0=0.6):
    """
    执行一次随机参数生成 + 动力学积分，返回单次存活率（float）。
    sigma_d_used 和 sigma_e_used 固定为 0.5（由调用方传入）
    """
    c_i, _, d_ji, e_ijk = generate_parameters(s, phi_0, c_high, mu_d, sigma_d_used, rho_d, mu_e, sigma_e_used)
    x_init = np.full(s, x0, dtype=float)
    final_states = dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps)
    return calculate_survival_rate(final_states)

# -------------------- 网格计算与并行支持（mu_d x mu_e 网格） --------------------

def compute_grid(s,
                 phi_0,
                 c_high,
                 sigma_d_fixed,
                 sigma_e_fixed,
                 mu_d_vals,
                 mu_e_vals,
                 t_steps=2000,
                 repeats=50,
                 use_parallel=False,
                 n_workers=None):
    """
    在 mu_d_vals x mu_e_vals 网格上计算 survival rate（每格点重复 repeats 次求平均）。
    sigma_d_fixed, sigma_e_fixed 为固定值（这里为 0.5）。
    返回：mu_d_vals, mu_e_vals, grid (shape: len(mu_e_vals) x len(mu_d_vals))
    """
    if n_workers is None:
        n_workers = max(1, cpu_count())

    # 固定的其他参数
    mu_c = 0.0
    sigma_c = 2.0 * np.sqrt(3.0) / 27.0
    rho_d = 1.0  # 保持与你原始脚本一致

    # 构造任务参数列表：按行 (mu_e) 优先，再列 (mu_d)，便于 reshape 回网格
    tasks = []
    for mu_e in mu_e_vals:
        for mu_d in mu_d_vals:
            tasks.append((s, phi_0, c_high, mu_d, sigma_d_fixed, rho_d, mu_e, sigma_e_fixed, t_steps, repeats))

    def worker_task(params):
        s_local, phi_0_local, c_high_local, mu_d_local, sigma_d_local, rho_d_local, mu_e_local, sigma_e_local, t_steps_local, repeats_local = params
        vals = []
        for _ in range(repeats_local):
            vals.append(single_simulation_once(s_local, phi_0_local, c_high_local, mu_d_local, sigma_d_local, rho_d_local, mu_e_local, sigma_e_local, t_steps_local))
        return float(np.mean(vals))

    if use_parallel:
        # 尝试并行计算
        with Pool(processes=n_workers) as pool:
            results = pool.map(worker_task, tasks)
    else:
        results = [worker_task(p) for p in tasks]

    grid = np.array(results).reshape((len(mu_e_vals), len(mu_d_vals)))
    return mu_d_vals, mu_e_vals, grid

# -------------------- 输出（PNG + CSV） --------------------

def plot_heatmap(mu_d_vals, mu_e_vals, grid, phi_0, sigma_fixed, out_png="phase_mu_d_mu_e.png", cmap='viridis'):
    """
    使用 imshow 绘制 heatmap 并保存为 PNG。返回保存路径。
    横轴为 mu_d, 纵轴为 mu_e。
    """
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(grid, origin='lower', aspect='auto',
                   extent=[mu_d_vals[0], mu_d_vals[-1], mu_e_vals[0], mu_e_vals[-1]],
                   cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xlabel("mu_d")
    ax.set_ylabel("mu_e")
    ax.set_title(f"Survival rate (s=?, phi_0={phi_0}, sigma_d=sigma_e={sigma_fixed})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Survival rate")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png

def save_grid_csv(mu_d_vals, mu_e_vals, grid, out_csv="phase_mu_d_mu_e.csv"):
    """
    将网格数据保存为 CSV。
    CSV 格式：第一行 header: mu_d values（列），第一列是 mu_e
    每行代表一个 mu_e（从小到大），列为对应 mu_d 的生存率。
    """
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["mu_e\\mu_d"] + [f"{v:.6g}" for v in mu_d_vals]
        writer.writerow(header)
        for j, mu_e in enumerate(mu_e_vals):
            row = [f"{mu_e:.6g}"] + [f"{v:.6g}" for v in grid[j, :]]
            writer.writerow(row)
    return out_csv

# -------------------- 命令行接口 --------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compute phase diagram (survival rate) over mu_d x mu_e for fixed s, phi_0 and sigma_d=sigma_e=0.5.")
    p.add_argument("--s", type=int, default=50, help="system size (default 30)")
    p.add_argument("--phi_0", type=float, default=0.3, help="initial fraction set to c_high (default 0.3)")
    p.add_argument("--nx", type=int, default=100, help="number of mu_d points (default 100)")
    p.add_argument("--ny", type=int, default=100, help="number of mu_e points (default 100)")
    p.add_argument("--t_steps", type=int, default=2000, help="number of integration steps (default 2000)")
    p.add_argument("--repeats", type=int, default=20, help="repeats per grid point to average (default 1)")
    p.add_argument("--parallel", action="store_true", help="enable multiprocessing")
    p.add_argument("--workers", type=int, default=None, help="number of workers for multiprocessing (default cpu_count())")
    p.add_argument("--out_dir", type=str, default="output_phase_mu", help="output directory (PNG + CSV) (default 'output_phase_mu')")
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

    # 固定 sigma_d 和 sigma_e 为 0.5
    sigma_fixed = 0.5

    mu_d_vals = np.linspace(0.0, 1.0, args.nx)
    mu_e_vals = np.linspace(0.0, 1.0, args.ny)

    out_dir = ensure_dir(args.out_dir)
    out_png = os.path.join(out_dir, f"phase_s{args.s}_phi{args.phi_0}_sigd{sigma_fixed}_sige{sigma_fixed}_nx{args.nx}_ny{args.ny}.png")
    out_csv = os.path.join(out_dir, f"phase_s{args.s}_phi{args.phi_0}_sigd{sigma_fixed}_sige{sigma_fixed}_nx{args.nx}_ny{args.ny}.csv")

    print("Parameters:")
    print(" s =", args.s)
    print(" phi_0 =", args.phi_0)
    print(" sigma_d = sigma_e =", sigma_fixed)
    print(" grid (mu_d x mu_e):", args.nx, "x", args.ny)
    print(" t_steps =", args.t_steps, " repeats =", args.repeats)
    print(" parallel =", args.parallel, " workers =", args.workers)
    print(" outputs ->", out_dir)
    print("Starting computation... (this will take time)")

    # c_high 固定值
    c_high = 2.0 * np.sqrt(3.0) / 9.0

    mu_d_vals, mu_e_vals, grid = compute_grid(
        s=args.s,
        phi_0=args.phi_0,
        c_high=c_high,
        sigma_d_fixed=sigma_fixed,
        sigma_e_fixed=sigma_fixed,
        mu_d_vals=mu_d_vals,
        mu_e_vals=mu_e_vals,
        t_steps=args.t_steps,
        repeats=args.repeats,
        use_parallel=args.parallel,
        n_workers=args.workers
    )

    print("Computation finished. Saving results...")
    png_path = plot_heatmap(mu_d_vals, mu_e_vals, grid, args.phi_0, sigma_fixed, out_png=out_png)
    csv_path = save_grid_csv(mu_d_vals, mu_e_vals, grid, out_csv=out_csv)
    print("Saved PNG to:", png_path)
    print("Saved CSV to:", csv_path)
    print("Done.")

if __name__ == "__main__":
    main()