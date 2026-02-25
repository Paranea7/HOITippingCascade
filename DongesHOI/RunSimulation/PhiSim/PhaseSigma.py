#!/usr/bin/env python3
import os
import csv
import time
from multiprocessing import Pool, cpu_count
import numpy as np
# import matplotlib.pyplot as plt  # <--- 注释掉绘图库，方便服务器运行
from numba import njit

# 全局常量
phi0 = 0.05
c_high = 0.4
c_low = 0.0


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
        x += (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6.0
    return x


def generate_parameters(s, mu_d, sigma_d, rho_d, mu_e, sigma_e):
    c_i = np.full(s, c_low)
    n_high = int(phi0 * s)
    # 防止 n_high 大于 s 或小于 0 的边界情况
    if n_high > 0:
        idx = np.random.choice(s, n_high, replace=False)
        c_i[idx] = c_high

    d_ij = np.random.normal(mu_d / s, sigma_d / np.sqrt(s), (s, s))
    d_ji = rho_d * d_ij + np.sqrt(max(0.0, 1 - rho_d ** 2)) * \
           np.random.normal(mu_d / s, sigma_d / np.sqrt(s), (s, s))

    for i in range(s):
        d_ij[i, i] = 0.0
        d_ji[i, i] = 0.0

    e_ijk = np.random.normal(mu_e / s ** 2, sigma_e / s, (s, s, s))

    for i in range(s):
        for j in range(s):
            e_ijk[i, j, i] = 0.0
        for k in range(s):
            e_ijk[i, i, k] = 0.0
    for i in range(s):
        e_ijk[i, i, i] = 0.0

    return c_i, d_ij, d_ji, e_ijk


def dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps):
    return dynamics_simulation_numba(s, c_i, d_ji, e_ijk, x_init, t_steps)


def calculate_survival_rate(final_states):
    return float(np.sum(final_states > 0)) / len(final_states)


def single_simulation_once(s, mu_d, sigma_d, rho_d, mu_e, sigma_e,
                           t_steps, x0=-0.6):
    c_i, _, d_ji, e_ijk = generate_parameters(
        s, mu_d, sigma_d, rho_d, mu_e, sigma_e
    )

    x_init = np.full(s, x0, float)
    final_states = dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps)

    return calculate_survival_rate(final_states)


def worker_task(params):
    s_local, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps, reps = params
    vals = []
    # 为了保证随机性独立，可以在进程内重新播种，但在 Pool 中通常不需要
    for _ in range(reps):
        vals.append(
            single_simulation_once(
                s_local, mu_d, sigma_d, rho_d,
                mu_e, sigma_e, t_steps
            )
        )
    return float(np.mean(vals))


def compute_grid(
        s,
        mu_e,
        mu_d,
        sigma_d_vals,
        sigma_e_vals,
        t_steps=3000,
        repeats=50,
        n_workers=None
):
    if n_workers is None:
        n_workers = max(1, cpu_count())

    rho_d = 0.0

    tasks = []
    # 构建任务列表
    for sigma_e in sigma_e_vals:
        for sigma_d in sigma_d_vals:
            tasks.append(
                (
                    s, mu_d, float(sigma_d), rho_d,
                    mu_e, float(sigma_e),
                    t_steps, repeats
                )
            )

    print(f"  -> Computing grid: {len(sigma_e_vals)}x{len(sigma_d_vals)} points with {n_workers} workers.")

    with Pool(processes=n_workers) as pool:
        results = pool.map(worker_task, tasks)

    grid = np.array(results).reshape(len(sigma_e_vals), len(sigma_d_vals))
    return sigma_d_vals, sigma_e_vals, grid


# -----------------------------------------------------------------------------
# 绘图部分已注释
# -----------------------------------------------------------------------------
# def plot_heatmap(sigma_d_vals, sigma_e_vals, grid,
#                  out_png="phase_sigma_d_sigma_e.png",
#                  cmap='viridis'):
#     fig, ax = plt.subplots(figsize=(8,6))
#     im = ax.imshow(
#         grid,
#         origin='lower',
#         aspect='auto',
#         extent=[sigma_d_vals[0], sigma_d_vals[-1],
#                 sigma_e_vals[0], sigma_e_vals[-1]],
#         cmap=cmap,
#         vmin=0.0, vmax=1.0
#     )
#     ax.set_xlabel("sigma_d")
#     ax.set_ylabel("sigma_e")
#     ax.set_title("Survival rate")
#     fig.colorbar(im, ax=ax).set_label("Survival rate")
#     fig.tight_layout()
#     fig.savefig(out_png, dpi=150)
#     plt.close(fig)
#     return out_png


def save_grid_csv(sigma_d_vals, sigma_e_vals, grid, out_csv):
    """保存计算结果到CSV"""
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        # 第一行表头：第一列标识，后面是 sigma_d 的值
        w.writerow(["sigma_e\\sigma_d"] + [f"{v:.6g}" for v in sigma_d_vals])
        # 后续行：第一列是 sigma_e 的值，后面是对应的 grid 数据
        for j, sigma_e in enumerate(sigma_e_vals):
            row = [f"{sigma_e:.6g}"] + [f"{v:.6g}" for v in grid[j, :]]
            w.writerow(row)
    print(f"  -> Saved CSV to: {out_csv}")
    return out_csv


def main():
    # ================= 配置区域 =================
    out_dir = "output_phase_batch"
    os.makedirs(out_dir, exist_ok=True)

    # 通用设置
    nx = 50  # sigma_d 的分辨率
    ny = 50  # sigma_e 的分辨率
    t_steps = 3000
    repeats = 50
    n_workers = None  # None 表示自动使用所有核心

    # 在这里定义你要批量运行的参数组合
    # 每一行字典代表一次完整的 grid 计算任务
    batch_configs = [
        # 组合 1: 默认参数
        {'s': 50, 'mu_e': 0.5, 'mu_d': 0.2},

        # 组合 2: 更大的系统尺寸
        {'s': 50, 'mu_e': -0.2, 'mu_d': 0.2},

        # 组合 3: 不同的 mu_e
        {'s': 50, 'mu_e': 0.2, 'mu_d': 0.5},

        # 组合 4: 不同的 mu_d
        {'s': 50, 'mu_e': -0.2, 'mu_d': -0.2},
    ]
    # ===========================================

    total_tasks = len(batch_configs)
    print(f"Starting batch processing of {total_tasks} configurations...")
    print(f"Output directory: {out_dir}")

    start_time_all = time.time()

    for idx, config in enumerate(batch_configs):
        s = config['s']
        mu_e = config['mu_e']
        mu_d = config['mu_d']

        print(f"\n[{idx + 1}/{total_tasks}] Processing: s={s}, mu_e={mu_e}, mu_d={mu_d}")

        # 定义扫描范围
        sigma_d_vals = np.linspace(0, 1, nx)
        sigma_e_vals = np.linspace(0, 1, ny)

        # 运行计算
        sigma_d_vals, sigma_e_vals, grid = compute_grid(
            s=s,
            mu_e=mu_e,
            mu_d=mu_d,
            sigma_d_vals=sigma_d_vals,
            sigma_e_vals=sigma_e_vals,
            t_steps=t_steps,
            repeats=repeats,
            n_workers=n_workers
        )

        # 生成唯一文件名
        file_tag = f"s{s}_phi0{phi0}_c{c_high}_muD{mu_d}_muE{mu_e}_t{t_steps}_rep{repeats}_nx{nx}_ny{ny}"
        out_csv = os.path.join(out_dir, f"{file_tag}_phase.csv")
        # out_png = os.path.join(out_dir, f"{file_tag}_phase.png") # 绘图文件名

        # 保存 CSV
        save_grid_csv(sigma_d_vals, sigma_e_vals, grid, out_csv)

        # 如果需要绘图，取消下面的注释，并取消上面 plot_heatmap 的注释
        # plot_heatmap(sigma_d_vals, sigma_e_vals, grid, out_png)

    end_time_all = time.time()
    print(f"\nAll tasks completed in {end_time_all - start_time_all:.2f} seconds.")


if __name__ == "__main__":
    main()