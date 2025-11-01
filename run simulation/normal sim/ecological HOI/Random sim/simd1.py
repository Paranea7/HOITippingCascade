#!/usr/bin/env python3
# simd_no_plot.py
# 服务器上运行：只生成 CSV，不依赖 matplotlib
#
# 用法示例：
#   python3 simd_no_plot.py --n-workers 64 --batch-size 10 --simulations-per-sigma 500 --t-steps 5000
#   python3 simd_no_plot.py --n-workers 48 --batch-size 5 --s-values 10 30 50 100
#
# 说明：
# - n-workers: 并行进程数（默认为 cpu_count()）
# - batch-size: 每个子任务内部执行多少次 single_simulation 再返回（减少调度开销）
# - simulations-per-sigma: 每个 sigma_d 总共要做多少次单次模拟（会被 batch-size 划分为若干子任务）
# - 如果 simulations-per-sigma 不能被 batch-size 整除，最后一个批次会较小以凑齐总数。

import os
import csv
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

def generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e):
    # 返回 c_i, d_ij, d_ji, e_ijk
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
    survival = np.sum(final_states > 0)
    total = len(final_states)
    return float(survival) / float(total)

def single_simulation(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps):
    # 执行一次完整的模拟并返回生存率
    c_i, _, d_ji, e_ijk = generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e)
    x_init = np.full(s, -0.6, dtype=float)
    final_states = dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps)
    return calculate_survival_rate(final_states)

def worker_batch(batch_size, s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps):
    # 在单个子进程内执行 batch_size 次 single_simulation，返回结果列表
    results = []
    for _ in range(batch_size):
        results.append(single_simulation(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps))
    return results

def save_results_csv(sigma_d_values, mu_d_values, survival_means, survival_ses, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}.csv")
    header = ['sigma_d']
    for mu_d in mu_d_values:
        header.append(f"mean_mu_d_{mu_d}")
        header.append(f"se_mu_d_{mu_d}")
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, sigma_d in enumerate(sigma_d_values):
            row = [float(sigma_d)]
            for j in range(len(mu_d_values)):
                row.append(float(survival_means[j][idx]))
                row.append(float(survival_ses[j][idx]))
            writer.writerow(row)
    return out_path

def run_one_parameter_combo(s, mu_e, sigma_e,
                            mu_d_values, sigma_d_values,
                            mu_c, sigma_c, rho_d,
                            t_steps, simulations_per_sigma,
                            out_csv_dir,
                            n_workers=None, batch_size=1):
    """
    对于给定 (s, mu_e, sigma_e) 的参数组合，遍历 mu_d_values 与 sigma_d_values，
    并为每个 sigma_d 运行 simulations_per_sigma 次 single_simulation（并行）。
    使用 n_workers 进程并以 batch_size 分割任务以减少调度开销。
    """
    if n_workers is None:
        n_workers = max(1, cpu_count())

    survival_means = []
    survival_ses = []

    # 为整个参数组合复用一个 Pool
    with Pool(processes=n_workers) as pool:
        for mu_d in mu_d_values:
            means_for_mu_d = []
            ses_for_mu_d = []
            for sigma_d in sigma_d_values:
                # 计算需要的批次数与每个批次的大小
                full_batches = simulations_per_sigma // batch_size
                remainder = simulations_per_sigma % batch_size
                tasks = []

                # 为每个 full batch 添加一个任务
                for _ in range(full_batches):
                    tasks.append((batch_size, s, mu_c, sigma_c, mu_d, float(sigma_d), rho_d, mu_e, sigma_e, t_steps))
                # 若 remainder > 0，则再添加一个小批次任务
                if remainder > 0:
                    tasks.append((remainder, s, mu_c, sigma_c, mu_d, float(sigma_d), rho_d, mu_e, sigma_e, t_steps))

                # 并行执行所有 tasks（每个 task 返回一个列表）
                if len(tasks) == 0:
                    # 罕见情况：simulations_per_sigma == 0
                    results_flat = []
                else:
                    results_lists = pool.starmap(worker_batch, tasks)
                    # 合并子列表为扁平列表
                    results_flat = [val for sub in results_lists for val in sub]

                # 计算均值与标准误（样本标准差除以 sqrt(n)）
                if len(results_flat) == 0:
                    mean_surv = 0.0
                    se_surv = 0.0
                else:
                    mean_surv = float(np.mean(results_flat))
                    se_surv = float(np.std(results_flat, ddof=1) / np.sqrt(len(results_flat))) if len(results_flat) > 1 else 0.0

                means_for_mu_d.append(mean_surv)
                ses_for_mu_d.append(se_surv)

            survival_means.append(means_for_mu_d)
            survival_ses.append(ses_for_mu_d)

    prefix = f"s_{s}_mue_{mu_e}_sigmae_{sigma_e}"
    csv_path = save_results_csv(sigma_d_values, mu_d_values, survival_means, survival_ses, out_csv_dir, prefix)

    return csv_path

def parse_floats_list(values):
    # 辅助：把 argparse 列表字符串解析为 float 列表（允许传入单个值）
    return [float(x) for x in values]

def main():
    parser = argparse.ArgumentParser(description="Run simulations and save CSVs (no plotting).")
    parser.add_argument("--n-workers", type=int, default=max(1, cpu_count()),
                        help="Number of worker processes to use (default: cpu_count()).")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of single_simulation runs per worker task (default 1). Set >1 to reduce scheduling overhead.")
    parser.add_argument("--simulations-per-sigma", type=int, default=500,
                        help="Total number of single simulations per sigma_d value (default 500).")
    parser.add_argument("--t-steps", type=int, default=2000,
                        help="Number of integration timesteps per simulation (default 5000).")
    parser.add_argument("--out-dir", type=str, default="outputcsvdrandom0",
                        help="Directory to save CSV output files.")
    parser.add_argument("--s-values", type=int, nargs="+", default=[10,30,50,100],
                        help="List of s values to run (default: 10 30 50 100).")
    parser.add_argument("--mu-e-values", type=float, nargs="+",
                        default=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                        help="List of mu_e values to run.")
    parser.add_argument("--sigma-e-values", type=float, nargs="+",
                        default=[0.0,0.1,0.2,0.3,0.5,0.7,1.0],
                        help="List of sigma_e values to run.")
    parser.add_argument("--mu-d-values", type=float, nargs="+",
                        default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                        help="List of mu_d values to iterate over.")
    parser.add_argument("--sigma-d-min", type=float, default=0.0,
                        help="Min sigma_d (default 0.0)")
    parser.add_argument("--sigma-d-max", type=float, default=1.0,
                        help="Max sigma_d (default 1.0)")
    parser.add_argument("--sigma-d-steps", type=int, default=21,
                        help="Number of sigma_d points between min and max (default 21)")
    parser.add_argument("--mu-c", type=float, default=0.0, help="mu_c (default 0.0)")
    parser.add_argument("--sigma-c", type=float, default=(2.0 * np.sqrt(3.0) / 9.0),
                        help="sigma_c (default 2*sqrt(3)/9)")
    parser.add_argument("--rho-d", type=float, default=1.0, help="rho_d (default 0.0)")
    args = parser.parse_args()

    mu_c = args.mu_c
    sigma_c = args.sigma_c
    rho_d = args.rho_d

    s_values = args.s_values
    mu_e_values = args.mu_e_values
    sigma_e_values = args.sigma_e_values

    mu_d_values = args.mu_d_values
    sigma_d_values = np.linspace(args.sigma_d_min, args.sigma_d_max, args.sigma_d_steps)

    t_steps = args.t_steps
    simulations_per_sigma = args.simulations_per_sigma
    batch_size = args.batch_size
    n_workers = args.n_workers
    out_csv_dir = args.out_dir

    if batch_size <= 0:
        raise ValueError("batch-size must be >= 1")
    if simulations_per_sigma < 0:
        raise ValueError("simulations-per-sigma must be >= 0")
    if simulations_per_sigma > 0 and batch_size > simulations_per_sigma:
        # 合理性检查：允许但会导致许多小任务，警告用户
        print("Warning: batch-size > simulations-per-sigma; some batches will be size < batch-size.")

    os.makedirs(out_csv_dir, exist_ok=True)

    generated_files = []

    total = len(s_values) * len(mu_e_values) * len(sigma_e_values)
    cnt = 0
    # 主循环：遍历 s, mu_e, sigma_e
    for s in s_values:
        for mu_e in mu_e_values:
            for sigma_e in sigma_e_values:
                cnt += 1
                print(f"[{cnt}/{total}] Starting run for s={s}, mu_e={mu_e}, sigma_e={sigma_e}  "
                      f"(n_workers={n_workers}, batch_size={batch_size}, sims_per_sigma={simulations_per_sigma})")
                csv_path = run_one_parameter_combo(
                    s, mu_e, sigma_e,
                    mu_d_values, sigma_d_values,
                    mu_c, sigma_c, rho_d,
                    t_steps, simulations_per_sigma,
                    out_csv_dir,
                    n_workers=n_workers, batch_size=batch_size
                )
                generated_files.append(csv_path)
                print(f"  Saved CSV: {csv_path}")

    print("All parameter combinations finished.")
    print("Generated files:")
    for c in generated_files:
        print("  ", c)

if __name__ == "__main__":
    main()