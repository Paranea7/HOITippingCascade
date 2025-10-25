#!/usr/bin/env python3
"""
Scan phi_0 (horizontal axis) vs survival rate (vertical axis).

功能：
- 参数可全部通过命令行或脚本顶部配置。
- 对每个 phi_0 做多次仿真，计算 mean ± SE。
- 支持固定或多个 sigma_d（每个 sigma_d 为一条曲线）。
- 输出 CSV（每组合一个）及 PNG（phi vs survival，带误差带）。

示例：
  python sim_vs_phi.py --t-steps 200 --sims-per-point 8 --phi-list 0.0,0.05,0.1,0.2 --sigma-d-list 0.5
  python sim_vs_phi.py --dry-run --phi-list 0.0,0.1,0.2
"""

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# ----------------- 核心模型（可根据需要调整） -----------------
def generate_parameters(s, mu_d, sigma_d, rho_d, mu_e, sigma_e):
    """
    生成 d_ij, d_ji, e_ijk（不生成 c_i）
    """
    d_ij = np.random.normal(mu_d / s, sigma_d / s, (s, s))
    np.fill_diagonal(d_ij, 0.0)

    noise = np.random.normal(mu_d / s, sigma_d / s, (s, s))
    d_ji = rho_d * d_ij + np.sqrt(max(0.0, 1.0 - rho_d ** 2)) * noise
    np.fill_diagonal(d_ji, 0.0)

    e_ijk = np.random.normal(mu_e / s**2, sigma_e / s**2, (s, s, s))
    for i in range(s):
        e_ijk[i, i, i] = 0.0
        for j in range(s):
            for k in range(j+1, s):
                val = 0.5 * (e_ijk[i, j, k] + e_ijk[i, k, j])
                e_ijk[i, j, k] = val
                e_ijk[i, k, j] = val

    return d_ij, d_ji, e_ijk

def dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps, dt=0.01):
    """
    RK4 积分，返回最终状态向量 x
    dx/dt = -x^3 + x + c_i + d_ji @ x + einsum(e_ijk, x, x)
    """
    x = x_init.copy()
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

def single_simulation(s, c_i, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps):
    """
    执行一次仿真：每次重新生成 d/e（若你想固定 d/e，可改成在外层生成并传入）
    """
    _, d_ji, e_ijk = generate_parameters(s, mu_d, sigma_d, rho_d, mu_e, sigma_e)
    x_init = np.full(s, -0.6, dtype=float)
    final = dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps)
    return calculate_survival_rate(final)

# ----------------- I/O 和绘图 -----------------
def save_results_csv(phi_list, sigma_d_list, survival_means, survival_ses, out_dir, prefix):
    """
    保存 CSV。结构：
    header: sigma_d, mean_phi_0.00, se_phi_0.00, mean_phi_0.05, se_phi_0.05, ...
    survival_means shape: (len(phi_list), len(sigma_d_list))
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}.csv")
    header = ['sigma_d']
    for phi in phi_list:
        header.append(f"mean_phi_{phi}")
        header.append(f"se_phi_{phi}")
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for j, sigma in enumerate(sigma_d_list):
            row = [float(sigma)]
            for i in range(len(phi_list)):
                row.append(float(survival_means[i, j]))
                row.append(float(survival_ses[i, j]))
            writer.writerow(row)
    return out_path

def plot_survival_vs_phi(phi_list, sigma_d_list, survival_means, survival_ses, out_dir, prefix,
                         cmap='viridis'):
    """
    横轴 phi_list，纵轴 survival rate。
    每条曲线为一个 sigma_d（若 sigma_d_list 只有一个值，则只画一条曲线）。
    """
    os.makedirs(out_dir, exist_ok=True)
    phi = np.array(phi_list, dtype=float)
    means = np.array(survival_means)  # shape (len(phi), len(sigma_d))
    ses = np.array(survival_ses)

    n_sigma = len(sigma_d_list)
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i / max(1, n_sigma-1)) for i in range(n_sigma)]

    fig, ax = plt.subplots(figsize=(8, 5))
    for j in range(n_sigma):
        y = means[:, j]
        yerr = ses[:, j]
        ax.plot(phi, y, marker='o', color=colors[j], label=f"sigma_d={sigma_d_list[j]:.3g}")
        ax.fill_between(phi, y - yerr, y + yerr, color=colors[j], alpha=0.2)
    ax.set_xlabel("phi_0")
    ax.set_ylabel("Survival Rate")
    ax.set_title("Survival Rate vs phi_0")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"{prefix}_vs_phi.png")
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return out_path

# ----------------- 运行控制 -----------------
def run_scan_for_combo(s, mu_e, sigma_e,
                       mu_d_values, sigma_d_values,
                       mu_c, sigma_c, rho_d,
                       phi_list, t_steps, sims_per_point,
                       out_plot_dir, out_csv_dir,
                       n_workers=None,
                       fixed_c_value=None,
                       phi_is_fraction=True,
                       generate_new_d_each_run=True):
    """
    对单一 (s, mu_e, sigma_e)：
    - phi_list: list of phi_0 值（可为比例 0-1，或绝对个数 >1）
    - phi_is_fraction: 如果 True，则 phi 在 [0,1] 表示占比；否则表示绝对个数
    - generate_new_d_each_run: True 表示每次 single_simulation 都重新生成 d/e（默认）
    返回 CSV 路径、PNG 路径
    """
    if n_workers is None:
        n_workers = max(1, cpu_count()-1)

    n_phi = len(phi_list)
    n_sigma = len(sigma_d_values)
    survival_means = np.zeros((n_phi, n_sigma), dtype=float)
    survival_ses = np.zeros((n_phi, n_sigma), dtype=float)

    for i_phi, phi in enumerate(phi_list):
        # 计算每次仿真需要激活的个体数 count
        if phi_is_fraction:
            if phi <= 0.0:
                count = 0
            elif phi >= 1.0:
                count = s
            else:
                count = int(round(phi * s))
        else:
            # phi 给的是绝对个数
            count = int(round(phi))
            if count < 0:
                count = 0
            if count > s:
                count = s

        for j_sigma, sigma_d in enumerate(sigma_d_values):
            # 为每个重复采样 c_i（独立重复）
            c_list = []
            for _ in range(sims_per_point):
                c_i = np.zeros(s, dtype=float)
                if count > 0:
                    idx = np.random.choice(s, size=count, replace=False)
                    c_i[idx] = float(fixed_c_value) if fixed_c_value is not None else float(mu_c)
                c_list.append(c_i)

            # 如果你想在同一 phi 与 sigma_d 下固定一个 d/e、只替换 c_i，将 generate_new_d_each_run=False，并在下面生成一次 d/e 并传入 single_simulation 的变体。
            # 当前 single_simulation 内部会为每次生成 d/e。
            args = [(s, c_list[k], mu_d_values[0] if len(mu_d_values)==1 else mu_d_values[0],  # mu_d passed but we loop over mu_d externally if needed
                     float(sigma_d), rho_d, mu_e, sigma_e, t_steps)
                    for k in range(sims_per_point)]

            # 使用并行池运行
            with Pool(processes=n_workers) as pool:
                results = pool.starmap(single_simulation, args)

            mean_surv = float(np.mean(results))
            se_surv = float(np.std(results, ddof=1) / np.sqrt(len(results))) if len(results) > 1 else 0.0

            survival_means[i_phi, j_sigma] = mean_surv
            survival_ses[i_phi, j_sigma] = se_surv

    prefix = f"s_{s}_mue_{mu_e}_sigmae_{sigma_e}_phi_scan"
    csv_path = save_results_csv(phi_list, sigma_d_values, survival_means, survival_ses, out_csv_dir, prefix)
    plot_path = plot_survival_vs_phi(phi_list, sigma_d_values, survival_means, survival_ses, out_plot_dir, prefix)

    return csv_path, plot_path, survival_means, survival_ses

# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Scan phi_0 vs survival rate.")
    p.add_argument('--s', type=int, default=100, help='system size (number of species)')
    p.add_argument('--mu-e', type=float, default=0.3, help='mu_e')
    p.add_argument('--sigma-e', type=float, default=0.3, help='sigma_e')
    p.add_argument('--mu-d', type=float, default=0.3, help='(single) mu_d value - if you want multiple mu_d, pass comma list and edit code accordingly')
    p.add_argument('--sigma-d-list', type=str, default='0.5', help='comma-separated sigma_d values (each will be a curve)')
    p.add_argument('--phi-list', type=str, default='0.0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9', help='comma-separated phi_0 values (fractions or absolute depending on --phi-absolute)')
    p.add_argument('--phi-absolute', action='store_true', help='if set, phi-list interpreted as absolute counts (not fraction)')
    p.add_argument('--t-steps', type=int, default=800, help='RK4 steps')
    p.add_argument('--sims-per-point', type=int, default=20, help='simulations per phi point')
    p.add_argument('--n-workers', type=int, default=max(1, cpu_count()-1), help='parallel workers')
    p.add_argument('--out-plot-dir', type=str, default='out_phi_plots', help='plot output dir')
    p.add_argument('--out-csv-dir', type=str, default='out_phi_csv', help='csv output dir')
    p.add_argument('--dry-run', action='store_true', help='only print planned combos')
    p.add_argument('--max-combos', type=int, default=1, help='limit combos (s,mu_e,sigma_e) to run')
    return p.parse_args()

def main():
    args = parse_args()

    # 用户可以在命令行传入多个 sigma_d（逗号分割）
    sigma_d_values = [float(x) for x in args.sigma_d_list.split(',') if x.strip()!='']
    # 用户传入 phi 列表（逗号分割）
    phi_list = [float(x) for x in args.phi_list.split(',') if x.strip()!='']

    # 其余参数
    s = args.s
    mu_e = args.mu_e
    sigma_e = args.sigma_e
    mu_d = args.mu_d  # 当前实现只用到单个 mu_d；如需多个 mu_d 可扩展
    mu_d_values = [float(mu_d)]
    rho_d = 1.0
    mu_c = 0.0
    sigma_c = 2.0 * np.sqrt(3.0) / 9.0
    fixed_c_value = 2.0 * np.sqrt(3.0) / 9.0  # 如果你想让 c_i 激活项的值不同可修改
    t_steps = args.t_steps
    sims_per_point = args.sims_per_point
    n_workers = args.n_workers

    out_plot_dir = args.out_plot_dir
    out_csv_dir = args.out_csv_dir
    os.makedirs(out_plot_dir, exist_ok=True)
    os.makedirs(out_csv_dir, exist_ok=True)

    combos = [(s, mu_e, sigma_e)]
    if args.max_combos is not None:
        combos = combos[:args.max_combos]

    if args.dry_run:
        print("Dry run: planned runs")
        print(" combos:", combos)
        print(" sigma_d_values:", sigma_d_values)
        print(" phi_list:", phi_list)
        print(" sims_per_point:", sims_per_point)
        return

    for idx, (s_val, mu_e_val, sigma_e_val) in enumerate(combos, 1):
        print(f"[{idx}/{len(combos)}] Running s={s_val}, mu_e={mu_e_val}, sigma_e={sigma_e_val}")
        csv_path, plot_path, means, ses = run_scan_for_combo(
            s_val, mu_e_val, sigma_e_val,
            mu_d_values, sigma_d_values,
            mu_c, sigma_c, rho_d,
            phi_list, t_steps, sims_per_point,
            out_plot_dir, out_csv_dir,
            n_workers=n_workers,
            fixed_c_value=fixed_c_value,
            phi_is_fraction=not args.phi_absolute,
            generate_new_d_each_run=True
        )
        print("  Saved CSV:", csv_path)
        print("  Saved plot:", plot_path)

    print("All done.")

if __name__ == '__main__':
    main()