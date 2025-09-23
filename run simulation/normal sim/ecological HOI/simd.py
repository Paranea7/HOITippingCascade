#!/usr/bin/env python3
"""
完整仿真程序
- 对参数组合 s in {5,10,30,50}, mu_e in {0.0,0.1,0.2,0.3}, sigma_e in {0.0,0.1,0.3,0.5,1.0}
  的所有组合执行仿真。
- 对每个组合扫描 mu_d in {0.2,0.3,0.5} 与 sigma_d in np.linspace(0,1,21)。
- 每个 sigma_d 下并行运行 simulations_per_sigma 次 single_simulation，计算平均存活率与标准误（SE）。
- 将每个组合的结果保存为 CSV，绘图并保存为 PNG 到 outputplotd。绘图后关闭 figure 释放内存。
- 注意：默认参数较重，请根据机器调整 simulations_per_sigma、t_steps。
"""

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# ---------- 核心模型函数 ----------
def generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e):
    """
    生成模型参数：
    c_i: 控制参数向量（长度 s）
    d_ij: 二体耦合矩阵（未被使用，仅用于生成 d_ji）
    d_ji: 二体耦合矩阵的修正版本
    e_ijk: 三体耦合张量
    """
    c_i = np.random.normal(mu_c, sigma_c, s)
    # 将 mu_d 与 sigma_d 按规模 s 缩放（保持与你原始代码一致）
    d_ij = np.random.normal(mu_d / s, sigma_d / s, (s, s))
    d_ji = rho_d * d_ij + np.sqrt(max(0.0, 1 - rho_d**2)) * np.random.normal(mu_d / s, sigma_d / s, (s, s))
    e_ijk = np.random.normal(mu_e / s**2, sigma_e / s**2, (s, s, s))
    return c_i, d_ij, d_ji, e_ijk

def dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps):
    """
    使用四阶龙格-库塔（RK4）积分器迭代动力学方程：
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

def single_simulation(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps):
    """
    执行一次完整的随机参数生成 + 动力学模拟，返回单次的存活率（float）。
    该函数适合用于并行执行（multiprocessing.Pool.starmap）。
    """
    c_i, _, d_ji, e_ijk = generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e)
    x_init = np.full(s, 0.6, dtype=float)
    final_states = dynamics_simulation(s, c_i, d_ji, e_ijk, x_init, t_steps)
    return calculate_survival_rate(final_states)

# ---------- I/O 与绘图辅助函数 ----------
def save_results_csv(sigma_d_values, mu_d_values, survival_means, survival_ses, out_dir, prefix):
    """
    保存 CSV。CSV 列格式：
      sigma_d, mean_mu_d_0.2, se_mu_d_0.2, mean_mu_d_0.3, se_mu_d_0.3, ...
    """
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

def plot_and_save(sigma_d_values, mu_d_values, survival_means, survival_ses, out_dir, prefix):
    """
    绘图并保存 PNG。绘图后调用 plt.close() 释放内存。
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for j, mu_d in enumerate(mu_d_values):
        ax.errorbar(sigma_d_values, survival_means[j], yerr=survival_ses[j],
                    fmt='o-', capsize=4, label=f"mu_d={mu_d}")
    ax.set_title("Survival Rate vs Sigma_d (SE error bars)")
    ax.set_xlabel("Sigma_d")
    ax.set_ylabel("Survival Rate")
    ax.grid(True)
    ax.legend()

    out_path = os.path.join(out_dir, f"{prefix}.png")
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)  # 关键：关闭 figure 释放内存
    return out_path

# ---------- 主执行逻辑 ----------
def run_one_parameter_combo(s, mu_e, sigma_e,
                            mu_d_values, sigma_d_values,
                            mu_c, sigma_c, rho_d,
                            t_steps, simulations_per_sigma,
                            out_plot_dir, out_csv_dir,
                            n_workers=None):
    """
    对单一 (s, mu_e, sigma_e) 组合：
    - 遍历每个 mu_d 与每个 sigma_d，使用并行计算 simulations_per_sigma 次 single_simulation。
    - 返回保存的 csv_path 与 plot_path。
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    # 存放结果
    survival_means = []  # 对每个 mu_d: 长度为 len(sigma_d_values) 的均值列表
    survival_ses = []    # 对每个 mu_d: 长度为 len(sigma_d_values) 的 SE 列表

    # 为避免多次创建 Pool（在外层也可能创建），这里每次调用创建 Pool 以保证资源回收
    for mu_d in mu_d_values:
        means_for_mu_d = []
        ses_for_mu_d = []

        # 对每个 sigma_d 并行地运行多次 single_simulation
        for sigma_d in sigma_d_values:
            # 构造参数元组列表
            args = [(s, mu_c, sigma_c, mu_d, float(sigma_d), rho_d, mu_e, sigma_e, t_steps)
                    for _ in range(simulations_per_sigma)]

            # 使用 Pool 并行计算（在这里为每个 sigma_d 重建 Pool 可以避免长期占用大量内存，
            # 但也会增加开销。根据需要可改为在外层一次性创建 Pool。）
            with Pool(processes=n_workers) as pool:
                results = pool.starmap(single_simulation, args)

            # 计算均值与标准误（若 simulations_per_sigma <=1，则 SE 设为 0）
            mean_surv = float(np.mean(results))
            se_surv = float(np.std(results, ddof=1) / np.sqrt(len(results))) if len(results) > 1 else 0.0

            means_for_mu_d.append(mean_surv)
            ses_for_mu_d.append(se_surv)

            # 释放 results
            del results

        survival_means.append(means_for_mu_d)
        survival_ses.append(ses_for_mu_d)

    # 保存 CSV 与图像
    prefix = f"s_{s}_mue_{mu_e}_sigmae_{sigma_e}"
    csv_path = save_results_csv(sigma_d_values, mu_d_values, survival_means, survival_ses, out_csv_dir, prefix)
    plot_path = plot_and_save(sigma_d_values, mu_d_values, survival_means, survival_ses, out_plot_dir, prefix)

    # 释放大数组引用
    del survival_means, survival_ses

    return csv_path, plot_path

def main():
    # ----- 基本固定参数 -----
    mu_c = 0.0
    sigma_c = 2.0 * np.sqrt(3.0) / 9.0
    rho_d = 1.0

    # ----- 扫描参数集合 -----
    s_values = [30, 50, 100]
    mu_e_values = [0.0, 0.1, 0.2, 0.3]
    sigma_e_values = [0.0, 0.1, 0.3, 0.5, 1.0]

    mu_d_values = [0.2, 0.3, 0.5]
    sigma_d_values = np.linspace(0.0, 1.0, 21)

    # ----- 运行参数（请根据机器资源调整） -----
    # WARNING: Defaults below are heavy. For testing, set t_steps=500, simulations_per_sigma=50.
    t_steps = 5000
    simulations_per_sigma = 500

    # 并行 worker 数，默认使用 cpu_count()-1 避免占满所有核心
    n_workers = max(1, cpu_count() - 1)

    out_plot_dir = "outputplotd"
    out_csv_dir = "outputcsvd"
    os.makedirs(out_plot_dir, exist_ok=True)
    os.makedirs(out_csv_dir, exist_ok=True)

    # 记录所有生成文件
    generated_files = []

    total = len(s_values) * len(mu_e_values) * len(sigma_e_values)
    cnt = 0
    for s in s_values:
        for mu_e in mu_e_values:
            for sigma_e in sigma_e_values:
                cnt += 1
                print(f"[{cnt}/{total}] Starting run for s={s}, mu_e={mu_e}, sigma_e={sigma_e}")
                csv_path, plot_path = run_one_parameter_combo(
                    s, mu_e, sigma_e,
                    mu_d_values, sigma_d_values,
                    mu_c, sigma_c, rho_d,
                    t_steps, simulations_per_sigma,
                    out_plot_dir, out_csv_dir,
                    n_workers=n_workers
                )
                generated_files.append((csv_path, plot_path))
                print(f"  Saved CSV: {csv_path}")
                print(f"  Saved plot: {plot_path}")

    print("All parameter combinations finished.")
    print("Generated files:")
    for c, p in generated_files:
        print("  ", c, p)

if __name__ == "__main__":
    main()