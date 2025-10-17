#!/usr/bin/env python3
"""
完整仿真程序（已按要求修改 c_i 的生成方式）
- 对参数组合 s in {5,10,30,50}, mu_e in {0.0,0.1,0.2,0.3}, sigma_e in {0.0,0.1,0.3,0.5,1.0}
  的所有组合执行仿真。（当前 main() 中 s_values/mu_e_values/sigma_e_values 已改为示例值）
- 对每个组合扫描 mu_d in {0.2,0.3,0.5} 与 sigma_d in np.linspace(0,1,21)。
- 每个 sigma_d 下并行运行 simulations_per_sigma 次 single_simulation，计算平均存活率与标准误（SE）。
- 修改点：c_i 的生成方式：
    * 如果 phi_0 <= 1 且 phi_0 >= 0：解释为比例，count = int(round(phi_0*s))
    * 如果 phi_0 > 1：解释为非零元素的绝对个数 count = min(s, int(round(phi_0)))
    * 随机选择 count 个索引，将这些位置的 c_i 设为 fixed_c_value，其余为 0。
    * 每次 single_simulation 随机重采样这些索引（独立样本）。
- d_ji, e_ijk 的生成与原逻辑一致（带有 s 缩放）。
- 注意：此脚本仍然可能非常耗时，请根据机器资源调整 t_steps 与 simulations_per_sigma。
"""

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# ---------- 核心模型函数 ----------
def generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, phi_0=0.1, fixed_c_value=None):
    """
    生成模型参数：
    - c_i: 根据 phi_0 与 fixed_c_value 构造。
        如果 phi_0 在 [0,1] 范围内，按比例处理 count = round(phi_0*s)；
        如果 phi_0 > 1，按绝对个数处理 count = min(s, round(phi_0))。
        若 count == 0，则 c_i 全为 0。
        如果 fixed_c_value 为 None，则默认使用 mu_c 作为非零值。
        注意：每次调用该函数会随机选择不重复的索引进行赋值（即每次仿真随机分配哪些个体拥有该 c）。
    - d_ij, d_ji, e_ijk: 与原程序相同，按 s 缩放生成正态随机矩阵/张量。
    返回：c_i, d_ij, d_ji, e_ijk
    """
    # 处理 fixed_c_value 默认
    if fixed_c_value is None:
        fixed_c_value = mu_c

    # 计算 count（非零 c 的个数）
    if phi_0 is None:
        count = 0
    else:
        try:
            phi_val = float(phi_0)
        except Exception:
            phi_val = 0.0
        if phi_val <= 0.0:
            count = 0
        elif 0.0 < phi_val <= 1.0:
            count = int(round(phi_val * s))
        else:
            # phi_0 > 1，视为绝对个数
            count = int(round(phi_val))
            if count > s:
                count = s
    # 构造 c_i
    c_i = np.zeros(s, dtype=float)
    if count > 0:
        # 随机选择 count 个索引（不重复）
        idx = np.random.choice(s, size=count, replace=False)
        c_i[idx] = float(fixed_c_value)

    # d 与 e 的生成保持原样（与原脚本相同的缩放）
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

def single_simulation(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps, phi_0=0.0, fixed_c_value=None):
    """
    执行一次完整的随机参数生成 + 动力学模拟，返回单次的存活率（float）。
    注意：这里每次调用 generate_parameters 时会随机选择哪些索引为非零 c_i（独立抽样）。
    """
    c_i, _, d_ji, e_ijk = generate_parameters(s, mu_c, sigma_c, mu_d, sigma_d, rho_d, mu_e, sigma_e, phi_0, fixed_c_value)
    x_init = np.full(s, -0.6, dtype=float)
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
                            n_workers=None,
                            phi_0=0.0, fixed_c_value=None):
    """
    对单一 (s, mu_e, sigma_e) 组合：
    - 遍历每个 mu_d 与每个 sigma_d，使用并行计算 simulations_per_sigma 次 single_simulation。
    - phi_0 与 fixed_c_value 会传递到 single_simulation 中以控制 c_i 的生成。
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
            # 构造参数元组列表，注意增加了 phi_0 与 fixed_c_value
            args = [(s, mu_c, sigma_c, mu_d, float(sigma_d), rho_d, mu_e, sigma_e, t_steps, phi_0, fixed_c_value)
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
    prefix = f"s_{s}_mue_{mu_e}_sigmae_{sigma_e}_phi_{phi_0}"
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

    # ----- phi_0 与 fixed_c_value 设置 -----
    # 你要求 phi_0 = 16。由于 phi_0 超过 1，脚本将其视为“非零 c_i 的绝对个数”为 16（如果 s < 16 则取 s）。
    # 如果你原本意思是比例（如 0.16），请改为 phi_0 = 0.16。
    phi_0 = 16
    fixed_c_value = 2.0 * np.sqrt(3.0) / 9.0  # 你给出的数值

    # ----- 扫描参数集合 -----
    # 为示例演示，这里保留你原来的扫描设置的一个子集（可以按需改回原始组合）
    s_values = [5,10,20,30,50,100]         # 你原来脚本有 [5,10,30,50]，这里只跑 s=50（与原 main 保持一致）
    mu_e_values = [0.01,0.02,0.05,0.1,0.2,0.3,0.5]
    sigma_e_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    mu_d_values = [0.1,0.2, 0.3,0.4, 0.5,0.6,0.7,0.8,0.9,1.0]
    sigma_d_values = np.linspace(0.0, 1.0, 21)

    # ----- 运行参数（请根据机器资源调整） -----
    # WARNING: Defaults below 与你原脚本相同（可能非常重）。测试时建议调小 t_steps 与 simulations_per_sigma。
    t_steps = 5000
    simulations_per_sigma = 5

    # 并行 worker 数，默认使用 cpu_count()（可改为 cpu_count()-1）
    n_workers = max(1, cpu_count())

    out_plot_dir = "outputplotdnew"
    out_csv_dir = "outputcsvdnew"
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
                print(f"[{cnt}/{total}] Starting run for s={s}, mu_e={mu_e}, sigma_e={sigma_e}, phi_0={phi_0}")
                csv_path, plot_path = run_one_parameter_combo(
                    s, mu_e, sigma_e,
                    mu_d_values, sigma_d_values,
                    mu_c, sigma_c, rho_d,
                    t_steps, simulations_per_sigma,
                    out_plot_dir, out_csv_dir,
                    n_workers=n_workers,
                    phi_0=phi_0,
                    fixed_c_value=fixed_c_value
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