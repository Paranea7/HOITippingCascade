#!/usr/bin/env python3
"""
完整仿真程序（generate_parameters 不再生成 c_i）
- generate_parameters 只返回 d_ij, d_ji, e_ijk。
- c_i 的生成在 run_one_parameter_combo 中完成（并传入 single_simulation）。
"""

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# ---------- 核心模型函数 ----------
def generate_parameters(s, mu_d, sigma_d, rho_d, mu_e, sigma_e):
    """
    只生成耦合参数 d_ij, d_ji, e_ijk（不再生成 c_i）。
    - d_ij: 二体耦合（对角线设为0），按 mu_d/s, sigma_d/s 缩放
    - d_ji: 与 d_ij 相关的矩阵（使用 rho_d 生成相关噪声矩阵）
    - e_ijk: 三体耦合，满足 e_iii = 0 且对 j,k 对称（e_ijk = e_ikj）
    返回：d_ij, d_ji, e_ijk
    """
    # 生成 d_ij，缩放按 mu_d/s, sigma_d/s
    d_ij = np.random.normal(mu_d / s, sigma_d / s, (s, s))
    np.fill_diagonal(d_ij, 0.0)

    # 生成与 d_ij 相关的 d_ji：构造一个独立噪声矩阵并与 d_ij 做相关混合（保持矩阵形状）
    noise = np.random.normal(mu_d / s, sigma_d / s, (s, s))
    d_ji = rho_d * d_ij + np.sqrt(max(0.0, 1.0 - rho_d ** 2)) * noise
    np.fill_diagonal(d_ji, 0.0)

    # 生成 e_ijk，按 mu_e/s^2, sigma_e/s^2 缩放
    e_ijk = np.random.normal(mu_e / s ** 2, sigma_e / s ** 2, (s, s, s))

    # 设置 e_iii = 0
    for i in range(s):
        e_ijk[i, i, i] = 0.0

    # 确保 e_ijk = e_ikj（对 j,k 对称）
    for i in range(s):
        for j in range(s):
            for k in range(j + 1, s):
                val = 0.5 * (e_ijk[i, j, k] + e_ijk[i, k, j])
                e_ijk[i, j, k] = val
                e_ijk[i, k, j] = val

    return d_ij, d_ji, e_ijk

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

def single_simulation(s, c_i, mu_d, sigma_d, rho_d, mu_e, sigma_e, t_steps):
    """
    执行一次完整的动力学模拟（参数 c_i, d_ji, e_ijk 都应在外部生成并传入）。
    为保留原有缩放逻辑，这里仍在内部调用 generate_parameters 来生成 d_ij,d_ji,e_ijk
    但 generate_parameters 不再生成 c_i。
    返回单次的存活率（float）。
    """
    # 生成耦合参数（d_ij, d_ji, e_ijk）
    d_ij, d_ji, e_ijk = generate_parameters(s, mu_d, sigma_d, rho_d, mu_e, sigma_e)

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
    - c_i 的生成在这里进行（每次重复会重采样 c_i），generate_parameters 不再生成 c_i。
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    # 存放结果
    survival_means = []
    survival_ses = []

    for mu_d in mu_d_values:
        means_for_mu_d = []
        ses_for_mu_d = []

        for sigma_d in sigma_d_values:
            # 为每次重复生成不同的 c_i（独立抽样）
            c_list = []
            for _ in range(simulations_per_sigma):
                # 生成 c_i 根据 phi_0 语义（比例或绝对数）
                if phi_0 is None:
                    count = 0
                else:
                    phi_val = float(phi_0)
                    if phi_val <= 0.0:
                        count = 0
                    elif 0.0 < phi_val <= 1.0:
                        count = int(round(phi_val * s))
                    else:
                        count = int(round(phi_val))
                        if count > s:
                            count = s
                c_i = np.zeros(s, dtype=float)
                if count > 0:
                    idx = np.random.choice(s, size=count, replace=False)
                    c_i[idx] = float(fixed_c_value) if fixed_c_value is not None else float(mu_c)
                c_list.append(c_i)

            # 为每个 c_i 创建参数元组并并行运行 single_simulation
            args = [(s, c_list[i], mu_d, float(sigma_d), rho_d, mu_e, sigma_e, t_steps)
                    for i in range(simulations_per_sigma)]

            with Pool(processes=n_workers) as pool:
                results = pool.starmap(single_simulation, args)

            mean_surv = float(np.mean(results))
            se_surv = float(np.std(results, ddof=1) / np.sqrt(len(results))) if len(results) > 1 else 0.0

            means_for_mu_d.append(mean_surv)
            ses_for_mu_d.append(se_surv)

            del results
            del c_list

        survival_means.append(means_for_mu_d)
        survival_ses.append(ses_for_mu_d)

    # 保存 CSV 与图像
    prefix = f"s_{s}_mue_{mu_e}_sigmae_{sigma_e}_phi_{phi_0}"
    csv_path = save_results_csv(sigma_d_values, mu_d_values, survival_means, survival_ses, out_csv_dir, prefix)
    plot_path = plot_and_save(sigma_d_values, mu_d_values, survival_means, survival_ses, out_plot_dir, prefix)

    del survival_means, survival_ses

    return csv_path, plot_path

def main():
    # ----- 基本固定参数 -----
    rho_d = 1.0

    # ----- phi_0 与 fixed_c_value 设置 -----
    phi_0 = 0.16  # 视为比例 16%
    fixed_c_value = 2.0 * np.sqrt(3.0) / 9.0

    # ----- 扫描参数集合 -----
    s_values = [5,10,20,30,50,100]
    mu_e_values = [0.01,0.02,0.05,0.1,0.2,0.3,0.5]
    sigma_e_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    mu_d_values = [0.1,0.2, 0.3,0.4, 0.5,0.6,0.7,0.8,0.9,1.0]
    sigma_d_values = np.linspace(0.0, 1.0, 21)

    # ----- 运行参数（请根据机器资源调整） -----
    t_steps = 700
    simulations_per_sigma = 20

    n_workers = max(1, cpu_count())

    out_plot_dir = "outputplotdnew"
    out_csv_dir = "outputcsvdnew"
    os.makedirs(out_plot_dir, exist_ok=True)
    os.makedirs(out_csv_dir, exist_ok=True)

    generated_files = []

    total = len(s_values) * len(mu_e_values) * len(sigma_e_values)
    cnt = 0

    # 注意：下面使用了 mu_c 和 sigma_c 变量，但在此脚本中 mu_c, sigma_c 并未在 main 中定义（原脚本中有）。
    # 为保持兼容，你需要在此处定义 mu_c, sigma_c（即 c 的默认统计信息），否则会报 NameError。
    # 如果你不再需要 mu_c, sigma_c，可删去这行引用并相应调整 single_simulation/c_i 生成逻辑。
    mu_c = 0.0
    sigma_c = 2 * np.sqrt(3) / 9

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