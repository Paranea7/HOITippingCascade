#!/usr/bin/env python3
"""
只考虑三体相互作用的仿真程序

修改要点：
- 忽略二体耦合 d_ji，在动力学中仅保留 e_ijk 的三体项。
- mu_e 使用固定列表 miue = [0.02,0.03,0.05,0.1,0.2,0.3,0.5]
- 横坐标为 sigma_e，纵坐标为 survival rate（均值与 SE）
- 其它逻辑与原程序相似（并行重复、多 sigma_e 扫描、保存 CSV 与 PNG）
"""

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# ---------- 核心模型函数 ----------
def generate_parameters(s, mu_c, sigma_c, mu_e, sigma_e):
    """
    生成模型参数（仅三体耦合）：
    c_i: 控制参数向量（长度 s）
    e_ijk: 三体耦合张量（每分量尺度约为 mu_e/s^2, sigma_e/s^2）
    注：二体耦合已被忽略。
    """
    c_i = np.random.normal(mu_c, sigma_c, s)
    e_ijk = np.random.normal(mu_e / s**2, sigma_e / s**2, (s, s, s))
    return c_i, e_ijk

def dynamics_simulation(s, c_i, e_ijk, x_init, t_steps):
    """
    使用 RK4 积分器迭代动力学方程（忽略二体耦合）：
    dx/dt = -x^3 + x + c_i + einsum(e_ijk, x, x)
    返回最终状态向量 x（长度 s）
    """
    x = x_init.copy()
    dt = 0.01

    for _ in range(t_steps):
        def compute_dx(x_local):
            dx = -x_local**3 + x_local + c_i
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

def single_simulation(s, mu_c, sigma_c, mu_e, sigma_e, t_steps):
    """
    执行一次完整的随机参数生成 + 动力学模拟，返回单次的存活率（float）。
    该函数适合用于并行执行（multiprocessing.Pool.starmap）。
    """
    c_i, e_ijk = generate_parameters(s, mu_c, sigma_c, mu_e, sigma_e)
    x_init = np.full(s, -0.6, dtype=float)
    final_states = dynamics_simulation(s, c_i, e_ijk, x_init, t_steps)
    return calculate_survival_rate(final_states)

# ---------- I/O 与绘图辅助函数 ----------
def save_results_csv(sigma_e_values, mu_e_values, survival_means, survival_ses, out_dir, prefix):
    """
    保存 CSV。CSV 列格式：
      sigma_e, mean_mu_e_0.02, se_mu_e_0.02, mean_mu_e_0.03, se_mu_e_0.03, ...
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}.csv")
    header = ['sigma_e']
    for mu_e in mu_e_values:
        header.append(f"mean_mue_{mu_e}")
        header.append(f"se_mue_{mu_e}")

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, sigma_e in enumerate(sigma_e_values):
            row = [float(sigma_e)]
            for j in range(len(mu_e_values)):
                row.append(float(survival_means[j][idx]))
                row.append(float(survival_ses[j][idx]))
            writer.writerow(row)
    return out_path

def plot_and_save(sigma_e_values, mu_e_values, survival_means, survival_ses, out_dir, prefix):
    """
    绘图并保存 PNG。绘图后调用 plt.close() 释放内存。
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for j, mu_e in enumerate(mu_e_values):
        ax.errorbar(sigma_e_values, survival_means[j], yerr=survival_ses[j],
                    fmt='o-', capsize=4, label=f"mu_e={mu_e}")
    ax.set_title("Survival Rate vs Sigma_e (SE error bars) — only 3-body interactions")
    ax.set_xlabel("Sigma_e")
    ax.set_ylabel("Survival Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True)
    ax.legend()

    out_path = os.path.join(out_dir, f"{prefix}.png")
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)  # 关键：关闭 figure 释放内存
    return out_path

# ---------- 主执行逻辑 ----------
def run_one_parameter_combo(s, mu_e, sigma_e_values,
                            mu_c, sigma_c,
                            t_steps, simulations_per_sigma,
                            out_plot_dir, out_csv_dir,
                            n_workers=None):
    """
    对单一 (s, mu_e) 组合：
    - 遍历每个 sigma_e，使用并行计算 simulations_per_sigma 次 single_simulation。
    - 返回保存的 csv_path 与 plot_path（针对这个 mu_e 的一组 sigma_e），
      但调用者将以 mu_e 为曲线分组保存为单个 CSV/PNG，所以这里返回列表形式。
    """
    if n_workers is None:
        n_workers = max(1, cpu_count())

    means = []
    ses = []

    for sigma_e in sigma_e_values:
        args = [(s, mu_c, sigma_c, float(mu_e), float(sigma_e), t_steps)
                for _ in range(simulations_per_sigma)]
        with Pool(processes=n_workers) as pool:
            results = pool.starmap(single_simulation, args)
        mean_surv = float(np.mean(results))
        se_surv = float(np.std(results, ddof=1) / np.sqrt(len(results))) if len(results) > 1 else 0.0
        means.append(mean_surv)
        ses.append(se_surv)
        del results

    return means, ses

def main():
    # ----- 基本固定参数 -----
    mu_c = 0.0
    sigma_c = 2.0 * np.sqrt(3.0) / 9.0

    # ----- 扫描参数集合 -----
    s_values = [10,30,50]  # 你可以改为 [5,10,30,50] 等
    miue = [0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]  # 指定的 mu_e 列表
    # sigma_e_values 为横坐标（扫描）
    sigma_e_values = np.linspace(0.0, 1.0, 21)  # 例如 0..1 分成21点

    # ----- 运行参数（请根据机器资源调整） -----
    # 下面为较保守的默认值。若你有强算力可恢复更大 t_steps 与 simulations_per_sigma
    t_steps = 2000
    simulations_per_sigma = 200

    # 并行 worker 数，默认使用 cpu_count()-1 避免占满所有核心
    n_workers = max(1, cpu_count())

    out_plot_dir = "outputplot_3body"
    out_csv_dir = "outputcsv_3body"
    os.makedirs(out_plot_dir, exist_ok=True)
    os.makedirs(out_csv_dir, exist_ok=True)

    generated_files = []

    total = len(s_values) * len(miue)
    cnt = 0
    for s in s_values:
        for mu_e in miue:
            cnt += 1
            print(f"[{cnt}/{total}] Starting runs for s={s}, mu_e={mu_e}")
            # 对当前 mu_e 在所有 sigma_e 上并行采样
            means, ses = run_one_parameter_combo(
                s, mu_e, sigma_e_values,
                mu_c, sigma_c,
                t_steps, simulations_per_sigma,
                out_plot_dir, out_csv_dir,
                n_workers=n_workers
            )

            # 为便于查看，把每个 mu_e 的数据暂存到列表，稍后统一写入 CSV/绘图
            generated_files.append((s, mu_e, means, ses))

    # 将所有 mu_e 的数据合并保存为一个 CSV 与一张图（横坐标 sigma_e，曲线按 mu_e 分组）
    # 构造 survival_means 为 shape (len(miue), len(sigma_e_values))
    survival_means = [item[2] for item in generated_files]
    survival_ses = [item[3] for item in generated_files]

    prefix = f"s_{s_values[0]}_3body_mue_set"
    csv_path = save_results_csv(sigma_e_values, miue, survival_means, survival_ses, out_csv_dir, prefix)
    plot_path = plot_and_save(sigma_e_values, miue, survival_means, survival_ses, out_plot_dir, prefix)
    print("Saved CSV:", csv_path)
    print("Saved plot:", plot_path)

if __name__ == "__main__":
    main()