#!/usr/bin/env python3
"""
精简后的仿真程序（带并行、CSV/PNG 输出）
动力学方程：
    dx_i/dt = r*x_i*(1 - x_i/K)
              - B * x_i / (A + x_i)
              + d_scale * sum_j m_ij x_j
              + e_scale * sum_{jk} n_ijk x_j x_k

说明:
- 对 s, mu_e, sigma_e 的组合进行扫描（main 中可配置）
- 对每个组合扫描 mu_d_values 与 sigma_d_values，并对每个 sigma_d 并行运行 simulations_per_sigma 次模拟
- 每组结果保存为 CSV 与 PNG（保存在 out_csv_dir / out_plot_dir）
- 注意：默认大规模运行会很耗资源，请根据机器调整 t_steps, simulations_per_sigma, n_workers
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# ---------- 参数生成与动力学 ----------
def generate_couplings(s, mu_d, sigma_d, rho_d, mu_e, sigma_e):
    """
    生成二体耦合矩阵 m_ij 与三体耦合张量 n_ijk
    按照原代码缩放习惯：mu_d/s, sigma_d/s; mu_e/s^2, sigma_e/s^2
    """
    m_ij = np.random.normal(mu_d / s, sigma_d / s, (s, s))
    # 保持矩阵对称相关（使用 rho_d 生成一个相关版本）
    m_ji = rho_d * m_ij + np.sqrt(max(0.0, 1 - rho_d**2)) * np.random.normal(mu_d / s, sigma_d / s, (s, s))
    n_ijk = np.random.normal(mu_e / s**2, sigma_e / s**2, (s, s, s))
    return m_ji, n_ijk

def dynamics_rk4(s, r, K, B, A, d_scale, e_scale, m_ji, n_ijk, x_init, t_steps, dt=0.01):
    """
    用 RK4 积分器求解动力学，返回最终状态向量 x（长度 s）
    dx_i/dt = r*x_i*(1 - x_i/K) - B*x_i/(A + x_i) + d_scale * (m_ji @ x)_i + e_scale * einsum(n_ijk, x, x)
    对 A + x 做数值保护，避免除以接近 0。
    """
    x = x_init.astype(float).copy()
    eps = 1e-8  # 防止除零

    for _ in range(t_steps):
        def compute_dx(xloc):
            # logistic growth
            dx = r * xloc * (1.0 - xloc / K)
            # saturating loss term (protected denominator)
            denom = A + xloc
            denom = np.where(denom <= eps, eps, denom)
            dx = dx - B * xloc / denom
            # linear coupling
            dx = dx + d_scale * np.dot(m_ji, xloc)
            # quadratic coupling
            dx = dx + e_scale * np.einsum('ijk,j,k->i', n_ijk, xloc, xloc)
            return dx

        k1 = compute_dx(x)
        k2 = compute_dx(x + 0.5 * dt * k1)
        k3 = compute_dx(x + 0.5 * dt * k2)
        k4 = compute_dx(x + dt * k3)
        x += (k1 + 2*k2 + 2*k3 + k4) * dt / 6.0

        # 可选数值修正：防止出现 NaN/inf
        if np.any(~np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

    return x

def survival_rate(final_states):
    """存活率定义为最终状态 > 0 的分量比例"""
    return float(np.sum(final_states > 0)) / float(len(final_states))

def single_simulation(s, mu_d, sigma_d, rho_d, mu_e, sigma_e,
                      t_steps, r, K, B, A, d_scale, e_scale, x_init):
    """
    一次完整模拟：生成耦合，求解动力学，返回存活率（float）
    设计为可用于 Pool.starmap
    """
    m_ji, n_ijk = generate_couplings(s, mu_d, sigma_d, rho_d, mu_e, sigma_e)
    final = dynamics_rk4(s, r, K, B, A, d_scale, e_scale, m_ji, n_ijk, x_init, t_steps)
    return survival_rate(final)

# ---------- I/O 与绘图 ----------
def save_csv(sigma_d_values, mu_d_values, means, ses, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}.csv")
    header = ['sigma_d']
    for mu in mu_d_values:
        header += [f"mean_mu_d_{mu}", f"se_mu_d_{mu}"]
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, sigma_d in enumerate(sigma_d_values):
            row = [float(sigma_d)]
            for j in range(len(mu_d_values)):
                row.append(float(means[j][idx]))
                row.append(float(ses[j][idx]))
            writer.writerow(row)
    return out_path

def plot_save(sigma_d_values, mu_d_values, means, ses, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for j, mu in enumerate(mu_d_values):
        ax.errorbar(sigma_d_values, means[j], yerr=ses[j], fmt='o-', capsize=4, label=f"mu_d={mu}")
    ax.set_title("Survival Rate vs Sigma_d (SE)")
    ax.set_xlabel("sigma_d")
    ax.set_ylabel("survival rate")
    ax.grid(True)
    ax.legend()
    out_path = os.path.join(out_dir, f"{prefix}.png")
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return out_path

# ---------- 主执行函数 ----------
def run_one_combo(s, mu_e, sigma_e,
                  mu_d_values, sigma_d_values,
                  rho_d,
                  t_steps, simulations_per_sigma,
                  r, K, B, A, d_scale, e_scale,
                  x_init,
                  out_plot_dir, out_csv_dir,
                  n_workers=None):
    """
    针对单个(s, mu_e, sigma_e)组合扫描 mu_d_values 与 sigma_d_values。
    返回 csv_path, plot_path。
    """
    if n_workers is None:
        n_workers = max(1, cpu_count())

    means_all = []
    ses_all = []

    for mu_d in mu_d_values:
        means = []
        ses = []
        for sigma_d in sigma_d_values:
            # 构造参数元组列表用于并行
            args = [(s, mu_d, float(sigma_d), rho_d, mu_e, sigma_e,
                     t_steps, r, K, B, A, d_scale, e_scale, x_init)
                    for _ in range(simulations_per_sigma)]
            # 并行计算
            with Pool(processes=n_workers) as pool:
                results = pool.starmap(single_simulation, args)

            mean_val = float(np.mean(results))
            se_val = float(np.std(results, ddof=1) / np.sqrt(len(results))) if len(results) > 1 else 0.0

            means.append(mean_val)
            ses.append(se_val)

            del results

        means_all.append(means)
        ses_all.append(ses)

    prefix = f"s_{s}_mue_{mu_e}_sigmae_{sigma_e}"
    csv_path = save_csv(sigma_d_values, mu_d_values, means_all, ses_all, out_csv_dir, prefix)
    plot_path = plot_save(sigma_d_values, mu_d_values, means_all, ses_all, out_plot_dir, prefix)

    return csv_path, plot_path

def main():
    # ------------- 固定/扫描参数 -------------
    rho_d = 1.0  # 二体矩阵相关度（可调整或设为0以取消相关）
    s_values = [5,10,30,50]                # 群体规模列表（示例仅使用 s=50）
    mu_e_values = [0.0,0.03,0.05,0.1,0.3]            # 三体均值列表（示例）
    sigma_e_values = [0.0,0.2,0.3,0.5,1.0]         # 三体标准差列表（示例）

    mu_d_values = [0.2, 0.3, 0.5]  # 二体均值扫描
    sigma_d_values = np.linspace(0.0, 1.0, 21)

    # ------------- 运行参数（请根据机器资源调整） -------------
    t_steps = 2000       # 时间步数（测试时较小），正式运行可增大
    simulations_per_sigma = 100  # 每个 sigma_d 下的独立重复次数（测试用较小）
    n_workers = max(1, cpu_count())

    out_plot_dir = "output_plots"
    out_csv_dir = "output_csv"
    os.makedirs(out_plot_dir, exist_ok=True)
    os.makedirs(out_csv_dir, exist_ok=True)

    # ------------- 模型参数（可调整） -------------
    r = 1.0        # 内在增长率
    K = 1.0        # 承载力（K>0）
    B = 0.5        # 抑制项幅度（B>=0）
    A = 0.1        # 抑制项半饱和常数（A>0）
    d_scale = 1.0  # 二体耦合系数 d
    e_scale = 1.0  # 三体耦合系数 e

    # 初始条件（均一正值，通常更物理）
    x_init_value = 0.1

    generated = []
    total = len(s_values) * len(mu_e_values) * len(sigma_e_values)
    counter = 0
    for s in s_values:
        for mu_e in mu_e_values:
            for sigma_e in sigma_e_values:
                counter += 1
                print(f"[{counter}/{total}] Running s={s}, mu_e={mu_e}, sigma_e={sigma_e}")
                x_init = np.full(s, x_init_value, dtype=float)
                csv_path, plot_path = run_one_combo(
                    s, mu_e, sigma_e,
                    mu_d_values, sigma_d_values,
                    rho_d,
                    t_steps, simulations_per_sigma,
                    r, K, B, A, d_scale, e_scale,
                    x_init,
                    out_plot_dir, out_csv_dir,
                    n_workers=n_workers
                )
                generated.append((csv_path, plot_path))
                print("  Saved CSV:", csv_path)
                print("  Saved plot:", plot_path)

    print("Done. Generated files:")
    for c, p in generated:
        print(" ", c, p)

if __name__ == "__main__":
    main()