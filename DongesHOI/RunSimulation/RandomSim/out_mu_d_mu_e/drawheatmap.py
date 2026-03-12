#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import glob
from scipy.special import erfinv
from scipy.optimize import fsolve


# ==========================================
# 🚀 PNAS 风格全局设置
# ==========================================
def set_pnas_style():
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02
    })


set_pnas_style()


def load_csv(csv_file):
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        rows = list(reader)
    mu_d = np.array([float(x) for x in rows[0][1:] if x.strip() != ''])
    mu_e, grid = [], []
    for r in rows[1:]:
        if not r or r[0].strip() == '': continue
        mu_e.append(float(r[0]))
        grid.append([float(x) for x in r[1:] if x.strip() != ''][:len(mu_d)])
    return mu_d, np.array(mu_e), np.array(grid)


# ==========================================
# 🚀 噪声修正理论计算逻辑
# ==========================================
def solve_theoretical_boundary(mu_d_vec, Hc, mu_u, su, sd, se, phi_target=0.01):
    """
    通过数值求解自洽方程组得到考虑噪声 sigma 后的理论边界线
    """
    mu_e_boundary = []

    # 预计算常数项
    rhs_val = np.sqrt(2) * erfinv(2 * phi_target - 1)

    def equations(vars, md):
        me = vars[0]
        # 1. 假设一个 M，根据场方程计算 m, q
        # 但由于 M 也依赖 m, q，我们需要联立求解。
        # 为了稳定，我们直接解 M 和 me
        M = vars[1]
        m = 2 * phi_target - 1 + M / 2.0
        q = 1 + (2 * phi_target - 1) * M
        Gamma = np.sqrt(su ** 2 + sd ** 2 * q + se ** 2 * q ** 2)

        # 方程 1: M 满足概率分布定义的临界值
        eq1 = M - Hc - rhs_val * Gamma
        # 方程 2: M 满足平均场定义的自洽值
        eq2 = M - (mu_u + md * m + me * q)
        return [eq1, eq2]

    for md in mu_d_vec:
        # 初值估计：基于确定性极限的 me 和 M=Hc
        me_guess = (Hc * (1 - md / 2) - mu_u + md) / (1 - Hc)
        sol = fsolve(equations, [me_guess, Hc], args=(md,))
        mu_e_boundary.append(sol[0])

    return np.array(mu_e_boundary)


def plot_heatmap_from_csv(csv_file, out_png):
    try:
        mu_d, mu_e, grid = load_csv(csv_file)
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    fig_width = 3.42
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.85))

    # 1. 绘制热力图
    im = ax.imshow(grid, origin='lower', aspect='auto',
                   extent=[mu_d[0], mu_d[-1], mu_e[0], mu_e[-1]],
                   cmap='viridis', vmin=0, vmax=0.4)

    # 2. 计算考虑噪声的理论线
    # 参数设置
    params = {
        'Hc': 0.3849, 'mu_u': 0.0,
        'su': 0.1283, 'sd': 0.1, 'se': 0.3,
        'phi_target': 0.05  # 定义 phi 显著增长的起点
    }

    mu_d_fine = np.linspace(mu_d[0], mu_d[-1], 100)
    mu_e_theory = solve_theoretical_boundary(mu_d_fine, **params)

    # 3. 绘制理论曲线
    ax.plot(mu_d_fine, mu_e_theory, color='white', linestyle='--',
            linewidth=1.2, label=r'Theory $\phi \approx 0$ (with $\sigma$)')

    # 格式化
    ax.set_xlabel(r"Linear coupling $\mu_d$")
    ax.set_ylabel(r"Higher-order coupling $\mu_e$")
    ax.set_xlim(mu_d[0], mu_d[-1])
    ax.set_ylim(mu_e[0], mu_e[-1])

    cbar = plt.colorbar(im, ax=ax, pad=0.03, aspect=20)
    cbar.set_label(r"Tipping Rate $\phi$")
    ax.legend(loc='upper left', frameon=False)

    output_base = os.path.splitext(out_png)[0]
    fig.savefig(f"{output_base}.png", dpi=400)
    fig.savefig(f"{output_base}.pdf")
    plt.close(fig)
    print(f"✅ Processed with Noise Correction: {base_name}")


if __name__ == "__main__":
    current_dir = os.getcwd()
    csv_files = glob.glob(os.path.join(current_dir, "*.csv"))
    output_folder = os.path.join(current_dir, "plots_noise_corrected")
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    for csv_file in csv_files:
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        plot_heatmap_from_csv(csv_file, os.path.join(output_folder, base_name + ".png"))