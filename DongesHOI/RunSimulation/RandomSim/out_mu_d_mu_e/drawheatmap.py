#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import glob


# ==========================================
# 🚀 PNAS 风格全局设置 (Professional Style)
# ==========================================
def set_pnas_style():
    mpl.rcParams.update({
        "text.usetex": False,
        # PNAS 规范：优先使用无衬线字体 (Arial/Helvetica)
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],

        # 数学字体也需同步为无衬线风格
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",

        "font.size": 8,  # PNAS 标准字号 (8-9pt)
        "axes.labelsize": 9,
        "xtick.labelsize": 7,  # 刻度字号可略小
        "ytick.labelsize": 7,
        "legend.fontsize": 7,

        # PNAS 刻度规范：向外 (out)
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,  # 关闭顶部刻度
        "ytick.right": False,  # 关闭右侧刻度

        # 边框与线宽
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02
    })


set_pnas_style()


def load_csv(csv_file):
    """加载 CSV 文件数据 (逻辑保持兼容性)"""
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows: raise ValueError("CSV file is empty.")

    # 解析 X 轴 (sigma_d)
    mu_d = np.array([float(x) for x in rows[0][1:] if x.strip() != ''])

    mu_e = []
    grid = []

    for r in rows[1:]:
        if not r: continue
        try:
            mu_e.append(float(r[0]))
            data_row = [float(x) for x in r[1:] if x.strip() != '']
            grid.append(data_row[:len(mu_d)])  # 确保长度匹配
        except ValueError:
            continue

    return mu_d, np.array(mu_e), np.array(grid)


def plot_heatmap_from_csv(csv_file, out_png):
    """绘制 PNAS 风格热力图"""
    try:
        mu_d, mu_e, grid = load_csv(csv_file)
        if len(mu_e) == 0: return
    except Exception as e:
        print(f"❌ Error loading {csv_file}: {e}")
        return

    # PNAS 单栏标准宽度: 3.42 英寸
    fig_width = 3.42
    # 根据数据动态调整高度，但保持 PNAS 紧凑感
    aspect_ratio = abs((mu_e[-1] - mu_e[0]) / (mu_d[-1] - mu_d[0]))
    aspect_ratio = np.clip(aspect_ratio, 0.5, 1.2)
    fig_height = fig_width * aspect_ratio * 0.85

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 绘制热力图
    # 使用 matplotlib.colormaps 替代已弃用的 get_cmap
    cmap = mpl.colormaps['viridis']

    im = ax.imshow(
        grid,
        origin='lower',
        aspect='auto',
        extent=[mu_d[0], mu_d[-1], mu_e[0], mu_e[-1]],
        cmap=cmap,
        vmin=0,
        vmax=0.7,
    )

    # 设置 PNAS 风格标签 (变量用斜体)
    ax.set_xlabel(r"$\mu_d$")
    ax.set_ylabel(r"$\mu_e$")

    # 去掉冗余边框 (PNAS 风格可选)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Colorbar 优化
    cbar = plt.colorbar(im, ax=ax, pad=0.03, aspect=18)
    cbar.set_label(r"Tipping Rate $\phi$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)

    # 保存为 PDF (PNAS 投稿首选) 和 PNG
    output_base = os.path.splitext(out_png)[0]
    fig.savefig(f"{output_base}.png", dpi=400)
    fig.savefig(f"{output_base}.pdf")

    plt.close(fig)
    print(f"✅ Saved PNAS style: {output_base}.png/pdf")


# ==========================================
# 🚀 批量处理主程序
# ==========================================
if __name__ == "__main__":
    current_dir = os.getcwd()
    csv_files = glob.glob(os.path.join(current_dir, "*.csv"))

    if not csv_files:
        print("No CSV files found.")
    else:
        output_folder = os.path.join(current_dir, "plots_pnas")
        if not os.path.exists(output_folder): os.makedirs(output_folder)

        for csv_file in csv_files:
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            plot_heatmap_from_csv(csv_file, os.path.join(output_folder, base_name + ".png"))

        print("\nAll tasks completed in PNAS style.")