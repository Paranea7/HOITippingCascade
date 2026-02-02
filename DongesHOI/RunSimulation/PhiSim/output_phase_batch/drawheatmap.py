#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import glob

# ==========================================
# 🚀 PRL 风格全局设置
# ==========================================
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    "xtick.bottom": True,
    "ytick.left": True,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.grid": False,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05
})


def load_csv(csv_file):
    """
    加载 CSV 文件数据。
    兼容左上角带有 'sigma_e\\sigma_d' 等标签的情况。
    """
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty.")

    # ========================================================
    # [修改点] 解析 mu_d (X轴)
    # 强制跳过第一行的第一个元素 (rows[0][0])，因为它通常是标签或空
    # ========================================================
    try:
        # rows[0][1:] 表示从第一行的第二个元素开始读取
        # 增加 strip() 去除可能存在的空格
        mu_d = np.array([float(x) for x in rows[0][1:] if x.strip() != ''])
    except ValueError as e:
        print(f"DEBUG: Header row content: {rows[0]}")
        raise ValueError(f"Error parsing header (mu_d): {e}")

    mu_e = []
    grid = []

    if len(rows) < 2:
        raise ValueError("CSV file contains no data rows.")

    # 从第二行开始遍历数据
    for r_idx, r in enumerate(rows[1:], start=2):
        if not r:  # 跳过空行
            continue

        try:
            # 第一列是 Y 轴的值 (mu_e)
            val_e = float(r[0])
            mu_e.append(val_e)

            # 后面的列是数据网格
            # 同样过滤掉可能的空字符串
            data_row = [float(x) for x in r[1:] if x.strip() != '']

            # 简单检查长度是否匹配
            if len(data_row) != len(mu_d):
                print(f"⚠️ Warning in file {os.path.basename(csv_file)} at row {r_idx}: "
                      f"Data length ({len(data_row)}) does not match header length ({len(mu_d)}).")
                # 如果数据多了或少了，这里可以决定是截断还是补零，目前选择截断以防报错
                min_len = min(len(data_row), len(mu_d))
                data_row = data_row[:min_len]

            grid.append(data_row)

        except ValueError as e:
            print(f"⚠️ Skipping malformed row {r_idx}: {r} (Error: {e})")
            continue

    return mu_d, np.array(mu_e), np.array(grid)


def plot_heatmap_from_csv(csv_file, out_png):
    """
    从 CSV 文件绘制热力图，并保存。
    """
    try:
        mu_d, mu_e, grid = load_csv(csv_file)

        # 如果数据读取为空（可能所有行都报错了），直接返回
        if len(mu_e) == 0 or len(grid) == 0:
            print(f"❌ Error: No valid data extracted from {csv_file}")
            return

    except Exception as e:
        print(f"❌ Error loading {csv_file}: {e}")
        return

    # 计算宽高比
    data_width = mu_d[-1] - mu_d[0]
    data_height = mu_e[-1] - mu_e[0]

    fig_width_prl = 3.375
    if data_width == 0 or data_height == 0:
        aspect_ratio = 1.0
    else:
        aspect_ratio = abs(data_height / data_width)

    # 限制宽高比，防止图像过扁或过长
    if aspect_ratio > 2.5: aspect_ratio = 2.5
    if aspect_ratio < 0.4: aspect_ratio = 0.4

    fig_height_prl = fig_width_prl * aspect_ratio * 0.9

    fig, ax = plt.subplots(figsize=(fig_width_prl, fig_height_prl))

    im = ax.imshow(
        grid,
        origin='lower',
        aspect='auto',
        extent=[mu_d[0], mu_d[-1], mu_e[0], mu_e[-1]],
        cmap='viridis',
        vmin=0,
        vmax=0.7,
    )

    # 设置标签
    ax.set_xlabel(r"$\sigma_d$")
    ax.set_ylabel(r"$\sigma_e$")

    cbar = plt.colorbar(im, ax=ax, label=r"Tipping Rate $\phi$", pad=0.05, aspect=20)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()

    output_dir = os.path.dirname(out_png)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"✅ Saved: {out_png}")


# ==========================================
# 🚀 批量处理主程序
# ==========================================
if __name__ == "__main__":
    current_dir = os.getcwd()

    # 查找所有 CSV 文件
    csv_pattern = os.path.join(current_dir, "*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f"No CSV files found in {current_dir}")
    else:
        print(f"Found {len(csv_files)} CSV files. Starting processing...")

        output_folder = os.path.join(current_dir, "plots")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for i, csv_file in enumerate(csv_files):
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            output_filename = f"{base_name}.png"
            output_path = os.path.join(output_folder, output_filename)

            print(f"[{i + 1}/{len(csv_files)}] Processing {base_name}...")
            plot_heatmap_from_csv(csv_file, output_path)

        print("\nAll done! Check the 'plots' folder.")