#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl  # 引入 matplotlib 模块以访问 rcParams
import os  # 引入 os 模块用于创建目录

# ==========================================
# 🚀 PRL 风格全局设置 (已整合所有修改)
# ==========================================
mpl.rcParams.update({
    "text.usetex": False,  # 设为 False 以避免 LaTeX 环境问题
    "font.family": "serif",  # 使用衬线字体
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],  # 优先使用 Times New Roman
    "mathtext.fontset": "stix",  # 使用 STIX 字体渲染公式，效果最像 LaTeX
    "font.size": 10,  # 全局基础字号 (PRL 正文约 10pt)
    "axes.labelsize": 10,  # 轴标签字号
    "xtick.labelsize": 8,  # x轴刻度字号
    "ytick.labelsize": 8,  # y轴刻度字号
    "legend.fontsize": 8,  # 图例字号 (如果需要)
    "xtick.direction": "in",  # 刻度向内
    "ytick.direction": "in",  # 刻度向内

    # --- [关键修改] 刻度只显示在左侧和底部 ---
    "xtick.top": False,  # 关闭顶部刻度线
    "ytick.right": False,  # 关闭右侧刻度线
    "xtick.bottom": True,  # 开启底部刻度线
    "ytick.left": True,  # 开启左侧刻度线
    # ----------------------------------------

    "axes.linewidth": 0.8,  # 轴线宽度 (PRL 通常较细)
    "xtick.major.width": 0.8,  # 主要刻度线宽度
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,  # 主要刻度线长度 (PRL 通常较短)
    "ytick.major.size": 3,
    "axes.grid": False,  # 默认不显示网格线

    "figure.dpi": 300,  # 高 DPI 预览 (最终出版可能需要 600)
    "savefig.bbox": "tight",  # 保存时去除多余白边
    "savefig.pad_inches": 0.05  # 紧凑保存，留一点点边距
})


def load_csv(csv_file):
    """
    加载 CSV 文件数据。
    CSV 格式：
    第一行: mu_d_0, mu_d_1, ...
    其他行: mu_e_i, grid_i_0, grid_i_1, ...
    """
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # 确保 mu_d 列表不为空
    if not rows[0][1:]:
        raise ValueError("CSV header for mu_d is empty or malformed.")
    mu_d = np.array([float(x) for x in rows[0][1:]])

    mu_e = []
    grid = []

    # 确保有数据行
    if len(rows) < 2:
        raise ValueError("CSV file contains no data rows for mu_e and grid.")

    for r in rows[1:]:
        if not r:  # 跳过空行
            continue
        # 确保 mu_e 值存在
        if not r[0]:
            raise ValueError(f"Empty mu_e value in row: {r}")
        mu_e.append(float(r[0]))

        # 确保 grid 数据存在且长度与 mu_d 匹配
        if len(r[1:]) != len(mu_d):
            raise ValueError(f"Grid data length mismatch in row {r[0]}. Expected {len(mu_d)}, got {len(r[1:])}")
        grid.append([float(x) for x in r[1:]])

    return mu_d, np.array(mu_e), np.array(grid)


def plot_heatmap_from_csv(csv_file, out_png):
    """
    从 CSV 文件绘制热力图，并保存为 PNG/PDF。
    """
    try:
        mu_d, mu_e, grid = load_csv(csv_file)
    except Exception as e:
        print(f"Error loading CSV file {csv_file}: {e}")
        return

    # PRL 单栏图宽度通常约为 3.375 英寸 (约 8.57 cm)
    # 对于热力图，宽高比可以根据数据范围调整，这里给一个示例
    fig_width_prl = 3.375
    # 假设 mu_e 和 mu_d 的范围大致相当，可以尝试 4:3 或 1:1 的宽高比
    # 这里我们根据数据实际的 extent 比例来调整高度，让图看起来不被拉伸
    # 计算实际数据的宽高比
    data_width = mu_d[-1] - mu_d[0]
    data_height = mu_e[-1] - mu_e[0]
    if data_width == 0 or data_height == 0:  # 避免除以零
        aspect_ratio = 1.0
    else:
        aspect_ratio = data_height / data_width

    fig_height_prl = fig_width_prl * aspect_ratio * 0.8  # 乘以一个系数微调，为 colorbar 留出空间

    fig, ax = plt.subplots(figsize=(fig_width_prl, fig_height_prl))

    # 绘制热力图
    im = ax.imshow(
        grid,
        origin='lower',
        aspect='auto',  # 'auto' 会根据数据和图框自动调整，'equal' 会强制 x/y 轴单位长度一致
        extent=[mu_d[0], mu_d[-1], mu_e[0], mu_e[-1]],
        cmap='viridis',
        vmin=0,  # 明确设置最小值
        vmax=1,  # 明确设置最大值
    )

    # 设置轴标签，使用 LaTeX 风格的符号
    ax.set_xlabel(r"$\mu_d$")
    ax.set_ylabel(r"$\mu_e$")

    # PRL 中图表标题通常在图注中，这里注释掉，如果需要可以取消注释并调整字号
    # ax.set_title("Survival rate", fontsize=10)

    # 添加 Colorbar
    # 调整 colorbar 的位置和大小，使其更紧凑
    cbar = plt.colorbar(im, ax=ax, label="Survival Rate", pad=0.05, aspect=20)
    cbar.ax.tick_params(labelsize=8)  # Colorbar 刻度标签字号

    # ---------------- 辅助虚线接口（你自己填坐标） ----------------
    # 示例：在 mu_d = 0 画一条竖虚线
    # ax.axvline(x=0.0, color='gray', linestyle='--', linewidth=0.8) # 调整颜色和线宽
    # 示例：在 mu_e = 0 画一条横虚线
    # ax.axhline(y=0.0, color='gray', linestyle='--', linewidth=0.8) # 调整颜色和线宽
    # ------------------------------------------------------------

    fig.tight_layout()  # 自动调整子图参数，使之填充整个图表区域

    # 确保输出目录存在
    output_dir = os.path.dirname(out_png)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig.savefig(out_png, dpi=300)  # 保存为高分辨率图像
    plt.close(fig)
    print(f"Saved PRL-style heatmap to: {out_png}")


# ==========================================
# 🚀 示例用法
# ==========================================
if __name__ == "__main__":
    # 创建一个虚拟的 CSV 文件用于演示
    dummy_csv_file = "dummy_heatmap_data.csv"
    dummy_output_png = "prl_style_heatmap.png"

    # 确保输出目录存在
    if not os.path.exists("output_prl_style"):
        os.makedirs("output_prl_style")
    dummy_output_png = os.path.join("output_prl_style", dummy_output_png)

    # 生成一些示例数据
    mu_d_vals = np.linspace(-1, 1, 20)
    mu_e_vals = np.linspace(-0.5, 0.5, 15)
    dummy_grid = np.zeros((len(mu_e_vals), len(mu_d_vals)))

    for i, me in enumerate(mu_e_vals):
        for j, md in enumerate(mu_d_vals):
            # 模拟一些数据，例如距离中心的生存率
            dummy_grid[i, j] = 0.5 + 0.5 * np.exp(-(md ** 2 + me ** 2) / 0.5)

    with open(dummy_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + mu_d_vals.tolist())  # 第一行是空的，然后是 mu_d 值
        for i, me in enumerate(mu_e_vals):
            writer.writerow([me] + dummy_grid[i].tolist())  # mu_e 值，然后是该行的 grid 数据

    print(f"Dummy CSV file '{dummy_csv_file}' created.")

    # 调用绘图函数
    try:
        plot_heatmap_from_csv(dummy_csv_file, dummy_output_png)
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        print("\nCommon issues:")
        print(
            "1. If 'Failed to process string with tex because latex could not be found' appears, ensure mpl.rcParams['text.usetex'] = False.")
        print(
            "2. If 'findfont: Font family ['Times New Roman'] not found' appears, ensure Times New Roman is installed on your system or choose another serif font.")
    finally:
        # 清理虚拟 CSV 文件
        if os.path.exists(dummy_csv_file):
            os.remove(dummy_csv_file)
            print(f"Dummy CSV file '{dummy_csv_file}' removed.")